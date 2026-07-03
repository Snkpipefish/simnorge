"""Iterativ proporsjonal tilpasning (IPF) — Modul 2.

SSB gir oss marginalfordelingene per kommune (alder/kjønn, utdanning,
økonomisk status), men IKKE fellesfordelingen (krysstabellen). IPF
approksimerer fellesfordelingen ved å skalere en kontingenstabell iterativt
til den matcher hver margin.

VIKTIG — dette er en APPROKSIMASJON, ikke målt sannhet: IPF konvergerer mot
den fordelingen som matcher marginene og ellers er nærmest uniform (maksimal
entropi). Det tilsvarer en antakelse om betinget uavhengighet mellom aksene
gitt marginene. Den ekte fellesfordelingen kan ha samvariasjon vi ikke ser
(f.eks. at lavinntekt og lav utdanning henger tettere sammen lokalt enn
marginene alene tilsier). Vi gjør denne antakelsen bevisst og logger den, på
linje med de andre ekstrapoleringene i systemet.

IPFs minimumsgaranti: uansett hvor unøyaktig fellesfordelingen er, vil
marginene til resultatet matche inputmarginene (det verifiserer Modul 2-testen).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("simnorge.populasjon")

Margin = tuple[tuple[int, ...], np.ndarray]   # (akser, målmargin)


@dataclass
class IPFResult:
    table: np.ndarray            # tilpasset fellesfordeling (tellinger)
    iterations: int
    max_margin_dev: float        # største avvik mot en målmargin etter siste sveip


def _marginalize(table: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Summer tabellen ned på ``axes`` (i stigende rekkefølge)."""
    comp = tuple(i for i in range(table.ndim) if i not in axes)
    return table.sum(axis=comp) if comp else table.copy()


def _apply(table: np.ndarray, axes: tuple[int, ...], factor: np.ndarray) -> np.ndarray:
    shape = [1] * table.ndim
    for ax, size in zip(axes, factor.shape):
        shape[ax] = size
    return table * factor.reshape(shape)


def ipf(
    shape: tuple[int, ...],
    margins: list[Margin],
    *,
    init: np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> IPFResult:
    """Tilpass en tabell av form ``shape`` til ``margins``.

    ``margins`` er liste av (akser, målmargin) der akser er stigende
    dimensjonsindekser og målmargin har form lik tabellens størrelser langs
    de aksene. Totalsummen i alle marginer bør være lik (konsistens).
    """
    table = np.ones(shape, dtype=float) if init is None else init.astype(float).copy()

    last_dev = float("inf")
    for it in range(1, max_iter + 1):
        for axes, target in margins:
            current = _marginalize(table, axes)
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(current > 0, target / current, 0.0)
            table = _apply(table, axes, factor)

        dev = max(float(np.max(np.abs(_marginalize(table, axes) - target)))
                  for axes, target in margins)
        if dev < tol or abs(last_dev - dev) < tol * 1e-3:
            return IPFResult(table=table, iterations=it, max_margin_dev=dev)
        last_dev = dev

    logger.warning("IPF nådde max_iter=%d uten full konvergens (avvik=%.3g).",
                   max_iter, last_dev)
    return IPFResult(table=table, iterations=max_iter, max_margin_dev=last_dev)


def integerize(table: np.ndarray, n_total: int) -> np.ndarray:
    """Gjør en reell fellesfordeling om til heltalls celletall som summerer
    EKSAKT til ``n_total`` (største-rest-metoden). Deterministisk og
    reproduserbar."""
    flat = table.flatten()
    total = flat.sum()
    if total <= 0:
        return np.zeros_like(flat, dtype=int).reshape(table.shape)
    exact = flat / total * n_total
    floor = np.floor(exact).astype(int)
    remainder = int(n_total - floor.sum())
    if remainder > 0:
        frac = exact - floor
        # Stabil sortering -> reproduserbar fordeling av restene.
        order = np.argsort(-frac, kind="stable")
        floor[order[:remainder]] += 1
    return floor.reshape(table.shape)
