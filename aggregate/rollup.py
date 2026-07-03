"""Vektet opprulling persona → kommune → fylke → nasjonalt — Modul 6.

Returnerer FORDELINGER med usikkerhetsbånd, ikke bare punktestimat. Per-kommune-
utslag bevares så spørretaben kan dele geografisk. Fylke = de to første sifrene
i kommunekoden (2024-vintage).

Usikkerhetsbåndet kommer fra en bootstrap over personaene: vi resampler personas
(med vekt) og ser hvor mye den vektede fordelingen varierer. Det fanger at en
kommune representeres av et endelig antall personas hver med intern spredning —
ærlig usikkerhet, ikke falsk presisjon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Distribution:
    """En fordeling over kategorier med usikkerhetsbånd (lav/høy persentil)."""

    categories: list[str]
    point: np.ndarray            # punktestimat (andeler, sum 1)
    lo: np.ndarray               # nedre bånd
    hi: np.ndarray               # øvre bånd
    weight: float                # total vekt (innbyggere) bak fordelingen

    def as_dict(self) -> dict:
        return {c: {"andel": float(self.point[i]), "lo": float(self.lo[i]),
                    "hi": float(self.hi[i])} for i, c in enumerate(self.categories)}


def _weighted_dist(per_unit: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vektet gjennomsnitt av enhets-fordelinger -> aggregert fordeling."""
    w = weights / weights.sum() if weights.sum() else weights
    return (per_unit * w[:, None]).sum(axis=0)


def bootstrap_distribution(
    per_unit: np.ndarray,           # (n_units, n_cat) fordeling per enhet (persona)
    weights: np.ndarray,            # (n_units,)
    categories: list[str],
    *,
    n_boot: int = 400,
    ci: float = 0.90,
    seed: int = 0,
) -> Distribution:
    """Vektet aggregert fordeling + bootstrap-bånd over enhetene."""
    point = _weighted_dist(per_unit, weights)
    n = len(weights)
    if n <= 1:
        return Distribution(categories, point, point.copy(), point.copy(), float(weights.sum()))
    rng = np.random.default_rng(seed)
    p = weights / weights.sum()
    boots = np.empty((n_boot, per_unit.shape[1]))
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True, p=p)
        boots[b] = per_unit[idx].mean(axis=0)
    alpha = (1 - ci) / 2
    lo = np.quantile(boots, alpha, axis=0)
    hi = np.quantile(boots, 1 - alpha, axis=0)
    return Distribution(categories, point, lo, hi, float(weights.sum()))


# --------------------------------------------------------------------------- #
# Hierarkisk opprulling                                                       #
# --------------------------------------------------------------------------- #
def fylke_of(kommune: str) -> str:
    return kommune[:2]


def rollup(
    commune_dists: dict[str, Distribution],
    commune_weight: dict[str, float],
) -> dict[str, dict[str, Distribution]]:
    """Rull kommune-fordelinger opp til fylke og nasjonalt, vektet med
    befolkning. Returnerer {'kommune':{...}, 'fylke':{...}, 'nasjonalt':{...}}.
    Bånd på høyere nivå kommer fra befolkningsvektet kombinasjon av kommune-
    punktene (bånd-bredden propageres konservativt som vektet snitt)."""
    out: dict[str, dict[str, Distribution]] = {"kommune": dict(commune_dists), "fylke": {}, "nasjonalt": {}}
    if not commune_dists:
        return out
    cats = next(iter(commune_dists.values())).categories

    def combine(codes: list[str]) -> Distribution:
        w = np.array([commune_weight[c] for c in codes], float)
        pts = np.array([commune_dists[c].point for c in codes])
        los = np.array([commune_dists[c].lo for c in codes])
        his = np.array([commune_dists[c].hi for c in codes])
        ww = w / w.sum() if w.sum() else w
        return Distribution(cats, (pts * ww[:, None]).sum(0),
                            (los * ww[:, None]).sum(0), (his * ww[:, None]).sum(0),
                            float(w.sum()))

    by_fylke: dict[str, list[str]] = {}
    for c in commune_dists:
        by_fylke.setdefault(fylke_of(c), []).append(c)
    for f, codes in by_fylke.items():
        out["fylke"][f] = combine(codes)
    out["nasjonalt"]["alle"] = combine(list(commune_dists))
    return out
