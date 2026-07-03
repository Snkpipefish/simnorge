"""Kalibreringskorreksjon av LLM-skjevhet — PLAN punkt 47/48.

Prompten FORSØKER å hindre at LLM-en demper mot midten («IKKE legg alt på ett
parti»), men uten en post-hoc korreksjon lært på fasit vet vi ikke om den
lyktes. Denne modulen lærer en enkel, gjennomsiktig korreksjon per parti på
backtest-residualene:

    korrigert_p = a_p + b_p · predikert_p        (b_p >= 0, monoton)

deretter klippes negative andeler til 0 og vektoren renormaliseres til 100.
To skjevhetstyper fanges: systematisk nivåfeil per parti (a_p) og demping/
overdrivelse av variasjon (b_p != 1 — demping mot midten gir b_p > 1 i
korreksjonen).

ÆRLIGHETSREGLER (håndhevet/dokumentert):
- Korreksjonen læres på residualer fra ETT valgpar og brukes på et ANNET —
  eller, når bare ett par finnes, estimeres gevinsten med kommune-kryss-
  validering (``cv_corrected_mae``): korreksjonen for en kommune læres alltid
  på ANDRE kommuners residualer. Aldri fit + evaluer på samme observasjoner.
- Korreksjonen er lært på VALG-fasit og gjelder partipreferanse. Den skal
  ALDRI brukes som skjult presisjon på frie spørsmål i spørretaben — der
  finnes ingen fasit å lære den på.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("simnorge.korreksjon")


@dataclass(frozen=True)
class AffineCorrection:
    """Per-parti affin korreksjon. ``a``/``b`` i prosentpoeng-rommet."""

    parties: tuple[str, ...]
    a: dict[str, float]
    b: dict[str, float]
    n_fit: int                      # antall (kommune)-observasjoner bak fitten

    def apply(self, pred: pd.DataFrame) -> pd.DataFrame:
        """Korriger en prediksjons-DataFrame [kommune, party, pred_pct].

        Partier uten lært korreksjon passerer uendret. Negative andeler
        klippes til 0 og hver kommune renormaliseres til sum 100."""
        out = pred.copy()
        out["pred_pct"] = [
            self.a.get(p, 0.0) + self.b.get(p, 1.0) * v
            for p, v in zip(out["party"], out["pred_pct"])
        ]
        out["pred_pct"] = out["pred_pct"].clip(lower=0.0)
        tot = out.groupby("kommune")["pred_pct"].transform("sum")
        out["pred_pct"] = np.where(tot > 0, out["pred_pct"] / tot * 100.0, 0.0)
        return out

    def summary(self) -> str:
        rows = [f"Affin korreksjon (n={self.n_fit} kommuner):"]
        for p in self.parties:
            rows.append(f"  {p:8s} korrigert = {self.a[p]:+5.2f} + {self.b[p]:.2f}·pred")
        return "\n".join(rows)


def fit_correction(pred: pd.DataFrame, actual: pd.DataFrame,
                   parties: list[str]) -> AffineCorrection:
    """Lær per-parti OLS på (pred_pct -> faktisk pct) over kommuner.

    ``pred``: [kommune, party, pred_pct], ``actual``: [kommune, party, pct].
    Monotoni-vakt: b_p klemmes til >= 0. Degenerert variasjon (alle
    prediksjoner like) gir b=1 og ren nivåkorreksjon."""
    m = pred.merge(actual, on=["kommune", "party"], how="inner").dropna(
        subset=["pred_pct", "pct"])
    a, b = {}, {}
    n_fit = m["kommune"].nunique()
    for p in parties:
        g = m[m["party"] == p]
        if len(g) < 3:
            logger.warning("Korreksjon %s: bare %d observasjoner — identitet brukes.",
                           p, len(g))
            a[p], b[p] = 0.0, 1.0
            continue
        x, y = g["pred_pct"].to_numpy(), g["pct"].to_numpy()
        vx = float(np.var(x))
        if vx < 1e-9:
            a[p], b[p] = float(np.mean(y) - np.mean(x)), 1.0
            continue
        slope = float(np.cov(x, y, bias=True)[0, 1] / vx)
        slope = max(0.0, slope)                       # monotoni-vakt
        a[p] = float(np.mean(y) - slope * np.mean(x))
        b[p] = slope
    return AffineCorrection(parties=tuple(parties), a=a, b=b, n_fit=n_fit)


def cv_corrected_mae(pred: pd.DataFrame, actual: pd.DataFrame,
                     parties: list[str], *, folds: int = 4,
                     seed: int = 0) -> dict[str, float]:
    """Ærlig gevinstmåling med kommune-kryssvalidering.

    Korreksjonen for hver fold læres KUN på de andre foldenes kommuner.
    Returnerer {'raw': MAE uten, 'corrected': MAE med, 'n': antall kommuner}.
    Bruk denne når bare ett valgpar finnes; med to par: fit på det ene,
    evaluer på det andre."""
    kommuner = sorted(set(pred["kommune"]) & set(actual["kommune"]))
    if len(kommuner) < folds:
        folds = max(2, len(kommuner) // 2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(kommuner))
    fold_of = {kommuner[idx]: i % folds for i, idx in enumerate(order)}

    merged = pred.merge(actual, on=["kommune", "party"], how="inner").dropna(
        subset=["pred_pct", "pct"])
    raw_err, cor_err = [], []
    for f in range(folds):
        tr = [k for k in kommuner if fold_of[k] != f]
        te = [k for k in kommuner if fold_of[k] == f]
        if not te or not tr:
            continue
        corr = fit_correction(pred[pred["kommune"].isin(tr)],
                              actual[actual["kommune"].isin(tr)], parties)
        te_pred = pred[pred["kommune"].isin(te)]
        te_corr = corr.apply(te_pred)
        base = merged[merged["kommune"].isin(te)]
        raw_err.extend((base["pred_pct"] - base["pct"]).abs().tolist())
        c = te_corr.merge(actual, on=["kommune", "party"], how="inner").dropna(
            subset=["pred_pct", "pct"])
        cor_err.extend((c["pred_pct"] - c["pct"]).abs().tolist())
    return {"raw": float(np.mean(raw_err)), "corrected": float(np.mean(cor_err)),
            "n": len(kommuner)}
