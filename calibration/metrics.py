"""Backtest-metrikker og dekningsrapportering — Modul 7.

Kjerneregel (fra brukeren): et MAE-tall skal ALDRI stå alene. Hver
oppsummering parrer feilmålet med en dekningslinje — "dekker X av Y kommuner"
— slik at vi alltid vet om resultatet gjelder hele landet eller bare nesten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# "Store partier" — de tradisjonelt største. Skjelettet skiller disse fra
# resten fordi MAE på små partier er liten i absolutte prosentpoeng og lett
# drukner signalet fra de store. Konfigurerbart i harnessen.
BIG_PARTIES = ["Ap", "Høyre", "FrP", "Sp"]


@dataclass
class CoverageReport:
    """Hvor mye av landet backtesten faktisk målte mot."""

    reference_year: str
    total_country_kommunes: int          # Y — alle kommuner i referanseårgangen
    target_measurable: int               # kommuner med faktisk resultat (etter norm)
    scored: int                          # X — kommuner faktisk scoret (pred ∩ fasit)
    dropped_target_split: dict[str, list[str]] = field(default_factory=dict)
    dropped_build_split: dict[str, list[str]] = field(default_factory=dict)
    dropped_no_prediction: list[str] = field(default_factory=list)

    @property
    def coverage_fraction(self) -> float:
        return self.scored / self.total_country_kommunes if self.total_country_kommunes else 0.0

    def line(self) -> str:
        return (f"dekker {self.scored} av {self.total_country_kommunes} kommuner "
                f"({100 * self.coverage_fraction:.1f} %)")

    def reasons(self) -> str:
        bits = []
        if self.dropped_target_split:
            bits.append(f"{len(self.dropped_target_split)} droppet i målåret "
                        f"(ikke-reverserbar splitt: {sorted(self.dropped_target_split)})")
        if self.dropped_build_split:
            bits.append(f"{len(self.dropped_build_split)} droppet i byggeåret "
                        f"(splitt: {sorted(self.dropped_build_split)})")
        if self.dropped_no_prediction:
            bits.append(f"{len(self.dropped_no_prediction)} uten prediksjon "
                        f"({sorted(self.dropped_no_prediction)[:6]}…)"
                        if len(self.dropped_no_prediction) > 6
                        else f"{len(self.dropped_no_prediction)} uten prediksjon "
                             f"({sorted(self.dropped_no_prediction)})")
        return "; ".join(bits) if bits else "ingen kommuner droppet"


def mae(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.mean(np.abs(pred.to_numpy() - actual.to_numpy())))


def pearson(pred: pd.Series, actual: pd.Series) -> float:
    """Kommune-korrelasjon for ett parti. NaN hvis prediksjonen er konstant
    (f.eks. nasjonalt-snitt-baseline har null geografisk variasjon — ærlig
    rapportert som udefinert)."""
    p, a = pred.to_numpy(), actual.to_numpy()
    # Bruk variasjonsbredde (max-min) for konstant-sjekken: eksakt 0 for en
    # konstant prediksjon, robust mot flyttalls-støy i np.std (~1e-16).
    if len(p) < 2 or np.ptp(p) < 1e-9 or np.ptp(a) < 1e-9:
        return float("nan")
    return float(np.corrcoef(p, a)[0, 1])


@dataclass
class BacktestResult:
    build_year: str
    target_year: str
    predictor_name: str
    reference_year: str
    coverage: CoverageReport
    per_party_mae: dict[str, float]
    per_party_corr: dict[str, float]
    mae_big: float
    mae_rest: float
    mae_overall: float
    big_parties: list[str]
    per_kommune: pd.DataFrame            # kommune, mae (snitt over partier)

    def summary(self) -> str:
        lines = [
            f"Backtest [{self.predictor_name}]  bygg {self.build_year} → mål "
            f"{self.target_year}  (ref {self.reference_year})",
            f"  {self.coverage.line()}  |  {self.coverage.reasons()}",
            f"  MAE store partier {self.big_parties}: {self.mae_big:.2f} pp",
            f"  MAE øvrige partier:              {self.mae_rest:.2f} pp",
            f"  MAE alle partier:                {self.mae_overall:.2f} pp",
            "  Per parti (MAE pp / kommune-korr):",
        ]
        for p in self.per_party_mae:
            corr = self.per_party_corr[p]
            corr_s = "  n/a" if np.isnan(corr) else f"{corr:+.2f}"
            tag = "*" if p in self.big_parties else " "
            lines.append(f"    {tag}{p:8s} {self.per_party_mae[p]:5.2f}   r={corr_s}")
        return "\n".join(lines)
