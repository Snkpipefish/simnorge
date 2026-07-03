"""Backtest-harness — Modul 7 (skjelett).

Tar et byggeår, et målår og en pluggbar prediktor, kjører end-to-end og
returnerer et :class:`BacktestResult` med metrikker OG dekning.

Out-of-sample er strukturelt håndhevet:
1. ``build_year == target_year`` er forbudt (reises som feil).
2. Prediktoren får bare byggeårets panel (``fit``). Harnessen holder målårets
   panel privat og bruker det kun til scoring — prediktoren kan ikke se det.
   Det gjør det umulig å trene og validere på samme valg ved et uhell.

Skjelettet bruker normaliserte kommunekoder fra Klass-laget gjennomgående, og
rapporterer alltid dekning sammen med ethvert feilmål.
"""

from __future__ import annotations

import pandas as pd

from data import SSBClient, KlassClient
from .election_data import ElectionPanel, load_election_panel, municipality_code_set
from .metrics import (
    BIG_PARTIES, BacktestResult, CoverageReport, mae, pearson,
)
from .baselines import Predictor


def run_backtest(
    ssb: SSBClient,
    klass: KlassClient,
    *,
    build_year: str,
    target_year: str,
    predictor: Predictor,
    table_id: str = "08092",
    reference_year: str = "2024",
    big_parties: list[str] | None = None,
) -> BacktestResult:
    build_year, target_year = str(build_year), str(target_year)
    if build_year == target_year:
        raise ValueError(
            "Out-of-sample-brudd: build_year og target_year er samme valg "
            f"({build_year}). Bygg og valider ALDRI på samme valg."
        )
    big_parties = big_parties or BIG_PARTIES

    fit = load_election_panel(ssb, klass, build_year,
                              table_id=table_id, reference_year=reference_year)
    test = load_election_panel(ssb, klass, target_year,
                               table_id=table_id, reference_year=reference_year)

    parties = sorted(test.shares["party"].unique())
    target_kommunes = sorted(test.kommunes)

    # Prediktoren ser KUN fit. (Strukturell out-of-sample-garanti.)
    preds = predictor(fit, target_kommunes, parties)

    return _score(
        fit=fit, test=test, preds=preds, predictor_name=predictor.name,
        build_year=build_year, target_year=target_year,
        reference_year=reference_year, big_parties=big_parties,
        total_country=len(municipality_code_set(klass, reference_year)),
    )


def _score(
    *, fit: ElectionPanel, test: ElectionPanel, preds: pd.DataFrame,
    predictor_name: str, build_year: str, target_year: str,
    reference_year: str, big_parties: list[str], total_country: int,
) -> BacktestResult:
    # Slå sammen prediksjon og fasit på (kommune, parti).
    actual = test.shares.rename(columns={"pct": "actual_pct"})
    merged = preds.merge(actual, on=["kommune", "party"], how="inner")

    scored_kommunes = sorted(merged["kommune"].unique())

    # Hvilke målkommuner manglet prediksjon? (Typisk: kommuner som finnes i
    # målåret, men ikke kunne predikeres fra byggeåret.)
    no_pred = sorted(test.kommunes - set(scored_kommunes))

    coverage = CoverageReport(
        reference_year=reference_year,
        total_country_kommunes=total_country,
        target_measurable=len(test.kommunes),
        scored=len(scored_kommunes),
        dropped_target_split=dict(test.dropped_splits),
        dropped_build_split=dict(fit.dropped_splits),
        dropped_no_prediction=no_pred,
    )

    # Per-parti MAE og kommune-korrelasjon over de scorede kommunene.
    per_party_mae: dict[str, float] = {}
    per_party_corr: dict[str, float] = {}
    for party, grp in merged.groupby("party"):
        per_party_mae[party] = mae(grp["pred_pct"], grp["actual_pct"])
        per_party_corr[party] = pearson(grp["pred_pct"], grp["actual_pct"])

    def avg(parties: list[str]) -> float:
        vals = [per_party_mae[p] for p in parties if p in per_party_mae]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    big = [p for p in per_party_mae if p in big_parties]
    rest = [p for p in per_party_mae if p not in big_parties]

    # Per-kommune MAE (snitt over partier) — for kart og diagnose senere.
    merged["abs_err"] = (merged["pred_pct"] - merged["actual_pct"]).abs()
    per_kommune = (merged.groupby("kommune")["abs_err"].mean()
                   .reset_index().rename(columns={"abs_err": "mae"})
                   .sort_values("mae", ascending=False, ignore_index=True))

    return BacktestResult(
        build_year=build_year, target_year=target_year,
        predictor_name=predictor_name, reference_year=reference_year,
        coverage=coverage,
        per_party_mae=dict(sorted(per_party_mae.items())),
        per_party_corr=per_party_corr,
        mae_big=avg(big), mae_rest=avg(rest), mae_overall=avg(list(per_party_mae)),
        big_parties=big_parties, per_kommune=per_kommune,
    )
