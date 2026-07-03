"""Lukk valg-backtest-loopen: motor-varianter vs baseliner — Modul 6.

Kjører tre motorvarianter side om side mot persistence og national_mean:
- engine_demographic_only : stedløs, lekkasjefri referanse
- engine_geo_no_name      : + sentralitet + inntektsnivå, UTEN kommunenavn (resonnement)
- engine_with_geography   : + kommunenavn (resonnement ELLER memorering)

Slik ser vi (a) om geografi-kontekst løfter, og (b) hvor mye av et eventuelt løft
som kan være memorering (bare navn-varianten løfter) vs ekte resonnement
(geo_no_name løfter like mye). Brutalt ærlig — ingen pynt.

Alle vektorer renormaliseres til de 9 partiene (sum 100) for samme basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import SSBClient, KlassClient
from population import build_population
from .election_data import load_election_panel, municipality_code_set
from .metrics import BIG_PARTIES, mae
from .engine_predictor import EnginePredictor, ENGINE_PARTIES, VARIANTS

logger = logging.getLogger("simnorge.backtest")


def _renorm9(shares: pd.DataFrame, communes: list[str]) -> pd.DataFrame:
    s = shares[shares["kommune"].isin(communes) & shares["party"].isin(ENGINE_PARTIES)]
    tot = s.groupby("kommune")["pct"].transform("sum")
    out = s.copy()
    out["pct"] = np.where(tot > 0, s["pct"] / tot * 100.0, 0.0)
    return out[["kommune", "party", "pct"]]


def _national_mean9(fit) -> dict[str, float]:
    votes = fit.shares.merge(fit.total_votes, on="kommune")
    votes["pv"] = votes["pct"] / 100 * votes["total_votes"]
    nat = votes[votes["party"].isin(ENGINE_PARTIES)].groupby("party")["pv"].sum()
    nat = nat / nat.sum() * 100.0
    return {p: float(nat.get(p, 0.0)) for p in ENGINE_PARTIES}


def _wide(df_long, value):
    return df_long.pivot(index="kommune", columns="party", values=value).reindex(columns=ENGINE_PARTIES)


def _score(pred_wide, actual_wide):
    per_party = {p: mae(pred_wide[p].fillna(0), actual_wide[p].fillna(0)) for p in ENGINE_PARTIES}
    big = [per_party[p] for p in ENGINE_PARTIES if p in BIG_PARTIES]
    return per_party, {"big": float(np.mean(big)), "all": float(np.mean(list(per_party.values())))}


def _demographic_deviation(ssb, communes, pop_year):
    def mix(code):
        p = build_population(ssb, code, year=pop_year).persons
        if len(p) == 0:
            return None
        return p.groupby(["aldersgruppe", "utdanning"], observed=True).size() / len(p)
    nat = mix("0"); devs = {}
    for c in communes:
        m = mix(c)
        if m is None:
            continue
        idx = nat.index.union(m.index)
        devs[c] = float(np.abs(nat.reindex(idx, fill_value=0) - m.reindex(idx, fill_value=0)).sum())
    return devs


@dataclass
class VariantsComparison:
    build_year: str
    target_year: str
    scored_communes: list[str]
    total_country: int
    methods: list[str]
    per_party: dict[str, dict[str, float]]
    overall: dict[str, dict[str, float]]
    by_deviation: dict
    cost: dict[str, str]

    def summary(self) -> str:
        L = [f"Valg-backtest — motorvarianter vs baseliner  bygg {self.build_year} → mål {self.target_year}",
             f"  dekker {len(self.scored_communes)} av {self.total_country} kommuner (delsett vi bygger)",
             "  MAE (prosentpoeng, 9-parti-basis):",
             f"    {'metode':24s} {'alle':>6s} {'store':>6s}   kostnad/skalering"]
        for m in self.methods:
            L.append(f"    {m:24s} {self.overall[m]['all']:6.2f} {self.overall[m]['big']:6.2f}   "
                     f"{self.cost.get(m, '')}")
        L.append("  Per parti (MAE pp): " + " / ".join(self.methods))
        for p in ENGINE_PARTIES:
            L.append(f"    {p:8s} " + " / ".join(f"{self.per_party[m][p]:5.2f}" for m in self.methods))
        L.append("  Brutt på demografisk avvik fra landssnittet (MAE alle):")
        for grp, md in self.by_deviation.items():
            L.append(f"    {grp:20s} (n={md['n']}): " +
                     " / ".join(f"{m}={md[m]:.2f}" for m in self.methods))
        return "\n".join(L)


def run_variants_backtest(
    ssb: SSBClient, klass: KlassClient, items, *,
    build_year: str, target_year: str, communes: list[str],
    pop_year: str = "2024", granularity: str = "full", max_workers: int = 8,
    variants: list[str] | None = None, turnout_weighting: bool = False,
) -> VariantsComparison:
    variants = variants or list(VARIANTS)
    fit = load_election_panel(ssb, klass, build_year)
    test = load_election_panel(ssb, klass, target_year)
    scored = sorted(c for c in communes if c in test.kommunes and c in fit.kommunes)

    actual = _wide(_renorm9(test.shares, scored), "pct")
    persistence = _wide(_renorm9(fit.shares, scored), "pct").reindex(actual.index)
    natvec = _national_mean9(fit)
    natmean = pd.DataFrame({p: natvec[p] for p in ENGINE_PARTIES}, index=actual.index)

    method_preds = {"persistence": persistence, "national_mean": natmean}
    cost = {"persistence": "(baseline)", "national_mean": "(baseline)"}

    for v in variants:
        pred = EnginePredictor(ssb, klass, items, variant=v, pop_year=pop_year,
                               granularity=granularity, communes=scored, max_workers=max_workers,
                               turnout_weighting=turnout_weighting)
        long = pred(fit, scored, ENGINE_PARTIES)
        method_preds[f"engine_{v}"] = _wide(long.rename(columns={"pred_pct": "pct"}), "pct").reindex(actual.index)
        scale = ("celler×kommuner (full nasjonal DYR)" if VARIANTS[v].name
                 else "celler×kontekst (full nasjonal billig)")
        cost[f"engine_{v}"] = f"{pred.n_unique_cells} kall, {scale}"

    methods = [f"engine_{v}" for v in variants] + ["persistence", "national_mean"]
    per_party, overall = {}, {}
    for m in methods:
        per_party[m], overall[m] = _score(method_preds[m], actual)

    devs = _demographic_deviation(ssb, scored, pop_year)
    med = float(np.median(list(devs.values()))) if devs else 0.0
    groups = {"høyt demogr. avvik": [c for c in scored if devs.get(c, 0) >= med],
              "nær landssnitt": [c for c in scored if devs.get(c, 0) < med]}
    by_dev = {}
    for grp, codes in groups.items():
        if not codes:
            continue
        md = {"n": len(codes)}
        for m in methods:
            pred = method_preds[m]
            errs = [abs(pred.loc[c, p] - actual.loc[c, p]) for c in codes for p in ENGINE_PARTIES
                    if c in pred.index and not np.isnan(pred.loc[c, p]) and not np.isnan(actual.loc[c, p])]
            md[m] = float(np.mean(errs)) if errs else float("nan")
        by_dev[grp] = md

    return VariantsComparison(
        build_year=build_year, target_year=target_year, scored_communes=scored,
        total_country=len(municipality_code_set(klass, "2024")), methods=methods,
        per_party=per_party, overall=overall, by_deviation=by_dev, cost=cost,
    )
