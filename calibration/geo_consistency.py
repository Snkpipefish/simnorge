"""Geografisk holdningskonsistens — Modul 7 (myk sjekk nr. 2, uten LLM).

PLAN.md antok at holdningsdata bare finnes nasjonalt. Det var litt for
pessimistisk: SSB 13798 (mening og mestring etter FYLKE) og 13799 (etter
SENTRALITET) er MÅLT voksen livskvalitet med geografi — samme undersøkelse
som seedet Modul 3 (13790-serien), men brutt geografisk i stedet for
demografisk.

Det gir en direkte test av kjernemekanikken uten ett eneste LLM-kall:
prediker fylkets/sentralitetsklassens livskvalitetsfordeling KUN fra
(a) nasjonale demografi-fordelinger (prior-LUT per alder×kjønn) og
(b) kommunens demografiske miks — og sammenlign med det SSB faktisk målte.

Dette er en test av ekstrapoleringslagets FØRSTE antakelse (demografi bærer
geografi). Forvent delvis treff: valg-backtesten viste at sted forklarer mye
utover demografi. Poenget er å MÅLE hvor mye, ærlig — ikke å pynte.

Merk skala-forbehold: prior-LUT-en er forsonet mot 16+-befolkningen (18-24
brukt for 16-24; 67-79/80+ vektet), mens 13798/13799 måler 18+. Avviket
logges, ikke skjules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from data import SSBClient, KlassClient
from engine.geo import load_centrality
from population.marginals import build_marginals
from population.schema import AGE_LABELS, SEX_LABELS

logger = logging.getLogger("simnorge.geokonsistens")

FYLKE_TABLE = "13798"
SENT_TABLE = "13799"
LEVEL_CC = ["AndelLavSkar", "AndelMidSkar", "AndelHoySkar"]
LEVELS = ["lav", "middels", "hoy"]


# --------------------------------------------------------------------------- #
# Prediksjon: prior-LUT × demografisk miks                                     #
# --------------------------------------------------------------------------- #
def kommune_predictions(ssb: SSBClient, items, kommuner: list[str], *,
                        pop_year: str, use_status: bool = True) -> dict[str, dict]:
    """Per kommune: predikert (lav, middels, høy) per livskvalitetsakse fra
    kjønn×alder-marginalen + (valgfritt) kommunens arbeidsmarkedsstatus-miks
    med målt status-tilt (13793 via 13563). use_status=False gir den rene
    demografi-prediksjonen (for å måle hva status-aksen bidrar med)."""
    from population.status import build_status_mix

    lk_items = [i for i in items if i.column.startswith("livskvalitet_")]
    out = {}
    for k in kommuner:
        km = build_marginals(ssb, k, year=pop_year)   # diskcachet
        if km.skipped or km.n_total <= 0:
            continue
        # sex_age er (kjønn, alder); LUT er (alder, kjønn).
        w = (km.sex_age / km.sex_age.sum()).T          # (4 alder, 2 kjønn)
        smix = build_status_mix(ssb, k, year=pop_year) if use_status else None
        preds = {}
        for it in lk_items:
            if (smix is not None and smix.available
                    and it.status_factor is not None):
                # P(l|a,s,st) ∝ lut[a,s,l]·faktor[l,st]; vektet over status-miks.
                tilted = it.lut[:, :, :, None] * it.status_factor[None, None, :, :]
                tilted = tilted / tilted.sum(axis=2, keepdims=True)
                # smix.probs er (alder, status) -> vekt inn per aldersbånd.
                per_as = np.einsum("aslt,at->asl", tilted, smix.probs)
                preds[it.column] = np.einsum("as,asl->l", w, per_as)
            else:
                preds[it.column] = np.einsum("as,asl->l", w, it.lut)
        out[k] = {"n": km.n_total, "pred": preds}
    return out


def _aggregate(pred_by_kommune: dict[str, dict], group_of: dict[str, str]) -> dict:
    """Vektet opprulling kommune -> gruppe (fylke/sentralitet)."""
    axes = None
    agg: dict[str, dict] = {}
    for k, d in pred_by_kommune.items():
        g = group_of.get(k)
        if g is None:
            continue
        if axes is None:
            axes = list(d["pred"])
        a = agg.setdefault(g, {"n": 0.0, **{ax: np.zeros(3) for ax in axes}})
        a["n"] += d["n"]
        for ax in axes:
            a[ax] += d["n"] * d["pred"][ax]
    for g, a in agg.items():
        for ax in (axes or []):
            a[ax] = a[ax] / a["n"]
    return agg


# --------------------------------------------------------------------------- #
# Fasit: målt fordeling per fylke / sentralitet                                #
# --------------------------------------------------------------------------- #
def measured_distributions(ssb: SSBClient, table: str, region_dim: str,
                           mening_codes: dict[str, str]) -> tuple[dict, str]:
    """-> {region: {akse: (3,) fordeling}}, år. Renormalisert; prikking -> NaN."""
    codes = ssb.variable_codes(table)
    year = codes["Tid"][-1]
    # Region/sentralitet er eliminerbar — må bes om eksplisitt, ellers
    # kollapser uttrekket til landstotalen.
    df = ssb.fetch(table, {region_dim: codes[region_dim],
                           "Mening": list(mening_codes),
                           "ContentsCode": LEVEL_CC, "Tid": [year]})
    out: dict[str, dict] = {}
    for region, sub in df.groupby(region_dim):
        d = {}
        for code, name in mening_codes.items():
            vec = np.full(3, np.nan)
            for i, cc in enumerate(LEVEL_CC):
                row = sub[(sub["Mening"] == code) & (sub["ContentsCode"] == cc)]
                if not row.empty and not bool(row.iloc[0]["missing"]):
                    vec[i] = float(row.iloc[0]["value"]) / 100.0
            if not np.isnan(vec).any() and vec.sum() > 0:
                d[f"livskvalitet_{name}"] = vec / vec.sum()
        if d:
            out[str(region)] = d
    return out, year


# --------------------------------------------------------------------------- #
# Sammenligning                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class GeoConsistencyResult:
    label: str
    year: str
    groups: list[str]
    axes: list[str]
    mae_pred_pp: float                  # predikert vs målt (alle grupper×akser×nivå)
    mae_flat_pp: float                  # nasjonal-konstant baseline vs målt
    corr_lav: float                     # korrelasjon av avvik fra nasjonalt, nivå=lav
    corr_hoy: float
    national_gap_pp: float              # |predikert nasjonalt - målt nasjonalt| (maks)

    def summary(self) -> str:
        return (f"[{self.label} {self.year}] {len(self.groups)} grupper × "
                f"{len(self.axes)} akser\n"
                f"  MAE predikert vs målt: {self.mae_pred_pp:.2f} pp "
                f"(flat nasjonal baseline: {self.mae_flat_pp:.2f} pp)\n"
                f"  korrelasjon av geo-avvik (lav/høy skår): "
                f"{self.corr_lav:+.2f} / {self.corr_hoy:+.2f}\n"
                f"  nasjonal forankring (maks gap): {self.national_gap_pp:.2f} pp")


def compare(pred: dict, measured: dict, national_measured: dict,
            label: str, year: str) -> GeoConsistencyResult:
    groups = sorted(set(pred) & set(measured))
    axes = sorted(set.intersection(*(set(measured[g]) for g in groups))) if groups else []

    pred_nat = {ax: np.zeros(3) for ax in axes}
    tot = sum(pred[g]["n"] for g in groups)
    for g in groups:
        for ax in axes:
            pred_nat[ax] += pred[g]["n"] * pred[g][ax] / tot

    err_pred, err_flat, dev_p, dev_m = [], [], {0: [], 2: []}, {0: [], 2: []}
    for g in groups:
        for ax in axes:
            m, p = measured[g][ax], pred[g][ax]
            nat_m = national_measured.get(ax)
            err_pred.extend(np.abs(p - m) * 100)
            if nat_m is not None:
                err_flat.extend(np.abs(nat_m - m) * 100)
                for lvl in (0, 2):
                    dev_p[lvl].append(p[lvl] - pred_nat[ax][lvl])
                    dev_m[lvl].append(m[lvl] - nat_m[lvl])

    def _corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) < 3 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    gap = max(float(np.max(np.abs(pred_nat[ax] - national_measured[ax]) * 100))
              for ax in axes if ax in national_measured)
    return GeoConsistencyResult(
        label=label, year=year, groups=groups, axes=axes,
        mae_pred_pp=float(np.mean(err_pred)),
        mae_flat_pp=float(np.mean(err_flat)) if err_flat else float("nan"),
        corr_lav=_corr(dev_p[0], dev_m[0]), corr_hoy=_corr(dev_p[2], dev_m[2]),
        national_gap_pp=gap,
    )


def run_geo_consistency(ssb: SSBClient, klass: KlassClient, items,
                        kommuner: list[str], *, pop_year: str = "2024",
                        ) -> tuple[GeoConsistencyResult, GeoConsistencyResult]:
    """Kjør begge sjekker: fylke (13798) og sentralitet (13799)."""
    from enrich.attitudes import MENING_ITEMS

    logger.info("Geo-konsistens: prediker fra prior×miks for %d kommuner "
                "(NB: prior er 16+-forsonet, fasit måler 18+).", len(kommuner))
    preds = kommune_predictions(ssb, items, kommuner, pop_year=pop_year)

    meas_f, year_f = measured_distributions(ssb, FYLKE_TABLE, "Region", MENING_ITEMS)
    national = meas_f.pop("0", {})
    fylke_of = {k: k[:2] for k in preds}
    res_f = compare(_aggregate(preds, fylke_of), meas_f, national,
                    "fylke (13798)", year_f)

    meas_s, year_s = measured_distributions(ssb, SENT_TABLE, "SentralitetKomm",
                                            MENING_ITEMS)
    national_s = meas_s.pop("Se00", national)
    sent = load_centrality(klass)
    sent_of = {k: f"Se{sent[k]}" for k in preds if k in sent}
    res_s = compare(_aggregate(preds, sent_of), meas_s, national_s,
                    "sentralitet (13799)", year_s)
    return res_f, res_s
