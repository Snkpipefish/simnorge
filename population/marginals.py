"""Bygg marginalfordelinger per kommune fra strukturlaget — Modul 2.

Henter de marginalene IPF skal tilpasses mot:
- (kjønn × aldersgruppe)  fra 07459  — definerer total N (16+) og alder/kjønn
- (kjønn × utdanning)     fra 09429  — utdanningsmiks (16+), reskalert til N
- (økonomisk status)      fra 12944  — lavinntektsandel -> {lavinntekt, øvrig}

Kommune-skalar kontekst (ikke en IPF-akse, men festes på hver person):
- median husholdningsinntekt fra 06944.

Prikking/manglende celler håndteres EKSPLISITT: hver manglende celle logges og
føres i ``issues``. Vi imputerer aldri stille; mangler en kritisk margin, blir
kommunen markert som hoppet over (ikke bygget med gjettede tall).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from data import SSBClient
from .schema import (
    AGE_LABELS, ECON_LABELS, EDU_CODES, EDU_LABELS, SEX_CODES, SEX_LABELS,
    SHAPE, age_band,
)

logger = logging.getLogger("simnorge.populasjon")


@dataclass
class KommuneMarginals:
    kommune: str
    year: str
    n_total: int
    sex_age: np.ndarray                 # (2,4) tellinger
    sex_edu: np.ndarray                 # (2,6) tellinger, reskalert til n_total
    econ: np.ndarray                    # (2,)  [lavinntekt, øvrig]
    median_income: float | None
    lowincome_share: float | None
    econ_available: bool = True
    issues: list[str] = field(default_factory=list)
    skipped: bool = False

    def note(self, msg: str) -> None:
        self.issues.append(msg)
        logger.warning("Kommune %s: %s", self.kommune, msg)


def _latest_year(ssb: SSBClient, table_id: str) -> str:
    return ssb.variable_codes(table_id)["Tid"][-1]


def build_marginals(
    ssb: SSBClient,
    kommune: str,
    *,
    year: str | None = None,
) -> KommuneMarginals:
    year = year or _latest_year(ssb, "07459")

    km = KommuneMarginals(
        kommune=kommune, year=year,
        n_total=0, sex_age=np.zeros((2, 4)), sex_edu=np.zeros((2, 6)),
        econ=np.zeros(2), median_income=None, lowincome_share=None,
    )

    _fill_sex_age(ssb, km, year)
    if km.skipped:
        return km
    _fill_sex_edu(ssb, km, year)
    _fill_econ(ssb, km)
    _fill_median_income(ssb, km, year)
    return km


# --------------------------------------------------------------------------- #
# 07459 — befolkning etter kjønn × alder (1-årig -> bånd)                      #
# --------------------------------------------------------------------------- #
def _fill_sex_age(ssb: SSBClient, km: KommuneMarginals, year: str) -> None:
    codes = ssb.variable_codes("07459")
    ages = [a for a in codes["Alder"] if a.isdigit() and int(a) >= 16]
    df = ssb.fetch("07459", {
        "Region": [km.kommune], "Kjonn": ["1", "2"],
        "Alder": ages, "ContentsCode": ["Personer1"], "Tid": [year],
    })
    pricked = df[df["missing"]]
    if not pricked.empty:
        km.note(f"07459: {len(pricked)} prikkede alderskategorier — "
                f"alder/kjønn-marginen er ufullstendig, hopper over kommunen.")
        km.skipped = True
        return

    mat = np.zeros((2, 4))
    sex_idx = {"1": 0, "2": 1}
    band_idx = {b: i for i, b in enumerate(AGE_LABELS)}
    for _, r in df.iterrows():
        b = age_band(int(r["Alder"]))
        if b is None:
            continue
        mat[sex_idx[r["Kjonn"]], band_idx[b]] += r["value"]
    km.sex_age = mat
    km.n_total = int(round(mat.sum()))
    if km.n_total == 0:
        km.note("07459: total 16+-befolkning = 0, hopper over.")
        km.skipped = True


# --------------------------------------------------------------------------- #
# 09429 — personer 16+ etter kjønn × utdanningsnivå                            #
# --------------------------------------------------------------------------- #
def _fill_sex_edu(ssb: SSBClient, km: KommuneMarginals, year: str) -> None:
    edu_year = year if year in ssb.variable_codes("09429")["Tid"] else _latest_year(ssb, "09429")
    df = ssb.fetch("09429", {
        "Region": [km.kommune], "Nivaa": list(EDU_CODES),
        "Kjonn": ["1", "2"], "ContentsCode": ["Personer"], "Tid": [edu_year],
    })
    pricked = df[df["missing"]]
    if not pricked.empty:
        cells = pricked[["Kjonn", "Nivaa"]].values.tolist()
        km.note(f"09429: {len(pricked)} prikkede utdanningsceller {cells} satt til 0 "
                f"(eksplisitt logget, ikke imputert).")

    mat = np.zeros((2, 6))
    sex_idx = {"1": 0, "2": 1}
    edu_idx = {c: i for i, c in enumerate(EDU_CODES)}
    for _, r in df.iterrows():
        v = 0.0 if r["missing"] else float(r["value"])
        mat[sex_idx[r["Kjonn"]], edu_idx[r["Nivaa"]]] += v

    if mat.sum() <= 0:
        km.note("09429: ingen utdanningsdata, bruker uniform utdanningsmiks (logget).")
        mat = np.ones((2, 6))
    # Reskaler utdanningsmarginen PER KJØNN slik at dens kjønnstotaler er like
    # 07459 sine. Begge marginene deler kjønnsaksen; uten denne forsoningen blir
    # de inkonsistente og IPF konvergerer ikke. (To SSB-kilder med ulik
    # referanse/avrunding gir litt ulike kjønnstall.)
    sex_totals = km.sex_age.sum(axis=1)          # (2,) menn, kvinner fra 07459
    edu_row_sums = mat.sum(axis=1)               # (2,)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(edu_row_sums > 0, sex_totals / edu_row_sums, 0.0)
    km.sex_edu = mat * scale[:, None]


# --------------------------------------------------------------------------- #
# 12944 — vedvarende lavinntekt -> økonomisk status                           #
# --------------------------------------------------------------------------- #
def _fill_econ(ssb: SSBClient, km: KommuneMarginals) -> None:
    interval = _latest_year(ssb, "12944")
    try:
        df = ssb.fetch("12944", {
            "Region": [km.kommune], "Alder": ["999A"],
            "ContentsCode": ["EUskalaSeksti"], "Tid": [interval],
        })
    except Exception as e:  # noqa: BLE001
        km.note(f"12944: oppslag feilet ({e}); økonomisk status utelatt.")
        km.econ_available = False
        return

    if df.empty or bool(df["missing"].all()):
        km.note(f"12944: lavinntektsandel mangler/prikket for {interval}; "
                f"økonomisk status utelatt fra IPF (logget, ikke imputert).")
        km.econ_available = False
        return

    share = float(df.iloc[0]["value"]) / 100.0
    km.lowincome_share = share
    low = share * km.n_total
    km.econ = np.array([low, km.n_total - low])


# --------------------------------------------------------------------------- #
# 06944 — median husholdningsinntekt (skalar kontekst)                         #
# --------------------------------------------------------------------------- #
def _fill_median_income(ssb: SSBClient, km: KommuneMarginals, year: str) -> None:
    inc_year = year if year in ssb.variable_codes("06944")["Tid"] else _latest_year(ssb, "06944")
    try:
        df = ssb.fetch("06944", {
            "Region": [km.kommune], "HusholdType": ["0000"],
            "ContentsCode": ["SamletInntekt"], "Tid": [inc_year],
        })
    except Exception as e:  # noqa: BLE001
        km.note(f"06944: median inntekt utilgjengelig ({e}).")
        return
    if df.empty or bool(df["missing"].all()):
        km.note("06944: median inntekt prikket/mangler (logget).")
        return
    km.median_income = float(df.iloc[0]["value"])
