"""Valgdeltakelsesvekting fra 13360 + 13085 — Modul 7.

Problemet: uten vekting teller alle demografiske celler likt i kommune-
aggregatet, men de møter opp i svært ulik grad. To gradienter er dokumentert:
utdanning (2021: menn m/grunnskole 58.9 %, kvinner m/UH 88.2 %; 13360) og
alder — den STERKESTE gradienten (unge stemmer langt sjeldnere; 13085 gir
deltakelse per 1-årsalder × kjønn for stortingsvalg).

LUT-en er (kjønn × aldersbånd × utdanning). Ingen SSB-tabell krysser alle tre,
så aksene kombineres MULTIPLIKATIVT rundt kjønnstotalen:

    t(s, a, e) = t_alder(s, a) · t_utd(s, e) / t_total(s)      klippet [0.01, 0.99]

— en uavhengighetsantakelse (utdanningsgradienten antas lik i alle aldersbånd),
samme type ærlige, dokumenterte forenkling som resten av prosjektet.

Aldersbåndene aggregeres fra 1-årsaldre med NASJONALE befolkningsvekter
(07459, valgåret). 16–17-åringer har ikke stemmerett ved stortingsvalg og
inngår i 16-24-båndet med deltakelse 0 — personaene representerer bosatte,
ikke stemmeberettigede, så båndets effektive deltakelse skal reflektere det.

Lekkasjedisiplin: bruk BYGGEÅRETS deltakelse når målåret predikeres. Mangler
byggeåret i tabellene, returneres None og kallende kode faller tilbake til
uniform vekt — logget, aldri stille.

MÅLT (2021→2025, 8 kommuner): eksplisitt vekting gjorde LLM-motoren marginalt
SVAKERE — LLM-ens celleanslag er allerede implisitt deltakelsesvektet, så
vekting dobbelteller gradientene. Derfor av som standard i EnginePredictor.
"""

from __future__ import annotations

import logging

import numpy as np

from data import SSBClient
from population.schema import AGE_LABELS, EDU_LABELS, SEX_LABELS, age_band

logger = logging.getLogger("simnorge.deltakelse")

# Våre schema-etiketter -> SSB-koder.
SEX_TO_KJONN = {"mann": "1", "kvinne": "2"}
# NUS-nivå 4-5 (fagskole) ligger i SSBs gruppe "Videregående skolenivå" (3-5).
# Uoppgitt utdanning får totalnivået (TOT) — ingen bedre informasjon.
EDU_TO_NIVAA = {
    "grunnskole": "1-2",
    "videregaaende": "3-5",
    "fagskole": "3-5",
    "uh_kort": "6-8",
    "uh_lang": "6-8",
    "uoppgitt": "TOT",
}
VOTING_AGE = 18          # stortingsvalg; 16-17 inngår i 16-24-båndet med t=0

TurnoutLUT = dict[tuple[str, str, str], float]   # (kjønn, aldersbånd, utd) -> 0..1


def _edu_gradient(ssb: SSBClient, year: str):
    """13360: (kjønnskode, nivåkode) -> deltakelse, inkl. TOT per kjønn."""
    df = ssb.fetch("13360", {
        "Region": ["0"],
        "Kjonn": list(SEX_TO_KJONN.values()),
        "UtdNivaa": sorted(set(EDU_TO_NIVAA.values())),
        "ContentsCode": ["Valgdeltakelse"],
        "Tid": [year],
    })
    return {(r["Kjonn"], r["UtdNivaa"]): float(r["value"]) / 100.0
            for _, r in df.iterrows() if not r["missing"]}


def _age_band_turnout(ssb: SSBClient, year: str) -> dict[tuple[str, str], float]:
    """13085 (stortingsvalg, per 1-årsalder × kjønn) aggregert til våre
    aldersbånd med nasjonale befolkningsvekter (07459, samme år).
    16-17 teller med deltakelse 0 i 16-24-båndet."""
    # Alder-dimensjonene må filtreres eksplisitt — ellers eliminerer klienten dem.
    t = ssb.fetch("13085", {
        "Valgtype": ["2"], "Kjonn": list(SEX_TO_KJONN.values()),
        "Alder": [c for c in ssb.variable_codes("13085")["Alder"] if c != "999A"],
        "ContentsCode": ["Valgdeltakelse"], "Tid": [year],
    })
    t_age = {(r["Kjonn"], r["Alder"]): float(r["value"]) / 100.0
             for _, r in t.iterrows() if not r["missing"]}

    pop = ssb.fetch("07459", {"Region": ["0"], "Kjonn": list(SEX_TO_KJONN.values()),
                              "Alder": ssb.variable_codes("07459")["Alder"],
                              "Tid": [year]})
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, r in pop.iterrows():
        raw_age = r["Alder"].rstrip("+")
        try:
            age = int(raw_age)
        except ValueError:
            continue
        band = age_band(age)
        if band is None:
            continue
        if age < VOTING_AGE:
            turnout = 0.0
        else:
            # 13085 topper på '099+'; eldre aldre arver den.
            code = f"{min(age, 99):03d}" if age < 99 else "099+"
            turnout = t_age.get((r["Kjonn"], code))
            if turnout is None:
                continue
        acc = out.setdefault((r["Kjonn"], band), {"tw": 0.0, "w": 0.0})
        acc["tw"] += turnout * float(r["value"])
        acc["w"] += float(r["value"])
    return {k: v["tw"] / v["w"] for k, v in out.items() if v["w"] > 0}


def load_turnout(ssb: SSBClient, year: str) -> TurnoutLUT | None:
    """Nasjonal deltakelse per (kjønn, aldersbånd, utdanning) for ett valgår.

    Returnerer None (med logg) hvis året mangler i kildetabellene — kallende
    kode skal da bruke uniform vekt i stedet for å gjette."""
    year = str(year)
    for table in ("13360", "13085"):
        available = ssb.variable_codes(table)["Tid"]
        if year not in available:
            logger.warning("%s mangler valgåret %s (har %s) — deltakelsesvekting "
                           "deaktivert, uniform vekt brukes.", table, year, available)
            return None

    edu = _edu_gradient(ssb, year)
    band = _age_band_turnout(ssb, year)

    lut: TurnoutLUT = {}
    for sex in SEX_LABELS:
        kj = SEX_TO_KJONN[sex]
        tot = edu.get((kj, "TOT"))
        for a in AGE_LABELS:
            t_a = band.get((kj, a))
            for e in EDU_LABELS:
                t_e = edu.get((kj, EDU_TO_NIVAA[e]))
                if t_a is None or t_e is None or not tot:
                    logger.warning("Deltakelse %s: mangler (%s, %s, %s) — vekt 1.0.",
                                   year, sex, a, e)
                    continue
                lut[(sex, a, e)] = float(np.clip(t_a * t_e / tot, 0.01, 0.99))
    logger.info("Deltakelse %s lastet: %d celler, spenn %.1f–%.1f %%.",
                year, len(lut), min(lut.values()) * 100, max(lut.values()) * 100)
    return lut


def turnout_weight(lut: TurnoutLUT | None, kjonn: str, aldersgruppe: str,
                   utdanning: str) -> float:
    """Vekt for en celle; 1.0 når LUT mangler (uniform fallback)."""
    if lut is None:
        return 1.0
    return lut.get((kjonn, aldersgruppe, utdanning), 1.0)
