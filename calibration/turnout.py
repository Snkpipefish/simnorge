"""Valgdeltakelsesvekting fra tabell 13360 — Modul 7.

Problemet: uten vekting teller alle demografiske celler likt i kommune-
aggregatet, men de møter opp i svært ulik grad (2021: menn m/grunnskole 58.9 %,
kvinner m/UH-utdanning 88.2 %). Partipreferansen per celle må derfor vektes med
P(stemmer | celle) før den aggregeres til kommuneresultat.

13360 gir deltakelse nasjonalt per kjønn × utdanningsnivå (grunnskole/vgs/UH),
for valgårene 2021 og 2025. Ingen aldersdimensjon — vektingen fanger altså
utdannings-/kjønnsgradienten, ikke aldersgradienten (kjent begrensning).

Lekkasjedisiplin: bruk BYGGEÅRETS deltakelse når målåret predikeres
(2021-deltakelse for 2021→2025-backtesten). Mangler byggeåret i tabellen,
returneres None og kallende kode faller tilbake til uniform vekt — logget,
aldri stille.
"""

from __future__ import annotations

import logging

from data import SSBClient
from population.schema import EDU_LABELS, SEX_LABELS

logger = logging.getLogger("simnorge.deltakelse")

# Våre schema-etiketter -> 13360-koder.
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

TurnoutLUT = dict[tuple[str, str], float]   # (kjønn, utdanning) -> andel 0..1


def load_turnout(ssb: SSBClient, year: str) -> TurnoutLUT | None:
    """Nasjonal valgdeltakelse per (kjønn, utdanning) for ett valgår.

    Returnerer None (med logg) hvis året ikke finnes i 13360 — kallende kode
    skal da bruke uniform vekt i stedet for å gjette."""
    year = str(year)
    available = ssb.variable_codes("13360")["Tid"]
    if year not in available:
        logger.warning("13360 mangler valgåret %s (har %s) — deltakelsesvekting "
                       "deaktivert, uniform vekt brukes.", year, available)
        return None

    df = ssb.fetch("13360", {
        "Region": ["0"],
        "Kjonn": list(SEX_TO_KJONN.values()),
        "UtdNivaa": sorted(set(EDU_TO_NIVAA.values())),
        "ContentsCode": ["Valgdeltakelse"],
        "Tid": [year],
    })
    raw = {(r["Kjonn"], r["UtdNivaa"]): float(r["value"]) / 100.0
           for _, r in df.iterrows() if not r["missing"]}

    lut: TurnoutLUT = {}
    for sex in SEX_LABELS:
        for edu in EDU_LABELS:
            v = raw.get((SEX_TO_KJONN[sex], EDU_TO_NIVAA[edu]))
            if v is None:
                logger.warning("13360 %s: mangler (%s, %s) — cellen får vekt 1.0.",
                               year, sex, edu)
                continue
            lut[(sex, edu)] = v
    logger.info("Deltakelse %s lastet: %d celler, spenn %.1f–%.1f %%.",
                year, len(lut),
                min(lut.values()) * 100, max(lut.values()) * 100)
    return lut


def turnout_weight(lut: TurnoutLUT | None, kjonn: str, utdanning: str) -> float:
    """Vekt for en celle; 1.0 når LUT mangler (uniform fallback)."""
    if lut is None:
        return 1.0
    return lut.get((kjonn, utdanning), 1.0)
