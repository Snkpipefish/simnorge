"""Arbeidsmarkedsstatus per kommune — Modul 2, akse 5 (post-IPF).

Kilde: 13563 «Kommunefordelt prioritert arbeidsstyrkestatus (15+)» — en
PARTISJON (hver bosatt telles i nøyaktig én status), så ingen forsoning av
overlappende kilder trengs. NAV-ledighetstallene og trygdestatistikken
ligger allerede inne i partisjonen.

To eksplisitte ekstrapoleringer (logges, skjules ikke):
1. Aldersforsoning: 13563 bruker båndene 15-29/30-61/62+, personene våre
   16-24/25-44/45-66/67+. Vi veier om via kommunens ettårige alderstall
   (07459, samme kilde som marginalene) — antakelsen er at statusmiksen er
   konstant INNAD i kildebåndet.
2. Status tildeles uavhengig av kjønn/utdanning/inntekt gitt alder —
   13563 har ikke disse aksene per kommune. Samme klasse antakelse som
   IPF-ens betingede uavhengighet (jf. ipf.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from data import SSBClient
from .schema import AGE_LABELS, STATUS_LABELS, age_band

logger = logging.getLogger("simnorge.status")

ARBSTATUS_TABLE = "13563"
SRC_BANDS = ["15-29", "30-61", "62+"]
# 13563-leaf -> vår etikett. Tiltaksdeltakere regnes som arbeidsledige
# (nærmeste kategori i holdningstabellenes partisjon).
LEAF_MAP = {
    "A.01": "yrkesaktiv", "A.09": "arbeidsledig", "U.01": "arbeidsledig",
    "U.03": "student", "U.04-U.05": "ufor", "U.06-U.07": "pensjonist",
    "U.90A": "annet",
}


@dataclass
class StatusMix:
    kommune: str
    year: str
    probs: np.ndarray                    # (4 aldersbånd, 6 status), rader sum 1
    available: bool = True
    issues: list[str] = field(default_factory=list)


def _latest_year(ssb: SSBClient, year: str) -> str:
    valid = ssb.variable_codes(ARBSTATUS_TABLE)["Tid"]
    ok = [t for t in valid if t <= year]
    return ok[-1] if ok else valid[-1]


def _src_shares(ssb: SSBClient, region: str, year: str) -> np.ndarray | None:
    """(3 kildebånd, 6 status) andeler fra 13563, NaN-fri; None hvis ubrukelig."""
    df = ssb.fetch(ARBSTATUS_TABLE, {
        "Region": [region], "HovArbStyrkStatus": list(LEAF_MAP),
        "Alder": SRC_BANDS, "ContentsCode": ["Bosatte"], "Tid": [year],
    })
    counts = np.zeros((len(SRC_BANDS), len(STATUS_LABELS)))
    for _, r in df.iterrows():
        if bool(r["missing"]):
            return None
        bi = SRC_BANDS.index(r["Alder"])
        si = STATUS_LABELS.index(LEAF_MAP[r["HovArbStyrkStatus"]])
        counts[bi, si] += float(r["value"])
    row = counts.sum(axis=1, keepdims=True)
    if (row <= 0).any():
        return None
    return counts / row


def _overlap_weights(ssb: SSBClient, kommune: str, year: str) -> np.ndarray:
    """(4 våre bånd, 3 kildebånd): andel av vårt bånds befolkning som ligger i
    hvert kildebånd, fra kommunens ettårige alderstall (07459, diskcachet)."""
    codes = ssb.variable_codes("07459")
    ages = [a for a in codes["Alder"] if a.isdigit() and int(a) >= 16]
    df = ssb.fetch("07459", {"Region": [kommune], "Kjonn": ["1", "2"],
                             "Alder": ages, "ContentsCode": ["Personer1"],
                             "Tid": [year]})
    w = np.zeros((len(AGE_LABELS), len(SRC_BANDS)))
    for _, r in df.iterrows():
        age = int(r["Alder"])
        b4 = age_band(age)
        if b4 is None or bool(r["missing"]):
            continue
        b3 = "15-29" if age <= 29 else ("30-61" if age <= 61 else "62+")
        w[AGE_LABELS.index(b4), SRC_BANDS.index(b3)] += float(r["value"])
    row = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        w = np.where(row > 0, w / row, 0.0)
    return w


def build_status_mix(ssb: SSBClient, kommune: str, *, year: str) -> StatusMix:
    src_year = _latest_year(ssb, year)
    mix = StatusMix(kommune=kommune, year=src_year,
                    probs=np.full((len(AGE_LABELS), len(STATUS_LABELS)), np.nan))
    shares = _src_shares(ssb, kommune, src_year)
    if shares is None:
        nat = _src_shares(ssb, "0", src_year)
        if nat is None:
            mix.available = False
            mix.issues.append("13563: både kommune og landet prikket/utilgjengelig — "
                              "status utelatt.")
            logger.warning("Kommune %s: %s", kommune, mix.issues[-1])
            return mix
        shares = nat
        mix.issues.append("13563: kommunens statusmiks prikket — bruker NASJONAL "
                          "miks per aldersbånd (eksplisitt ekstrapolering).")
        logger.warning("Kommune %s: %s", kommune, mix.issues[-1])

    w = _overlap_weights(ssb, kommune, year)
    probs = w @ shares                              # (4, 6)
    row = probs.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        probs = np.where(row > 0, probs / row, 1.0 / len(STATUS_LABELS))
    mix.probs = probs
    return mix


def assign_status(persons, mix: StatusMix, *, column: str = "arbeidsmarkedsstatus"):
    """Tildel status deterministisk per aldersbånd (største-rest, samme
    disiplin som IPF-integeriseringen). 'ukjent' hvis miksen mangler."""
    import pandas as pd

    out = persons.copy()
    if not mix.available or len(out) == 0:
        out[column] = pd.Categorical(["ukjent"] * len(out),
                                     categories=STATUS_LABELS + ["ukjent"])
        return out

    labels = np.empty(len(out), dtype=object)
    for bi, band in enumerate(AGE_LABELS):
        idx = np.flatnonzero((out["aldersgruppe"] == band).to_numpy())
        n = len(idx)
        if n == 0:
            continue
        raw = mix.probs[bi] * n
        base = np.floor(raw).astype(int)
        rest = n - base.sum()
        order = np.argsort(-(raw - base))
        base[order[:rest]] += 1
        fill = np.repeat(np.arange(len(STATUS_LABELS)), base)
        labels[idx] = np.array(STATUS_LABELS, dtype=object)[fill]
    out[column] = pd.Categorical(labels.tolist(),
                                 categories=STATUS_LABELS + ["ukjent"])
    return out
