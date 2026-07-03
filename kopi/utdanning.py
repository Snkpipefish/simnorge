"""Utdanningsnivå per person — tre MÅLTE kilder, kombinert med rake.

- 09429: utdanningsfordeling per KOMMUNE × KJØNN (16+) — den harde marginalen.
- 08921: nasjonalt utdanningsnivå per ALDER × KJØNN — aldersgradienten.
- 13555: velgerundersøkelsen — utdanning per PARTI (stortingsvalget 2025).

Per person: P(utdanning) ∝ kommunens (kjønns)margin × alderstilt × partitilt,
raket per (kommune, kjønn) slik at kommunemarginalen (09429) holder EKSAKT.
Tiltene flytter altså utdanning MELLOM aldersgrupper/velgergrupper innen
kommunen — de endrer aldri kommunens målte fordeling. Samme mønster som
valgtildelingen i kopi/valg.py.

Antakelsen som står igjen: alders- og partigradienten (nasjonale) gjelder i
alle kommuner, og de to er uavhengige gitt kommune×kjønn. «Uoppgitt» i
kildene fordeles proporsjonalt (logget). Under 16 år: ingen utdanning (NaN).

Nivåene (4): grunnskole, videregaaende (inkl. fagskole), uni_kort, uni_lang.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient
from .valg import PARTIER

logger = logging.getLogger("simnorge.kopi.utdanning")

NIVAER = ["grunnskole", "videregaaende", "uni_kort", "uni_lang"]

# 09429/08921-koder -> våre nivåer ('09a' uoppgitt utelates og renormaliseres).
_NIVAA_MAP = {"01": 0, "02a": 1, "11": 1, "03a": 2, "04a": 3}
# 13555-koder -> våre nivåer.
_VUND_MAP = {"1-2": 0, "3-4-5": 1, "6": 2, "7-8": 3}
# 08921 aldersbånd (16+) med nedre grenser for oppslag.
_ALDER8 = ["16-19", "20-24", "25-29", "30-39", "40-49", "50-59", "60-66", "067+"]
_ALDER8_GRENSER = [20, 25, 30, 40, 50, 60, 67]
_KJONN = ["1", "2"]


def _rake_celler(basis: np.ndarray, teller: np.ndarray, margin: np.ndarray,
                 *, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Radnormaliser basis (C, L) og juster kolonnefaktorer til det
    telle-vektede aggregatet matcher margin (L). Som ValgModell._rake."""
    margin = np.maximum(margin, 0)
    s = margin.sum()
    if s <= 0 or teller.sum() == 0:
        return basis / np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)
    margin = margin / s
    f = np.ones(len(margin))
    n_tot = teller.sum()
    q = basis / np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)
    for _ in range(iters):
        q = basis * f[None, :]
        q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        agg = (teller[:, None] * q).sum(axis=0) / n_tot
        if np.abs(agg - margin).max() < tol:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            f = f * np.where(agg > 0, margin / agg, 1.0)
    return q


class UtdanningsModell:
    def __init__(self, ssb: SSBClient, *, valgaar: str = "2025"):
        self.margin = self._last_09429(ssb)      # (kommune, kjønn) -> (4,)
        self.alderstilt = self._last_08921(ssb)  # (2, 8, 4)
        self.partitilt = self._last_13555(ssb, valgaar)  # (11, 4); siste = ingen

    def _last_09429(self, ssb: SSBClient) -> dict[tuple[str, int], np.ndarray]:
        koder = ssb.variable_codes("09429")
        year = koder["Tid"][-1]
        kommuner = [k for k in koder["Region"] if k.isdigit() and len(k) == 4]
        df = ssb.fetch("09429", {
            "Region": kommuner, "Nivaa": list(_NIVAA_MAP), "Kjonn": _KJONN,
            "ContentsCode": ["Personer"], "Tid": [year],
        })
        ut: dict[tuple[str, int], np.ndarray] = {}
        for (kom, kj), g in df.groupby(["Region", "Kjonn"]):
            m = np.zeros(4)
            for _, r in g.iterrows():
                if not r["missing"] and not np.isnan(r["value"]):
                    m[_NIVAA_MAP[r["Nivaa"]]] += r["value"]
            if m.sum() > 0:
                ut[(kom, _KJONN.index(kj))] = m / m.sum()
        logger.info("09429 (%s): utdanningsmargin for %d (kommune, kjønn)-"
                    "celler (uoppgitt utelatt, renormalisert).", year, len(ut))
        return ut

    def _last_08921(self, ssb: SSBClient) -> np.ndarray:
        year = ssb.variable_codes("08921")["Tid"][-1]
        df = ssb.fetch("08921", {
            "Region": ["0"], "Alder": _ALDER8, "Kjonn": _KJONN,
            "UtdanNivaa": list(_NIVAA_MAP), "ContentsCode": ["Personer"],
            "Tid": [year],
        })
        tall = np.zeros((2, len(_ALDER8), 4))
        for _, r in df.iterrows():
            if r["missing"] or np.isnan(r["value"]):
                continue
            tall[_KJONN.index(r["Kjonn"]), _ALDER8.index(r["Alder"]),
                 _NIVAA_MAP[r["UtdanNivaa"]]] += r["value"]
        p_alder = tall / np.maximum(tall.sum(axis=2, keepdims=True), 1e-12)
        p_kjonn = (tall.sum(axis=1) /
                   np.maximum(tall.sum(axis=(1, 2), keepdims=False)[:, None], 1e-12))
        tilt = np.clip(p_alder / np.maximum(p_kjonn[:, None, :], 1e-12), 0.05, 20.0)
        logger.info("08921 (%s): nasjonal aldersgradient i utdanning "
                    "(2 kjønn × %d bånd).", year, len(_ALDER8))
        return tilt

    def _last_13555(self, ssb: SSBClient, valgaar: str) -> np.ndarray:
        from data import PARTY_CODES
        rev = {v: k for k, v in PARTY_CODES.items()}
        pkoder = [rev[p] for p in PARTY_CODES.values()] + ["92"]
        df = ssb.fetch("13555", {
            "PolitParti": pkoder, "Kjonn": ["0"],
            "UtdanNivaa": list(_VUND_MAP), "ContentsCode": ["Velgere"],
            "Tid": [valgaar],
        })
        tall = np.zeros((len(pkoder), 4))
        for _, r in df.iterrows():
            if r["missing"] or np.isnan(r["value"]):
                continue
            tall[pkoder.index(r["PolitParti"]), _VUND_MAP[r["UtdanNivaa"]]] += r["value"]
        nat = tall.sum(axis=0)
        nat = nat / nat.sum()
        p = tall / np.maximum(tall.sum(axis=1, keepdims=True), 1e-12)
        tilt = np.clip(p / nat[None, :], 0.2, 5.0)
        # Rad for ikke-velgere: ingen partiinformasjon -> tilt 1.
        tilt = np.vstack([tilt, np.ones(4)])
        logger.info("13555 (%s): målt utdanning-parti-tilt for %d partier.",
                    valgaar, len(pkoder))
        return tilt

    # ------------------------------------------------------------------ #
    def tildel(self, df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
        n = len(df)
        rng = np.random.default_rng(seed)
        alder = df["alder"].to_numpy()
        kjonn_i = df["kjonn"].cat.codes.to_numpy()
        voksen = alder >= 16

        band = np.searchsorted(_ALDER8_GRENSER, np.maximum(alder, 16),
                               side="right")
        parti_navn = list(df["parti"].cat.categories)
        til_tiltrad = np.array([PARTIER.index(p) if p in PARTIER else len(PARTIER)
                                for p in parti_navn])
        parti_i = til_tiltrad[df["parti"].cat.codes]          # 0..10

        celle = band * 11 + parti_i                            # 0..87
        koder = np.full(n, -1, dtype=np.int8)
        nasjonal = np.array([0.24, 0.41, 0.25, 0.10])          # grov fallback

        grupper = pd.DataFrame({
            "kommune": df["kommune"].astype(str), "kj": kjonn_i,
            "celle": celle, "idx": np.arange(n)})[voksen]
        mangler = 0
        for (kom, kj), g in grupper.groupby(["kommune", "kj"], sort=False):
            m = self.margin.get((kom, kj))
            if m is None:
                m = nasjonal; mangler += 1
            teller = np.bincount(g["celle"], minlength=88)
            aktive = np.flatnonzero(teller)
            basis = (self.alderstilt[kj, aktive // 11]
                     * self.partitilt[aktive % 11] * m[None, :])
            q = _rake_celler(basis, teller[aktive], m)
            posisjon = {c: i for i, c in enumerate(aktive)}
            for c in aktive:
                rader = g.loc[g["celle"] == c, "idx"].to_numpy()
                koder[rader] = rng.choice(4, size=len(rader), p=q[posisjon[c]])
        if mangler:
            logger.warning("09429-margin manglet for %d (kommune, kjønn)-celler "
                           "— nasjonal fordeling brukt (logget).", mangler)

        ut = df.copy()
        ut["utdanning"] = pd.Categorical.from_codes(koder, categories=NIVAER)
        fordeling = ut.loc[voksen, "utdanning"].value_counts(normalize=True)
        logger.info("Utdanning tildelt %d personer 16+: %s.", int(voksen.sum()),
                    {k: round(float(v), 3) for k, v in fordeling.items()})
        return ut
