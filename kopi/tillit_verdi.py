"""To akser til: institusjonstillit (MÅLT gradient) og verdiliberal (antatt).

INSTITUSJONSTILLIT (0 = lav, 100 = høy):
13908 (velgerundersøkelsen 2021, nyeste årgang) gir andel med lav/høy tillit
til STORTINGET etter alder og STEMT/IKKE STEMT — hjemmesitterne er altså målt,
ikke antatt — pluss en utdanningsgradient nasjonalt. Per person trekkes
lav/middels/høy fra den målte (alder × stemt)-fordelingen, tiltet med målt
utdanningsfaktor, og settes til skår 25/55/82 + støy. Partijustering
(ANTAKELSE, liten): protestpartier noe lavere systemtillit.

VERDILIBERAL (0 = verdikonservativ, 100 = verdiliberal):
Ingen god åpen kilde krysser dette — aksen er ANTAKELSE i samme regime som
personlighetstypene: partibasis + kirkemedlemskap (målt flagg, antatt effekt
-8) + alder (eldre mer konservative) + utdanning (uni_lang +6) + støy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient
from .utdanning import NIVAER
from .ess import les_posisjoner

logger = logging.getLogger("simnorge.kopi.tillitverdi")

# 13908: aldersbånd vi bruker (dekker 18+; under 18 får yngste bånd).
_T_ALDER = ["18-24", "25-44", "45-66", "67-79", "80+"]
_T_GRENSER = [25, 45, 67, 80]
_SKAR = np.array([25.0, 55.0, 82.0])       # lav / middels / høy tillit
_STOY = 9.0

# Partijustering av tillit (ANTAKELSE — brukes bare når ESS-aggregatene
# mangler; ellers måles avviket per parti fra kopi/ess_posisjoner.json).
TILLIT_PARTI = {"FrP": -10.0, "Rødt": -7.0, "Andre": -5.0}

# Verdiliberal: partibasis (ANTAKELSE — samme fallback-regel som over).
VERDI_PARTI = {
    "KrF": 15, "Sp": 35, "FrP": 40, "Høyre": 45, "Andre": 50,
    "Ap": 55, "Rødt": 70, "SV": 75, "Venstre": 78, "MDG": 78,
    "stemte ikke": 50, "ikke stemmerett": 50,
}


def _fra_ess() -> tuple[dict, dict] | None:
    """(tillit-partijustering, verdi-partibasis) MÅLT fra ESS, eller None.

    Tillitsjusteringen er partiets målte avvik fra landssnittet (klippet
    ±18) — basisen fra 13908 bærer fortsatt alder/stemt/utdanning. Verdi-
    basisen er partiets målte posisjon; 16-17-åringer får landssnittet."""
    ess = les_posisjoner()
    if not ess:
        return None
    t = ess["akser"]["institusjonstillit"]
    just = {p: float(np.clip(v["mean"] - t["_alle"]["mean"], -18, 18))
            for p, v in t.items() if p != "_alle"}
    v = ess["akser"]["verdiliberal"]
    basis = {p: float(x["mean"]) for p, x in v.items() if p != "_alle"}
    basis["ikke stemmerett"] = float(v["_alle"]["mean"])
    return just, basis
_VERDI_DNK = -8.0                           # målt flagg, antatt effekt
_VERDI_ALDER = np.array([8.0, 4.0, 0.0, -6.0, -12.0])   # bånd som livssyn
_VERDI_ALDER_GRENSER = [16, 25, 45, 67]
_VERDI_UNI_LANG = 6.0
_VERDI_STOY = 10.0


class TillitVerdiModell:
    def __init__(self, ssb: SSBClient):
        self.p_tillit, self.utd_tilt = self._last_13908(ssb)

    def _last_13908(self, ssb: SSBClient):
        year = ssb.variable_codes("13908")["Tid"][-1]
        cc = ["LavTillit", "HoyTillit"]
        df = ssb.fetch("13908", {
            "InstitusjonOff": ["04"], "Alder": _T_ALDER + ["999"],
            "Kjonn": ["0"], "Stemt": ["0", "1", "2"],
            "UtdanNivaa": ["TOT", "1", "2", "3", "4"],
            "ContentsCode": cc, "Tid": [year],
        })

        def vec(alder: str, stemt: str, utd: str) -> np.ndarray:
            sub = df[(df["Alder"] == alder) & (df["Stemt"] == stemt)
                     & (df["UtdanNivaa"] == utd)]
            v = []
            for k in cc:
                r = sub[sub["ContentsCode"] == k]
                v.append(np.nan if r.empty or bool(r.iloc[0]["missing"])
                         else float(r.iloc[0]["value"]) / 100.0)
            lav, hoy = v
            if np.isnan(lav) or np.isnan(hoy):
                return np.array([np.nan] * 3)
            return np.array([lav, max(1.0 - lav - hoy, 0.0), hoy])

        # Målt basis: (stemt 1/2 × aldersbånd) -> [lav, middels, høy].
        p = np.zeros((2, len(_T_ALDER), 3))
        for si, stemt in enumerate(["1", "2"]):
            for ai, aband in enumerate(_T_ALDER):
                v = vec(aband, stemt, "TOT")
                if np.isnan(v).any():
                    v = vec("999", stemt, "TOT")
                    logger.warning("13908 (%s, %s): prikket — bruker alders-"
                                   "totalen for stemt-gruppen (logget).",
                                   stemt, aband)
                p[si, ai] = v

        # Målt utdanningstilt (nasjonalt, alle): P(nivå|utd)/P(nivå|alle).
        nat = vec("999", "0", "TOT")
        tilt = np.ones((4, 3))
        for ui, utd in enumerate(["1", "2", "3", "4"]):
            v = vec("999", "0", utd)
            if not np.isnan(v).any() and not np.isnan(nat).any():
                tilt[ui] = np.clip(v / np.maximum(nat, 1e-9), 0.2, 5.0)
        logger.info("13908 (%s): målt tillit til Stortinget etter "
                    "(stemt × alder) + utdanningstilt.", year)
        return p, tilt

    # ------------------------------------------------------------------ #
    def tildel(self, df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
        n = len(df)
        rng = np.random.default_rng(seed)
        alder = df["alder"].to_numpy()
        voksen = alder >= 16

        # --- Institusjonstillit ---
        band = np.searchsorted(_T_GRENSER, np.maximum(alder, 18), side="right")
        stemte = (~df["parti"].isin(["stemte ikke", "ikke stemmerett"])).to_numpy()
        probs = self.p_tillit[np.where(stemte, 0, 1), band]   # (n, 3)
        utd_i = df["utdanning"].cat.codes.to_numpy() if "utdanning" in df else \
            np.full(n, -1)
        kjent_utd = utd_i >= 0
        probs = probs.copy()
        probs[kjent_utd] *= self.utd_tilt[utd_i[kjent_utd]]
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

        u = rng.random(n)
        nivaa = (u[:, None] >= np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 2)
        tillit = _SKAR[nivaa] + rng.normal(0.0, _STOY, n)
        ess = _fra_ess()
        tillit_just = ess[0] if ess else TILLIT_PARTI
        verdi_basis = ess[1] if ess else VERDI_PARTI
        if ess:
            logger.info("Tillit-partijustering og verdibasis: MÅLT fra ESS.")
        for parti, just in tillit_just.items():
            tillit += np.where(df["parti"].to_numpy() == parti, just, 0.0)
        tillit = np.clip(tillit, 0.0, 100.0)

        # --- Verdiliberal ---
        basis = df["parti"].astype(str).map(verdi_basis).to_numpy(dtype=float)
        vband = np.searchsorted(_VERDI_ALDER_GRENSER, alder, side="right")
        verdi = (basis + _VERDI_ALDER[vband]
                 + _VERDI_DNK * df["medlem_dnk"].to_numpy()
                 + _VERDI_UNI_LANG * (utd_i == NIVAER.index("uni_lang"))
                 + rng.normal(0.0, _VERDI_STOY, n))
        verdi = np.clip(verdi, 0.0, 100.0)

        ut = df.copy()
        ut["institusjonstillit"] = np.where(voksen, np.round(tillit, 1),
                                            np.nan).astype(np.float32)
        ut["verdiliberal"] = np.where(voksen, np.round(verdi, 1),
                                      np.nan).astype(np.float32)
        logger.info("Tillit/verdi tildelt (16+): tillit snitt %.0f "
                    "(velgere %.0f, hjemmesittere %.0f), verdiliberal snitt %.0f.",
                    tillit[voksen].mean(),
                    tillit[voksen & stemte].mean(),
                    tillit[voksen & ~stemte].mean(),
                    verdi[voksen].mean())
        return ut
