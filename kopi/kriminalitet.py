"""Kriminalitet og trygghet — MÅLTE rater, antatte vekter.

Tre kilder med ekte tall:
- 04621 (levekårsundersøkelsen, 2023): andel utsatt for vold/trusler (kode 01),
  tyveri/skadeverk (07) og «urolig for vold på bosted» (06), per kjønn × alder.
- 08487: anmeldte lovbrudd per 1 000 innbyggere PER KOMMUNE (2024-2025-snitt),
  vold og mishandling + eiendomstyveri — det lokale, målte nivået.
- 11453 (2024): siktede for vold og mishandling per kjønn × alder; rate regnes
  mot kopiens egen befolkning (som ER landsbefolkningen).

Per person gir det tre trukkede flagg (utsatt_vold, utsatt_tyveri, siktet_vold)
og en TRYGGHET-skår 0-100. Ratene er målte; sammenveiingen til trygghet bruker
ANTATTE vekter (dokumentert under) — samme regime som resten av kopien:

    trygghet = 75  - 25·(uro-rate relativt til landet - 1)   [målt gradient]
                   - 12·(kommunens voldsrate-faktor - 1)      [målt nivå]
                   - 18·utsatt_vold - 8·utsatt_tyveri         [målte flagg]
                   + støy N(0, 10),  klippet [0, 100]

Kommunefaktorer klippes [0.5, 2.5]; prikkede kommuner får faktor 1 (logget).
Barn under 16 får ingen verdier (kildene dekker 16+; siktet fra 15).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient
from population.schema import AGE_LABELS

logger = logging.getLogger("simnorge.kopi.kriminalitet")

_UTSATT_AGE = {"16-24": "16-24", "25-44": "25-44", "45-66": "45-66", "67+": "067+"}
_KJONN = ["1", "2"]

# 08487-koder.
_VOLD = "4AAAAA-4ZZZZz"
_TYVERI = "1AAAAA-1ZZZZz"

# 11453: aldersbånd (15+) med nedre grense for oppslag via searchsorted.
_SIKTET_BAND = ["15-17", "18-20", "21-24", "25-29", "30-39", "40-49", "50-59", "60-150"]
_SIKTET_GRENSER = [18, 21, 25, 30, 40, 50, 60]

# Antatte vekter i trygghetsformelen (se moduldocstring).
_V_URO, _V_KOMMUNE, _V_UTSATT_VOLD, _V_UTSATT_TYV, _STOY = 25.0, 12.0, 18.0, 8.0, 10.0


class KriminalitetsModell:
    def __init__(self, ssb: SSBClient):
        self.p_utsatt_vold, self.p_utsatt_tyv, self.p_uro = self._last_04621(ssb)
        self.voldsfaktor, self.tyvfaktor = self._last_08487(ssb)
        self.siktede = self._last_11453(ssb)          # (2, 8) antall personer

    # ------------------------------------------------------------------ #
    def _last_04621(self, ssb: SSBClient):
        year = ssb.variable_codes("04621")["Tid"][-1]
        df = ssb.fetch("04621", {
            "Lovbrudd": ["01", "07", "06"], "Kjonn": _KJONN,
            "Alder": list(_UTSATT_AGE.values()),
            "ContentsCode": ["Utsatthet"], "Tid": [year],
        })
        ut = {}
        for kode in ["01", "07", "06"]:
            m = np.zeros((2, len(AGE_LABELS)))
            for si, kj in enumerate(_KJONN):
                for ai, band in enumerate(AGE_LABELS):
                    r = df[(df["Lovbrudd"] == kode) & (df["Kjonn"] == kj)
                           & (df["Alder"] == _UTSATT_AGE[band])]
                    if r.empty or bool(r.iloc[0]["missing"]):
                        logger.warning("04621 %s (%s,%s): prikket — 0 (logget).",
                                       kode, kj, band)
                        continue
                    m[si, ai] = float(r.iloc[0]["value"]) / 100.0
            ut[kode] = m
        logger.info("04621 (%s): målt utsatthet/uro per kjønn × aldersbånd.", year)
        return ut["01"], ut["07"], ut["06"]

    def _last_08487(self, ssb: SSBClient):
        koder = ssb.variable_codes("08487")
        year = koder["Tid"][-1]
        kommuner = [k for k in koder["Gjerningssted"]
                    if k.isdigit() and len(k) == 4]
        df = ssb.fetch("08487", {
            "Gjerningssted": kommuner, "LovbruddKrim": [_VOLD, _TYVERI],
            "ContentsCode": ["AnmLovbrPer1000"], "Tid": [year],
        })
        ut = []
        for gruppe in [_VOLD, _TYVERI]:
            sub = df[(df["LovbruddKrim"] == gruppe) & ~df["missing"]
                     & df["value"].notna()]
            rater = dict(zip(sub["Gjerningssted"], sub["value"].astype(float)))
            snitt = float(np.mean(list(rater.values()))) if rater else 1.0
            fakt = {k: float(np.clip(v / snitt, 0.5, 2.5))
                    for k, v in rater.items() if snitt > 0}
            logger.info("08487 (%s) %s: rate for %d kommuner (usnittet %.2f "
                        "per 1000); manglende får faktor 1.", year,
                        "vold" if gruppe == _VOLD else "tyveri", len(fakt), snitt)
            ut.append(fakt)
        return ut[0], ut[1]

    def _last_11453(self, ssb: SSBClient):
        year = ssb.variable_codes("11453")["Tid"][-1]
        df = ssb.fetch("11453", {
            "Kjonn": _KJONN, "Alder": _SIKTET_BAND,
            "ContentsCode": ["VoldMishandling"], "Tid": [year],
        })
        m = np.zeros((2, len(_SIKTET_BAND)))
        for si, kj in enumerate(_KJONN):
            for ai, band in enumerate(_SIKTET_BAND):
                r = df[(df["Kjonn"] == kj) & (df["Alder"] == band)]
                if r.empty or bool(r.iloc[0]["missing"]):
                    logger.warning("11453 (%s,%s): prikket — 0 (logget).", kj, band)
                    continue
                m[si, ai] = float(r.iloc[0]["value"])
        logger.info("11453 (%s): siktede for vold/mishandling per kjønn × "
                    "aldersbånd (rate regnes mot kopiens befolkning).", year)
        return m

    # ------------------------------------------------------------------ #
    def tildel(self, df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
        """Legg på utsatt_vold/utsatt_tyveri/siktet_vold/trygghet."""
        n = len(df)
        rng = np.random.default_rng(seed)
        alder = df["alder"].to_numpy()
        kjonn_i = df["kjonn"].cat.codes.to_numpy()
        voksen = alder >= 16

        band4 = np.searchsorted([25, 45, 67], np.maximum(alder, 16), side="right")
        vfakt = df["kommune"].map(
            lambda k: self.voldsfaktor.get(str(k), 1.0)).to_numpy(dtype=float)
        tfakt = df["kommune"].map(
            lambda k: self.tyvfaktor.get(str(k), 1.0)).to_numpy(dtype=float)

        # Utsatthet: målt (kjønn × alder)-rate × kommunens målte nivå.
        p_vold = np.clip(self.p_utsatt_vold[kjonn_i, band4] * vfakt, 0, 0.6)
        p_tyv = np.clip(self.p_utsatt_tyv[kjonn_i, band4] * tfakt, 0, 0.6)
        utsatt_vold = voksen & (rng.random(n) < p_vold)
        utsatt_tyv = voksen & (rng.random(n) < p_tyv)

        # Siktet for vold: nasjonal rate per (kjønn × fint aldersbånd) mot
        # kopiens egen befolkning som nevner; skalert med kommunens voldsnivå.
        kan_siktes = alder >= 15
        band8 = np.searchsorted(_SIKTET_GRENSER, np.maximum(alder, 15),
                                side="right")
        nevner = np.zeros((2, len(_SIKTET_BAND)))
        np.add.at(nevner, (kjonn_i[kan_siktes], band8[kan_siktes]), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_sikt = np.where(nevner > 0, self.siktede / nevner, 0.0)
        siktet = kan_siktes & (rng.random(n) < np.clip(
            p_sikt[kjonn_i, band8] * vfakt, 0, 0.5))

        # Trygghet: målt uro-gradient + målt lokalnivå + egne (trukkede) flagg,
        # med antatte vekter. Kun 16+.
        uro = self.p_uro[kjonn_i, band4]
        uro_rel = uro / self.p_uro.mean() if self.p_uro.mean() > 0 else uro * 0 + 1
        trygghet = (75.0
                    - _V_URO * (uro_rel - 1.0)
                    - _V_KOMMUNE * (vfakt - 1.0)
                    - _V_UTSATT_VOLD * utsatt_vold
                    - _V_UTSATT_TYV * utsatt_tyv
                    + rng.normal(0.0, _STOY, n))
        trygghet = np.clip(trygghet, 0.0, 100.0)

        ut = df.copy()
        ut["utsatt_vold"] = pd.array(np.where(voksen, utsatt_vold, None),
                                     dtype="boolean")
        ut["utsatt_tyveri"] = pd.array(np.where(voksen, utsatt_tyv, None),
                                       dtype="boolean")
        ut["siktet_vold"] = pd.array(np.where(kan_siktes, siktet, None),
                                     dtype="boolean")
        ut["trygghet"] = np.where(voksen, np.round(trygghet, 1),
                                  np.nan).astype(np.float32)
        logger.info("Kriminalitet: %.1f %% utsatt vold, %.1f %% tyveri, "
                    "%.2f %% siktet vold (16+/15+); trygghet snitt %.0f.",
                    utsatt_vold[voksen].mean() * 100,
                    utsatt_tyv[voksen].mean() * 100,
                    siktet[kan_siktes].mean() * 100,
                    trygghet[voksen].mean())
        return ut
