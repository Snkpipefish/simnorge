"""Helselag — MÅLTE rater fra FHI (Folkehelsestatistikk) og ESS.

FHIs åpne API (statistikk-data.fhi.no, kilde «nokkel» — arvtakeren etter
Kommunehelsa statistikkbank) gir per kommune, alders-/kjønnsstandardisert:

- KPR (tabell 369, 3-årssnitt): brukere av primærhelsetjenesten per 1000
  innbyggere for «Psykiske symptomer og lidelser» og «Muskel og skjelett» —
  nasjonale rater per kjønn × aldersbånd, og SMR per kommune (Norge = 100).
- Forventet levealder (tabell 660, 25-årssnitt): leveår ved fødsel per
  kommune × kjønn × utdanningsnivå.

Per person gir det:
- psykisk_kontakt / muskel_kontakt: Bernoulli av nasjonal (kjønn, alder)-
  rate × kommunens SMR/100 — målt gradient × målt lokalnivå. Merk: dette er
  KONTAKT MED PRIMÆRHELSETJENESTEN i løpet av et år, ikke diagnoseprevalens.
- forventet_levealder: gruppens målte leveår ved fødsel (kommune × kjønn ×
  utdanning; prikket -> kommunetotal -> landstall, logget).
- helse_god: egenvurdert god/svært god helse fra ESS (målt basis per
  kjønn × aldersbånd + målte avvik for inntektsdesil og utdanning).

Aldersgrensen 75-79 i KPR brukes for alle 75+ (eldste bånd i kilden).
FHI-svar caches som json under .cache/fhi/.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("simnorge.kopi.helse")

_API = "https://statistikk-data.fhi.no/api/open/v1/nokkel/table/{}/data"
_CACHE = Path(".cache/fhi")

_KPR_TABELL = 369
_LEVEALDER_TABELL = 660
_LEVEALDER_UTD_TABELL = 507
_PSYKISK = "P01_P29ogP70_P99"
_MUSKEL = "L01_L29ogL70_L71ogL82_L99"
_KPR_BAND = ["0_14", "15_24", "25_44", "45_64", "65_74", "75_79"]
_KPR_GRENSER = [15, 25, 45, 65, 75]
_UTD_TIL_FHI = {"grunnskole": "1", "videregaaende": "2",
                "uni_kort": "3", "uni_lang": "3"}


def _hent(tabell: int, dimensjoner: list[dict]) -> pd.DataFrame:
    body = {"dimensions": dimensjoner, "response": {"format": "csv3"}}
    nokkel = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
    fil = _CACHE / f"{tabell}_{nokkel}.csv"
    if fil.exists():
        tekst = fil.read_text(encoding="utf-8")
    else:
        r = requests.post(_API.format(tabell), json=body, timeout=120)
        r.raise_for_status()
        tekst = r.text
        _CACHE.mkdir(parents=True, exist_ok=True)
        fil.write_text(tekst, encoding="utf-8")
    return pd.read_csv(io.StringIO(tekst), sep=";", dtype=str)


def _tall(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class HelseModell:
    def __init__(self, *, ess: dict | None = None):
        self.rater = self._last_kpr_rater()          # {gruppe: (2, 6) per 1000}
        self.smr = self._last_kpr_smr()              # {gruppe: {kommune: smr}}
        self.levealder = self._last_levealder()      # {(kommune, kjønn, utd): år}
        self.ess_helse = (ess or {}).get("helse")

    def _last_kpr_rater(self) -> dict[str, np.ndarray]:
        df = _hent(_KPR_TABELL, [
            {"code": "GEO", "filter": "item", "values": ["0"]},
            {"code": "AAR", "filter": "item", "values": ["2022_2024"]},
            {"code": "KJONN", "filter": "item", "values": ["1", "2"]},
            {"code": "ALDER", "filter": "item", "values": _KPR_BAND},
            {"code": "KODEGRUPPE", "filter": "item", "values": [_PSYKISK, _MUSKEL]},
            {"code": "MEASURE_TYPE", "filter": "item", "values": ["RATE"]},
        ])
        ut = {}
        for gruppe in [_PSYKISK, _MUSKEL]:
            m = np.zeros((2, len(_KPR_BAND)))
            sub = df[df["KODEGRUPPE"] == gruppe]
            for _, r in sub.iterrows():
                v = _tall(pd.Series([r["RATE"]]))[0]
                if not np.isnan(v):
                    m[int(r["KJONN"]) - 1, _KPR_BAND.index(r["ALDER"])] = v
            ut[gruppe] = m
        logger.info("FHI KPR (2022-2024): nasjonale rater per kjønn × %d "
                    "aldersbånd for psykisk og muskel/skjelett.", len(_KPR_BAND))
        return ut

    def _last_kpr_smr(self) -> dict[str, dict[str, float]]:
        df = _hent(_KPR_TABELL, [
            {"code": "GEO", "filter": "all", "values": ["*"]},
            {"code": "AAR", "filter": "item", "values": ["2022_2024"]},
            {"code": "KJONN", "filter": "item", "values": ["0"]},
            {"code": "ALDER", "filter": "item", "values": ["0_74"]},
            {"code": "KODEGRUPPE", "filter": "item", "values": [_PSYKISK, _MUSKEL]},
            {"code": "MEASURE_TYPE", "filter": "item", "values": ["SMR"]},
        ])
        df = df[df["GEO"].str.len() == 4]
        ut = {}
        for gruppe in [_PSYKISK, _MUSKEL]:
            sub = df[df["KODEGRUPPE"] == gruppe]
            v = _tall(sub["SMR"])
            ut[gruppe] = {g: float(np.clip(x / 100.0, 0.6, 1.6))
                          for g, x in zip(sub["GEO"], v) if not np.isnan(x)}
            logger.info("FHI KPR SMR: %s for %d kommuner (klippet [0.6, 1.6]; "
                        "manglende får 1).", "psykisk" if gruppe == _PSYKISK
                        else "muskel", len(ut[gruppe]))
        return ut

    def _last_levealder(self) -> dict:
        """Kommunenivå fra 660 (25-årig, kun totaler publisert) + utdannings-
        gradient fra 507 (7-årig, land/fylke). Én ordbok med samme nøkler."""
        ut = {}
        df = _hent(_LEVEALDER_TABELL, [
            {"code": "GEO", "filter": "all", "values": ["*"]},
            {"code": "AAR", "filter": "item", "values": ["2000_2024"]},
            {"code": "KJONN", "filter": "item", "values": ["1", "2"]},
            {"code": "ALDER", "filter": "item", "values": ["0"]},
            {"code": "UTDANN", "filter": "item", "values": ["0"]},
            {"code": "MEASURE_TYPE", "filter": "item", "values": ["MEIS"]},
        ])
        df = df[df["GEO"].str.len().isin([1, 4])]
        for _, r in df.iterrows():
            v = _tall(pd.Series([r["MEIS"]]))[0]
            if not np.isnan(v):
                ut[(r["GEO"], int(r["KJONN"]) - 1, "0")] = float(v)

        # Utdanningsbrutt levealder publiseres ved ALDER 30 (utdanning er
        # ikke definert ved fødsel) — differansene brukes som gradient.
        grad = _hent(_LEVEALDER_UTD_TABELL, [
            {"code": "GEO", "filter": "all", "values": ["*"]},
            {"code": "AAR", "filter": "item", "values": ["2018_2024"]},
            {"code": "KJONN", "filter": "item", "values": ["1", "2"]},
            {"code": "ALDER", "filter": "item", "values": ["30"]},
            {"code": "UTDANN", "filter": "item", "values": ["0", "1", "2", "3"]},
            {"code": "MEASURE_TYPE", "filter": "item", "values": ["MEIS"]},
        ])
        grad = grad[grad["GEO"].str.len().isin([1, 2])]
        for _, r in grad.iterrows():
            v = _tall(pd.Series([r["MEIS"]]))[0]
            if not np.isnan(v):
                nokkel = (r["GEO"], int(r["KJONN"]) - 1, r["UTDANN"])
                if nokkel not in ut:
                    ut[nokkel] = float(v)
        logger.info("FHI levealder: %d celler (kommunetotaler 660 + "
                    "utdanningsbrutt land/fylke 507).", len(ut))
        return ut

    # ------------------------------------------------------------------ #
    def tildel(self, df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
        n = len(df)
        rng = np.random.default_rng(seed)
        alder = df["alder"].to_numpy()
        kjonn_i = df["kjonn"].cat.codes.to_numpy()
        kommune = df["kommune"].astype(str).to_numpy()
        band = np.searchsorted(_KPR_GRENSER, alder, side="right")

        ut = df.copy()
        for gruppe, navn in [(_PSYKISK, "psykisk_kontakt"),
                             (_MUSKEL, "muskel_kontakt")]:
            smr = pd.Series(kommune).map(
                lambda k: self.smr[gruppe].get(k, 1.0)).to_numpy()
            p = np.clip(self.rater[gruppe][kjonn_i, band] / 1000.0 * smr, 0, 0.85)
            ut[navn] = rng.random(n) < p

        # Forventet levealder: utdanningsbrutte tall er prikket på kommune-
        # nivå, så vi kombinerer MÅLT kommunenivå med MÅLT fylkesgradient:
        # levealder = kommunetotal + (fylkets utd-verdi - fylkets total).
        # Fallback-kjede for hvert ledd; barn/ukjent utdanning får total.
        utd_fhi = df["utdanning"].astype(str).map(_UTD_TIL_FHI).fillna("0")
        kart = {f"{k}|{kj}|{u}": v for (k, kj, u), v in self.levealder.items()}
        kj_s = pd.Series(kjonn_i.astype(str), index=df.index)
        kom_s = pd.Series(kommune, index=df.index)
        fylke_s = kom_s.str[:2]
        basis = (kom_s + "|" + kj_s + "|0").map(kart)
        basis = basis.where(basis.notna(), ("0|" + kj_s + "|0").map(kart))
        f_utd = (fylke_s + "|" + kj_s + "|" + utd_fhi).map(kart)
        f_tot = (fylke_s + "|" + kj_s + "|0").map(kart)
        f_utd = f_utd.where(f_utd.notna(),
                            ("0|" + kj_s + "|" + utd_fhi).map(kart))
        f_tot = f_tot.where(f_tot.notna(), ("0|" + kj_s + "|0").map(kart))
        grad = (f_utd - f_tot).fillna(0.0)
        ut["forventet_levealder"] = (basis + grad).astype(np.float32).round(1)

        # Egenvurdert helse fra ESS (16+): målt basis + målte avvik.
        if self.ess_helse:
            basis = np.array(self.ess_helse["basis"])
            b4 = np.searchsorted([25, 45, 67], np.maximum(alder, 16), side="right")
            p = basis[kjonn_i, b4]
            inntekt = df["brutto_inntekt"].to_numpy()
            har = ~np.isnan(inntekt)
            pct = np.full(n, 0.5)
            pct[har] = pd.Series(inntekt[har]).rank(pct=True).to_numpy()
            desil = np.clip((pct * 10).astype(int), 0, 9)
            p = p + np.array(self.ess_helse["inntektsdesil"], dtype=float)[desil]
            utd_i = df["utdanning"].cat.codes.to_numpy()
            dp_utd = np.array([self.ess_helse["utdanning"].get(str(i), 0.0)
                               for i in range(4)])
            p = p + np.where(utd_i >= 0, dp_utd[np.clip(utd_i, 0, 3)], 0.0)
            god = rng.random(n) < np.clip(p, 0.05, 0.98)
            ut["helse_god"] = pd.array(np.where(alder >= 16, god, None),
                                       dtype="boolean")
        logger.info("Helse tildelt: %.1f %% psykisk kontakt, %.1f %% muskel/"
                    "skjelett, levealder snitt %.1f år.",
                    ut["psykisk_kontakt"].mean() * 100,
                    ut["muskel_kontakt"].mean() * 100,
                    float(ut["forventet_levealder"].mean()))
        return ut
