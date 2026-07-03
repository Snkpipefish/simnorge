"""Tilbøyeligheter per person — målt basis der den finnes, antatt resten.

Fem skårer 0-100 (for sannsynlighetslignende skårer: ~forventet prosent):

- tilb_stemme    Ville stemt ved neste valg. MÅLT basis: 10440 (valgdeltakelse
                 2025 etter alder × utdanning, kombinert multiplikativt rundt
                 totalen — etablert mønster). ANTATT justering: vane (stemte
                 2025 +8 / satt hjemme −8) og institusjonstillit (±4).
- tilb_flytte    Flytter (innenlands) i løpet av et år. MÅLT: 05540-flyttinger
                 2025 per kjønn × 5-årsbånd delt på kopiens egen befolkning i
                 båndet. Ingen antakelser oppå — ren målt rate.
- tilb_frivillig Gjør frivillig innsats i løpet av året. MÅLT basis: 13826
                 (2025) per kjønn × aldersbånd, typene kombinert under
                 uavhengighetsantakelse (1 - prod(1-p)). ANTATT justering:
                 utadvendt personlighet +6, kirkemedlem +5, minst sentrale
                 kommuner (klasse 5-6) +4.
- tilb_protest   Politisk protestvilje. REN ANTAKELSE fra egne kolonner:
                 lav institusjonstillit (hovedvekt), lav trygghet, ung alder.
- tilb_risiko    Risikovillighet. REN ANTAKELSE: alder (ung høyt), kjønn,
                 utadvendt/spontan personlighetstype, siktet-flagget.

Alle formler står her og kan justeres på ett sted. Under 16 år: NaN (barna
har ikke grunnlagskolonnene).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient

logger = logging.getLogger("simnorge.kopi.tilboyelighet")

_KJONN = ["1", "2"]

# 10440: aldersbånd vi forsøker, i stigende rekkefølge, med nedre grenser.
_STEMME_BAND = ["18-19", "20-24", "25-44", "45-66", "67-79", "080+"]
_STEMME_GRENSER = [20, 25, 45, 67, 80]
_UTD_10440 = ["1-2", "3-5", "6-8"]        # -> våre nivåer [grunnskole, vgs+, uni]

# 05540: 5-årsbånd 0-79 + 80+.
_FLYTT_BAND = ["00-04", "05-09", "10-14", "15-19", "20-24", "25-29", "30-34",
               "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69",
               "70-74", "80+"]

_FRIV_ALDER = {"16-24": "16-24", "25-44": "25-44", "45-66": "45-66", "67+": "067+"}
_FRIV_TYPER = ["01", "02", "03", "04", "90"]


class TilboyelighetsModell:
    def __init__(self, ssb: SSBClient):
        self.stemme_alder, self.stemme_utd, self.stemme_tot = self._last_10440(ssb)
        self.flytt = self._last_05540(ssb)            # (2, band) antall flyttinger
        self.friv = self._last_13826(ssb)             # (2, 4) andel

    # ------------------------------------------------------------------ #
    def _last_10440(self, ssb: SSBClient):
        year = ssb.variable_codes("10440")["Tid"][-1]
        gyldige = ssb.variable_codes("10440")["Alder"]
        band = [b for b in _STEMME_BAND if b in gyldige]
        df = ssb.fetch("10440", {
            "Kjonn": ["0"], "Alder": band + ["999"],
            "UtdNivaa": _UTD_10440 + ["TOT"],
            "ContentsCode": ["Deltakelse"], "Tid": [year],
        })

        def verdi(alder, utd):
            r = df[(df["Alder"] == alder) & (df["UtdNivaa"] == utd)]
            if r.empty or bool(r.iloc[0]["missing"]) or np.isnan(r.iloc[0]["value"]):
                return np.nan
            return float(r.iloc[0]["value"]) / 100.0

        tot = verdi("999", "TOT")
        a = np.array([verdi(b, "TOT") for b in band])
        u = np.array([verdi("999", k) for k in _UTD_10440])
        # Prikkede bånd arver totalen (logget).
        if np.isnan(a).any():
            logger.warning("10440: %d prikkede aldersbånd fikk totalen.",
                           int(np.isnan(a).sum()))
            a = np.where(np.isnan(a), tot, a)
        if np.isnan(u).any():
            u = np.where(np.isnan(u), tot, u)
        logger.info("10440 (%s): målt valgdeltakelse etter alder (%d bånd) og "
                    "utdanning.", year, len(band))
        self._stemme_bandnavn = band
        return a, u, tot

    def _last_05540(self, ssb: SSBClient):
        koder = ssb.variable_codes("05540")
        year = koder["Tid"][-1]
        band = [b for b in _FLYTT_BAND if b in koder["AldGrupp"]]
        ekstra = [b for b in koder["AldGrupp"] if b not in band]
        if ekstra:
            band = band + [b for b in ekstra if b[0].isdigit()]
        df = ssb.fetch("05540", {
            "Kjonn": _KJONN, "AldGrupp": band,
            "ContentsCode": ["Flyttinger"], "Tid": [year],
        })
        ut = np.zeros((2, len(band)))
        for _, r in df.iterrows():
            if r["missing"] or np.isnan(r["value"]):
                continue
            ut[_KJONN.index(r["Kjonn"]), band.index(r["AldGrupp"])] = r["value"]
        self._flytt_band = band
        logger.info("05540 (%s): målte flyttinger for %d aldersbånd.", year, len(band))
        return ut

    def _last_13826(self, ssb: SSBClient):
        year = ssb.variable_codes("13826")["Tid"][-1]
        df = ssb.fetch("13826", {
            "TypeFrivilligInnsats": _FRIV_TYPER, "Kjonn": _KJONN,
            "Alder": list(_FRIV_ALDER.values()),
            "ContentsCode": ["Andel"], "Tid": [year],
        })
        p_ingen = np.ones((2, 4))
        for _, r in df.iterrows():
            if r["missing"] or np.isnan(r["value"]):
                continue
            si = _KJONN.index(r["Kjonn"])
            ai = list(_FRIV_ALDER.values()).index(r["Alder"])
            p_ingen[si, ai] *= 1.0 - float(r["value"]) / 100.0
        logger.info("13826 (%s): målt frivillig innsats per kjønn × aldersbånd "
                    "(typer kombinert under uavhengighetsantakelse).", year)
        return 1.0 - p_ingen

    # ------------------------------------------------------------------ #
    def tildel(self, df: pd.DataFrame, *, seed: int = 0,
               sentralitet: dict[str, int] | None = None) -> pd.DataFrame:
        n = len(df)
        rng = np.random.default_rng(seed)
        alder = df["alder"].to_numpy()
        kjonn_i = df["kjonn"].cat.codes.to_numpy()
        voksen = alder >= 16

        # --- tilb_stemme: målt alder × utd rundt totalen + vane/tillit ---
        ab = np.searchsorted(_STEMME_GRENSER, np.maximum(alder, 18), side="right")
        ab = np.clip(ab, 0, len(self.stemme_alder) - 1)
        utd_i = df["utdanning"].cat.codes.to_numpy()
        utd3 = np.clip(utd_i, 0, 2)                    # uni_kort/lang -> 6-8
        utd3 = np.where(utd_i == 3, 2, utd3)
        base = self.stemme_alder[ab] * np.where(
            utd_i >= 0, self.stemme_utd[utd3], self.stemme_tot) / max(self.stemme_tot, 1e-9)
        stemte = (~df["parti"].isin(["stemte ikke", "ikke stemmerett"])).to_numpy()
        hjemme = (df["parti"] == "stemte ikke").to_numpy()
        tillit = np.nan_to_num(df["institusjonstillit"].to_numpy(), nan=50.0)
        stemme = (base * 100.0 + stemte * 8.0 - hjemme * 8.0
                  + (tillit - 50.0) * 0.08 + rng.normal(0, 4, n))
        stemme = np.clip(stemme, 2.0, 98.0)

        # --- tilb_flytte: målt rate = flyttinger / kopiens befolkning -----
        grenser = [int("".join(ch for ch in b.split("-")[0] if ch.isdigit()) or 0)
                   for b in self._flytt_band[1:]]
        fb = np.searchsorted(grenser, alder, side="right")
        nevner = np.zeros((2, len(self._flytt_band)))
        np.add.at(nevner, (kjonn_i, fb), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(nevner > 0, self.flytt / nevner, 0.0)
        flytte = np.clip(rate[kjonn_i, fb] * 100.0 + rng.normal(0, 1.5, n), 0.5, 80.0)

        # --- tilb_frivillig: målt basis + antatte justeringer -------------
        vb = np.searchsorted([25, 45, 67], np.maximum(alder, 16), side="right")
        ptype = df["personlighetstype"].astype(str).to_numpy()
        utadvendt = np.char.startswith(ptype.astype(str), "E")
        if sentralitet is None:
            from .holdning import hent_sentralitet
            sentralitet = hent_sentralitet()
        klasse = df["kommune"].astype(str).map(
            lambda k: sentralitet.get(k, 4)).to_numpy()
        friv = (self.friv[kjonn_i, vb] * 100.0
                + utadvendt * 6.0
                + df["medlem_dnk"].to_numpy() * 5.0
                + (klasse >= 5) * 4.0
                + rng.normal(0, 5, n))
        friv = np.clip(friv, 1.0, 95.0)

        # --- tilb_protest: ren antakelse ----------------------------------
        trygghet = np.nan_to_num(df["trygghet"].to_numpy(), nan=50.0)
        protest = (0.55 * (100.0 - tillit) + 0.25 * (100.0 - trygghet)
                   + np.where(alder < 30, 8.0, 0.0) + rng.normal(0, 6, n))
        protest = np.clip(protest, 0.0, 100.0)

        # --- tilb_risiko: ren antakelse ------------------------------------
        spontan = np.char.endswith(ptype.astype(str), "P")
        alder_just = np.select([alder < 25, alder < 35, alder >= 65],
                               [15.0, 8.0, -10.0], default=0.0)
        siktet = df["siktet_vold"].fillna(False).to_numpy(dtype=bool)
        risiko = (38.0 + alder_just + (kjonn_i == 0) * 8.0
                  + utadvendt * 5.0 + spontan * 5.0 + siktet * 15.0
                  + rng.normal(0, 8, n))
        risiko = np.clip(risiko, 0.0, 100.0)

        ut = df.copy()
        for navn, verdi in [("tilb_stemme", stemme), ("tilb_flytte", flytte),
                            ("tilb_frivillig", friv), ("tilb_protest", protest),
                            ("tilb_risiko", risiko)]:
            ut[navn] = np.where(voksen, np.round(verdi, 1), np.nan).astype(np.float32)
        logger.info("Tilbøyeligheter (16+, snitt): stemme %.0f, flytte %.1f, "
                    "frivillig %.0f, protest %.0f, risiko %.0f.",
                    stemme[voksen].mean(), flytte[voksen].mean(),
                    friv[voksen].mean(), protest[voksen].mean(),
                    risiko[voksen].mean())
        return ut
