"""Livssyn — medlemskap i Den norske kirke, forankret i kommunens MÅLTE rate.

12025 (KOSTRA kirke) gir per kommune: «Medlem og tilhørige i Dnk i prosent av
antall innbyggere» og «Medlemmer i tros- og livssynssamfunn utenfor Dnk i
prosent». Begge er ekte, ferske tall per kommune.

Per person trekkes Dnk-medlemskap slik at KOMMUNENS rate treffes eksakt i
forventning, med en ANTATT aldersgradient oppå (eldre er oftere medlem —
dåpstallene faller år for år): faktor per aldersbånd [0-15, 16-24, 25-44,
45-66, 67+] = [0.90, 0.80, 0.85, 1.10, 1.35], reskalert per kommune mot
kommunens aldersmiks slik at totalraten bevares. Medlemskap utenfor Dnk
trekkes flatt på kommuneraten (ingen kilde for gradient; NB inkluderer både
religiøse samfunn og f.eks. Human-Etisk Forbund).

Prikkede kommuner får landssnittet (logget).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient

logger = logging.getLogger("simnorge.kopi.livssyn")

_BAND_GRENSER = [16, 25, 45, 67]
_DNK_ALDERSFAKTOR = np.array([0.90, 0.80, 0.85, 1.10, 1.35])   # ANTAKELSE


class LivssynModell:
    def __init__(self, ssb: SSBClient):
        self.dnk, self.andre = self._last_12025(ssb)

    def _last_12025(self, ssb: SSBClient):
        koder = ssb.variable_codes("12025")
        year = koder["Tid"][-1]
        cc = ["KOSmedldnkinnb0000", "KOSmedltrolinnb0000"]
        df = ssb.fetch("12025", {
            "KOKkommuneregion0000": koder["KOKkommuneregion0000"],
            "ContentsCode": cc, "Tid": [year],
        })
        ut = []
        for kode in cc:
            sub = df[(df["ContentsCode"] == kode) & ~df["missing"]
                     & df["value"].notna()]
            rater = {str(r["KOKkommuneregion0000"]): float(r["value"]) / 100.0
                     for _, r in sub.iterrows()}
            ut.append(rater)
        logger.info("12025 (%s): Dnk-rate for %d kommuner, andre tros-/livssyns"
                    "samfunn for %d.", year, len(ut[0]), len(ut[1]))
        return ut[0], ut[1]

    def tildel(self, df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
        n = len(df)
        rng = np.random.default_rng(seed)
        alder = df["alder"].to_numpy()
        band = np.searchsorted(_BAND_GRENSER, alder, side="right")

        dnk_snitt = float(np.mean(list(self.dnk.values()))) if self.dnk else 0.62
        andre_snitt = float(np.mean(list(self.andre.values()))) if self.andre else 0.07

        p_dnk = np.empty(n)
        kommune_str = df["kommune"].astype(str).to_numpy()
        mangler: set[str] = set()
        for kom in np.unique(kommune_str):
            maske = kommune_str == kom
            rate = self.dnk.get(kom)
            if rate is None:
                mangler.add(kom); rate = dnk_snitt
            fakt = _DNK_ALDERSFAKTOR[band[maske]]
            # Reskaler gradienten mot kommunens aldersmiks -> raten bevares.
            p_dnk[maske] = np.clip(rate * fakt / fakt.mean(), 0.0, 1.0)
        if mangler:
            logger.warning("12025: %d kommuner uten Dnk-rate fikk landssnitt "
                           "%.2f (logget).", len(mangler), dnk_snitt)

        dnk = rng.random(n) < p_dnk
        p_andre = pd.Series(kommune_str).map(
            lambda k: self.andre.get(k, andre_snitt)).to_numpy()
        # Ikke medlem to steder: andre-samfunn trekkes blant ikke-Dnk.
        andre = (~dnk) & (rng.random(n) < np.clip(
            p_andre / np.maximum(1.0 - p_dnk, 1e-9), 0.0, 1.0))

        ut = df.copy()
        ut["medlem_dnk"] = dnk
        ut["medlem_annet_livssyn"] = andre
        logger.info("Livssyn: %.1f %% Dnk, %.1f %% andre samfunn (kommunens "
                    "målte rater; aldersgradient antatt).",
                    dnk.mean() * 100, andre.mean() * 100)
        return ut
