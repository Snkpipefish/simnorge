"""Bruttoinntekt per person — trukket fra SSBs faktiske fordeling (06655).

06655 gir ANTALL personer per inntektsintervall × kjønn × aldersbånd (17+),
nasjonalt. Hver person får et intervall trukket fra sin (kjønn, aldersbånd)-
fordeling, og et beløp uniformt innen intervallet (toppintervallet 2 mill+
får en Pareto-hale — antakelse, dokumentert under).

Geografi: 06655 finnes ikke per kommune. Vi skalerer derfor personinntekten
med kommunens medianhusholdningsinntekt (06944) relativt til landet, klippet
til [0.75, 1.5]. Det er en ANTAKELSE (husholdningsmedian som proxy for
personinntektsnivå), ikke en måling — men den gjør at Bærum ligger over
Kautokeino, som en kopi skal.

Personer under 17 år får ingen inntekt (NaN) — kilden dekker 17+.
"""

from __future__ import annotations

import logging

import numpy as np

from data import SSBClient

logger = logging.getLogger("simnorge.kopi.inntekt")

# 06655-intervallkoder i stigende rekkefølge med kronegrensene [lo, hi).
_INTERVALLER = [
    ("60", 0, 100_000), ("61", 100_000, 200_000), ("62", 200_000, 300_000),
    ("08", 300_000, 400_000), ("09", 400_000, 500_000),
    ("70", 500_000, 750_000), ("71", 750_000, 1_000_000),
    ("72", 1_000_000, 2_000_000), ("73", 2_000_000, None),
]
_ALDERSBAND = ["17-24", "25-34", "35-44", "45-54", "55-66", "67+"]
_BAND_GRENSER = [25, 35, 45, 55, 67]      # searchsorted på alder>=17
_KJONN = ["1", "2"]                        # mann, kvinne (indeks 0/1)

# Pareto-hale for toppintervallet (antakelse: alfa ~1.8 for norsk topphale).
_PARETO_ALFA = 1.8
_TAK = 30_000_000


class InntektsModell:
    """Nasjonal intervallfordeling per (kjønn, aldersbånd) + kommunefaktor."""

    def __init__(self, ssb: SSBClient, *, year: str | None = None):
        self.year = year or ssb.variable_codes("06655")["Tid"][-1]
        self.p = self._last_fordeling(ssb)        # (2, 6, 9)
        self.kommunefaktor = self._last_faktor(ssb)

    def _last_fordeling(self, ssb: SSBClient) -> np.ndarray:
        df = ssb.fetch("06655", {
            "BruttoInn": [k for k, _, _ in _INTERVALLER],
            "Alder": _ALDERSBAND, "Kjonn": _KJONN,
            "ContentsCode": ["Personar"], "Tid": [self.year],
        })
        p = np.zeros((2, len(_ALDERSBAND), len(_INTERVALLER)))
        for si, sex in enumerate(_KJONN):
            for ai, band in enumerate(_ALDERSBAND):
                for bi, (code, _, _) in enumerate(_INTERVALLER):
                    r = df[(df["Kjonn"] == sex) & (df["Alder"] == band)
                           & (df["BruttoInn"] == code)]
                    v = 0.0 if r.empty or bool(r.iloc[0]["missing"]) else float(r.iloc[0]["value"])
                    p[si, ai, bi] = v
        s = p.sum(axis=2, keepdims=True)
        if (s == 0).any():
            raise ValueError("06655: tom fordeling i minst én (kjønn, alder)-celle.")
        logger.info("06655 (%s): inntektsfordeling lastet for 2 kjønn × %d "
                    "aldersbånd × %d intervaller.", self.year,
                    len(_ALDERSBAND), len(_INTERVALLER))
        return p / s

    def _last_faktor(self, ssb: SSBClient) -> dict[str, float]:
        """kommune -> medianhusholdningsinntekt relativt til landet (klippet)."""
        koder = ssb.variable_codes("06944")
        year = koder["Tid"][-1]
        # Region må filtreres eksplisitt — ellers eliminerer klienten dimensjonen.
        df = ssb.fetch("06944", {
            "Region": koder["Region"], "HusholdType": ["0000"],
            "ContentsCode": ["SamletInntekt"], "Tid": [year],
        })
        nat_row = df[df["Region"] == "0"]
        nat = float(nat_row.iloc[0]["value"]) if not nat_row.empty else np.nan
        if np.isnan(nat) or nat <= 0:
            logger.warning("06944: mangler landstall — ingen geografisk "
                           "inntektsskala (faktor 1 overalt).")
            return {}
        ut = {}
        for _, r in df.iterrows():
            if r["Region"] == "0" or bool(r["missing"]) or np.isnan(r["value"]):
                continue
            ut[str(r["Region"])] = float(np.clip(r["value"] / nat, 0.75, 1.5))
        logger.info("06944 (%s): kommunefaktor for %d regioner "
                    "(median husholdning / land, klippet [0.75, 1.5]).", year, len(ut))
        return ut

    def trekk(self, alder: np.ndarray, kjonn_i: np.ndarray,
              kommune: np.ndarray, *, seed: int = 0) -> np.ndarray:
        """Bruttoinntekt i kroner per person. NaN for personer under 17."""
        n = len(alder)
        rng = np.random.default_rng(seed)
        ut = np.full(n, np.nan, dtype=np.float64)
        voksen = alder >= 17
        if not voksen.any():
            return ut

        band = np.searchsorted(_BAND_GRENSER, alder[voksen], side="right")
        cum = np.cumsum(self.p[kjonn_i[voksen], band], axis=1)
        u = rng.random(voksen.sum())
        bi = (u[:, None] >= cum).sum(axis=1).clip(0, len(_INTERVALLER) - 1)

        lo = np.array([iv[1] for iv in _INTERVALLER], dtype=float)[bi]
        hi_arr = np.array([iv[2] if iv[2] else 0 for iv in _INTERVALLER], dtype=float)[bi]
        v = rng.random(len(bi))
        belop = lo + v * (hi_arr - lo)
        topp = bi == len(_INTERVALLER) - 1
        if topp.any():
            belop[topp] = np.minimum(
                2_000_000 * (1.0 - v[topp]) ** (-1.0 / _PARETO_ALFA), _TAK)

        faktor = np.array([self.kommunefaktor.get(k, 1.0)
                           for k in kommune[voksen]])
        ut[voksen] = np.round(belop * faktor, -3)
        logger.info("Trakk inntekt for %d personer 17+ (nasjonal fordeling "
                    "06655 × kommunefaktor 06944).", int(voksen.sum()))
        return ut
