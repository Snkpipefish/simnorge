"""Personlighetstype (16 typer) — antatte basissatser + MÅLTE gruppetilt.

Ingen åpen norsk kilde måler MBTI-typer direkte. To lag:

1. BASISSATSER (antakelse, inspirert av publiserte normutvalg):
   - E/I: om lag 50/50, svakt fallende utadvendthet med alder.
   - S/N: ~70/30 i befolkningen.
   - T/F: menn oftere T, kvinner oftere F (60/40 mot 38/62).
   - J/P: J-andelen stiger med alder.

2. MÅLTE TILT (fra 2026-07, når ESS-aggregatene finnes): ESS-mikrodataene
   (Schwartz-verdier + sosial aktivitet, norsk utvalg) gir målte gruppe-
   avvik per bokstavkomposit for INNTEKTSDESIL, UTDANNING og PARTI — se
   kopi/ess.py. Koblingen komposit<->bokstav er en dokumentert antakelse
   (E<-sosialitet, N<-kreativitet/eventyr, F<-hjelpsomhet/likhet,
   J<-trygghet/regler/tradisjon); gradientene den flytter på er målte.
   Kjønn/alder ligger i basissatsene og holdes UTENFOR tiltene (ingen
   dobbelttelling). Uten ESS-aggregater: kun basissatsene (som før).

Barn under 16 får kun basissatsene (ESS dekker ikke barn, og de mangler
inntekt/utdanning/parti).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("simnorge.kopi.personlighet")

# De 16 typene i bitrekkefølge: (E=0/I=1, S=0/N=1, T=0/F=1, J=0/P=1).
TYPER = [f"{a}{b}{c}{d}" for a in "EI" for b in "SN" for c in "TF" for d in "JP"]

# Aldersbånd for tiltene (barn får også type — hele befolkningen simuleres).
_BAND_GRENSER = [16, 25, 45, 67]          # 0-15, 16-24, 25-44, 45-66, 67+

# P(E) per aldersbånd — svakt fallende med alder (antakelse).
_P_E = np.array([0.54, 0.53, 0.51, 0.49, 0.47])
# P(S) — flat (antakelse: ~70 % sansende i befolkningen).
_P_S = 0.70
# P(T) per kjønn [mann, kvinne] — dokumentert kjønnsforskjell i normutvalg.
_P_T = np.array([0.60, 0.38])
# P(J) per aldersbånd — stigende med alder (antakelse).
_P_J = np.array([0.40, 0.45, 0.55, 0.62, 0.66])

# Målt z-avvik -> prosentpoeng på bokstavsannsynligheten (antatt styrke),
# og demping fordi inntekt/utdanning/parti er korrelerte (additiv sum
# dobbeltteller ellers).
_ESS_GEVINST = 12.0
_ESS_DEMPING = 0.7


def tildel_personlighet(alder: np.ndarray, kjonn_i: np.ndarray,
                        *, seed: int = 0) -> np.ndarray:
    """Trekk én av de 16 typene per person.

    ``alder``: heltallsalder. ``kjonn_i``: 0=mann, 1=kvinne.
    Returnerer heltallskoder 0..15 inn i ``TYPER``. Deterministisk gitt seed.
    """
    n = len(alder)
    band = np.searchsorted(_BAND_GRENSER, alder, side="right")   # 0..4
    rng = np.random.default_rng(seed)
    u = rng.random((4, n))

    i_bit = (u[0] >= _P_E[band]).astype(np.int8)      # 0=E, 1=I
    n_bit = (u[1] >= _P_S).astype(np.int8)            # 0=S, 1=N
    f_bit = (u[2] >= _P_T[kjonn_i]).astype(np.int8)   # 0=T, 1=F
    p_bit = (u[3] >= _P_J[band]).astype(np.int8)      # 0=J, 1=P

    koder = (i_bit << 3) | (n_bit << 2) | (f_bit << 1) | p_bit
    logger.info("Tildelte personlighetstype til %d personer (basissatser, "
                "uavhengige bokstaver gitt kjønn/aldersbånd).", n)
    return koder


def tildel_personlighet_ess(df: pd.DataFrame, ess_pers: dict,
                            *, seed: int = 0) -> np.ndarray:
    """Typetildeling der bokstavsannsynlighetene i tillegg tiltes av MÅLTE
    gruppeavvik (inntektsdesil, utdanning, parti) fra ESS — se moduldocstring.

    ``df`` må ha alder, kjonn, brutto_inntekt, utdanning, parti.
    Returnerer koder 0..15 inn i TYPER. Deterministisk gitt seed.
    """
    import pandas as pd  # noqa: F811

    n = len(df)
    alder = df["alder"].to_numpy()
    kjonn_i = df["kjonn"].cat.codes.to_numpy()
    band = np.searchsorted(_BAND_GRENSER, alder, side="right")
    rng = np.random.default_rng(seed)
    u = rng.random((4, n))

    # Målte z-summer per bokstav (kun voksne; barn får 0-tilt).
    inntekt = df["brutto_inntekt"].to_numpy()
    har = ~np.isnan(inntekt)
    pct = np.full(n, 0.5)
    pct[har] = pd.Series(inntekt[har]).rank(pct=True).to_numpy()
    desil = np.clip((pct * 10).astype(int), 0, 9)
    utd_i = df["utdanning"].cat.codes.to_numpy()          # -1 for barn
    parti = df["parti"].astype(str).to_numpy()

    def z_sum(bokstav: str) -> np.ndarray:
        g = ess_pers[bokstav]
        z = np.array(g["inntektsdesil"], dtype=float)[desil]
        utd_v = np.array([g["utdanning"].get(str(i), 0.0) for i in range(4)])
        z = z + np.where(utd_i >= 0, utd_v[np.clip(utd_i, 0, 3)], 0.0)
        pmap = g["parti"]
        z = z + np.array([pmap.get(p, 0.0) for p in np.unique(parti)])[
            np.searchsorted(np.unique(parti), parti)]
        z[alder < 16] = 0.0
        return z * _ESS_DEMPING * _ESS_GEVINST / 100.0

    p_e = np.clip(_P_E[band] + z_sum("E"), 0.05, 0.95)
    p_n = np.clip((1.0 - _P_S) + z_sum("N"), 0.05, 0.95)
    p_f = np.clip((1.0 - _P_T[kjonn_i]) + z_sum("F"), 0.05, 0.95)
    p_j = np.clip(_P_J[band] + z_sum("J"), 0.05, 0.95)

    i_bit = (u[0] >= p_e).astype(np.int8)
    n_bit = (u[1] < p_n).astype(np.int8)
    f_bit = (u[2] < p_f).astype(np.int8)
    p_bit = (u[3] >= p_j).astype(np.int8)
    koder = (i_bit << 3) | (n_bit << 2) | (f_bit << 1) | p_bit
    logger.info("Tildelte personlighetstype til %d personer (basissatser + "
                "målte ESS-tilt for inntekt/utdanning/parti).", n)
    return koder
