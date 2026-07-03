"""ESS-laget — MÅLTE partiposisjoner på holdningsaksene.

European Social Survey runde 10 (2020-22, valgminne 2017/2021) og 11
(2023-24, valgminne 2021), norsk utvalg (~2 750 respondenter, hvorav ~1 950
med partivalg). Mikrodataene ligger i data/ess/ (lastet ned av eier fra
ess.sikt.no — gratis, registreringspliktig, ikke-kommersiell bruk).

Dette erstatter ANTAKELSENE om partiposisjoner i kopi/holdning.py og
kopi/tillit_verdi.py med MÅLTE verdier — inkludert målt spredning INNEN
hvert parti, og målt posisjon for hjemmesittere (vote == 2):

- oko_fordeling:  gincdif (omfordeling; 1 helt enig -> 0, 5 helt uenig -> 100)
- innvandring:    snitt av 6 mål (imsmetn/imdfetn/impcntr snudd, imbgeco/
                  imueclt/imwbcnt direkte), alle -> 0-100 der høy = liberal
- klima:          wrclmch (bekymring 1-5) + ccrdprs (personlig ansvar 0-10)
- institusjonstillit: snitt av trstprl + trstplt (0-10 -> 0-100)
- verdiliberal:   snitt av freehms (snudd) + rlgdgr (snudd)

I tillegg måles inntektsgradienten på økonomiaksen (hinctnta-desil mot
landssnittet) — den erstatter det antatte ±12-skyvet.

sentrum_distrikt har ikke noe ESS-mål og forblir antakelse. Aggregatene
lagres i kopi/ess_posisjoner.json; selve mikrodataene forlater aldri maskinen.
Partikoder verifisert både mot kodebok og empirisk (MDG-koden identifisert
via klimabekymring; utvalgsandeler mot faktisk valgresultat).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("simnorge.kopi.ess")

CSV = Path("data/ess/ESS10e03_3-ESS11e04_2-subset.csv")
JSON_UT = Path(__file__).parent / "ess_posisjoner.json"

# Runde -> (partikolonne, kode -> vårt partinavn).
_PARTIKODER = {
    10: ("prtvtbno", {1: "Rødt", 2: "SV", 3: "Ap", 4: "Venstre", 5: "KrF",
                      6: "Sp", 7: "Høyre", 8: "FrP", 10: "MDG", 11: "Andre"}),
    11: ("prtvtcno", {1: "Rødt", 2: "SV", 3: "Ap", 4: "Venstre", 5: "KrF",
                      6: "Sp", 7: "Høyre", 8: "FrP", 9: "MDG", 11: "Andre"}),
}
AKSER_ESS = ["oko_fordeling", "innvandring", "klima",
             "institusjonstillit", "verdiliberal"]

# Personlighetsbokstav -> ESS-mål (kolonne, maks gyldig kode, snudd skala).
# Schwartz-batteriet (1 = «svært lik meg» .. 6 = «ikke lik meg i det hele
# tatt», snudd) + målt sosial aktivitet. KOBLINGEN bokstav<->mål er en
# dokumentert ANTAKELSE (Schwartz/sosialitet som proxy for MBTI-bokstavene);
# gruppeforskjellene den henter ut er MÅLTE.
_BOKSTAV_MAAL = {
    "E": [("sclmeet", 7, False), ("sclact", 5, False)],
    "N": [("ipcrtiv", 6, True), ("impdiff", 6, True), ("ipadvnt", 6, True)],
    "F": [("iphlppl", 6, True), ("ipeqopt", 6, True), ("ipudrst", 6, True)],
    "J": [("impsafe", 6, True), ("ipfrule", 6, True), ("imptrad", 6, True),
          ("ipbhprp", 6, True)],
}
# eisced -> våre utdanningsnivåer (kopi/utdanning.NIVAER-indeks).
_EISCED_TIL_NIVAA = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3}


def _gyldig(s: pd.Series, maks: float) -> pd.Series:
    """ESS-manglende koder (6/7/8/9, 66/77/88/99) -> NaN."""
    return s.where(s <= maks)


def _akser(df: pd.DataFrame) -> pd.DataFrame:
    ut = pd.DataFrame(index=df.index)
    ut["oko_fordeling"] = (_gyldig(df["gincdif"], 5) - 1) / 4 * 100

    innv = pd.concat([
        (4 - _gyldig(df["imsmetn"], 4)) / 3,
        (4 - _gyldig(df["imdfetn"], 4)) / 3,
        (4 - _gyldig(df["impcntr"], 4)) / 3,
        _gyldig(df["imbgeco"], 10) / 10,
        _gyldig(df["imueclt"], 10) / 10,
        _gyldig(df["imwbcnt"], 10) / 10,
    ], axis=1)
    ut["innvandring"] = innv.mean(axis=1) * 100

    klima = pd.concat([
        (_gyldig(df["wrclmch"], 5) - 1) / 4,
        _gyldig(df["ccrdprs"], 10) / 10,
    ], axis=1)
    ut["klima"] = klima.mean(axis=1) * 100

    tillit = pd.concat([_gyldig(df["trstprl"], 10),
                        _gyldig(df["trstplt"], 10)], axis=1)
    ut["institusjonstillit"] = tillit.mean(axis=1) * 10

    verdi = pd.concat([
        (5 - _gyldig(df["freehms"], 5)) / 4,
        (10 - _gyldig(df["rlgdgr"], 10)) / 10,
    ], axis=1)
    ut["verdiliberal"] = verdi.mean(axis=1) * 100
    return ut


def _vektet(v: pd.Series, w: pd.Series) -> tuple[float, float, int]:
    m = v.notna() & w.notna()
    v, w = v[m], w[m]
    if len(v) < 5:
        return np.nan, np.nan, int(len(v))
    mu = float(np.average(v, weights=w))
    sd = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
    return mu, sd, int(len(v))


def _koalesser(df: pd.DataFrame, kol: str) -> pd.Series:
    """Runde 11 bruker a-varianter av Schwartz-kolonnene — slå sammen."""
    s = df[kol] if kol in df.columns else pd.Series(np.nan, index=df.index)
    if kol + "a" in df.columns:
        s = s.where(s.notna(), df[kol + "a"])
    return s


def _bokstav_komposit(df: pd.DataFrame) -> pd.DataFrame:
    ut = pd.DataFrame(index=df.index)
    for bokstav, maal in _BOKSTAV_MAAL.items():
        deler = []
        for kol, maks, snudd in maal:
            v = _gyldig(_koalesser(df, kol), maks)
            deler.append((maks - v) / (maks - 1) if snudd else (v - 1) / (maks - 1))
        ut[bokstav] = pd.concat(deler, axis=1).mean(axis=1)
    return ut


def _gruppe_z(komp: pd.Series, w: pd.Series, grupper: pd.Series,
              nokler, *, min_n: int = 20) -> dict:
    """Vektet snitt per gruppe i z-enheter (mot vektet totalsnitt/-sd)."""
    mu, sd, _ = _vektet(komp, w)
    ut = {}
    for g in nokler:
        m, _, n = _vektet(komp[grupper == g], w[grupper == g])
        ut[str(g)] = round((m - mu) / sd, 3) if n >= min_n and not np.isnan(m) else None
    return ut


def beregn_personlighet(df: pd.DataFrame, parti: pd.Series) -> dict:
    """MÅLTE gruppeavvik (z) per bokstavkomposit for inntektsdesil,
    utdanningsnivå og parti. Kjønn/alder holdes utenfor med vilje — de
    ligger allerede i basissatsene i kopi/personlighet.py."""
    komp = _bokstav_komposit(df)
    w = df["anweight"]
    desil = _gyldig(df["hinctnta"], 10)
    utd = _gyldig(df["eisced"], 7).map(_EISCED_TIL_NIVAA)
    ut = {}
    for bokstav in _BOKSTAV_MAAL:
        desil_z = _gruppe_z(komp[bokstav], w, desil, range(1, 11))
        desil_z = pd.Series([desil_z[str(d)] for d in range(1, 11)],
                            dtype="float").interpolate(
            limit_direction="both").fillna(0.0).round(3).tolist()
        utd_z = _gruppe_z(komp[bokstav], w, utd, range(4))
        parti_z = _gruppe_z(komp[bokstav], w, parti, parti.dropna().unique())
        ut[bokstav] = {
            "inntektsdesil": desil_z,
            "utdanning": {k: (v if v is not None else 0.0)
                          for k, v in utd_z.items()},
            "parti": {k: (v if v is not None else 0.0)
                      for k, v in parti_z.items()},
        }
    return ut


def beregn_posisjoner(csv: Path = CSV) -> dict:
    df = pd.read_csv(csv, low_memory=False)
    df = df[df["cntry"] == "NO"].copy()
    parti = pd.Series(pd.NA, index=df.index, dtype="object")
    for runde, (kol, koder) in _PARTIKODER.items():
        m = df["essround"] == runde
        parti[m] = df.loc[m, kol].map(koder)
    hjemme = _gyldig(df["vote"], 2) == 2
    parti[hjemme & parti.isna()] = "stemte ikke"

    akser = _akser(df)
    w = df["anweight"]
    ut: dict = {"kilde": f"ESS10+ESS11 Norge, n={len(df)}", "akser": {}}
    for akse in AKSER_ESS:
        rad: dict = {}
        mu_n, sd_n, n_n = _vektet(akser[akse], w)
        rad["_alle"] = {"mean": round(mu_n, 1), "sd": round(sd_n, 1), "n": n_n}
        for p in parti.dropna().unique():
            mu, sd, n = _vektet(akser[akse][parti == p], w[parti == p])
            if not np.isnan(mu):
                rad[p] = {"mean": round(mu, 1), "sd": round(sd, 1), "n": n}
        ut["akser"][akse] = rad

    # Målt inntektsgradient på økonomiaksen: desilavvik fra landssnittet.
    desil = _gyldig(df["hinctnta"], 10)
    mu_n = ut["akser"]["oko_fordeling"]["_alle"]["mean"]
    avvik = []
    for d in range(1, 11):
        mu, _, n = _vektet(akser["oko_fordeling"][desil == d], w[desil == d])
        avvik.append(round(mu - mu_n, 1) if not np.isnan(mu) and n >= 20 else None)
    # Hull (små desiler) fylles lineært mellom naboene.
    verdier = pd.Series(avvik, dtype="float").interpolate(
        limit_direction="both").round(1).tolist()
    ut["oko_inntektsdesil"] = verdier

    ut["personlighet"] = beregn_personlighet(df, parti)
    return ut


def skriv_posisjoner(csv: Path = CSV) -> dict:
    ut = beregn_posisjoner(csv)
    JSON_UT.write_text(json.dumps(ut, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    logger.info("Skrev %s (%s).", JSON_UT, ut["kilde"])
    return ut


def les_posisjoner() -> dict | None:
    """Aggregatene hvis de er beregnet — None ellers (da gjelder antakelsene)."""
    if not JSON_UT.exists():
        return None
    return json.loads(JSON_UT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ut = skriv_posisjoner()
    for akse, rad in ut["akser"].items():
        print(f"\n== {akse} (alle: {rad['_alle']['mean']})")
        for p, v in sorted((k, v) for k, v in rad.items() if k != "_alle"):
            print(f"  {p:14s} {v['mean']:5.1f} (sd {v['sd']:4.1f}, n {v['n']})")
    print("\nInntektsdesil-avvik øko:", ut["oko_inntektsdesil"])
