"""Generer kopi_norge.html — selvstendig, interaktiv oversikt over kopien.

    .venv/bin/python lag_html.py          # leser kopi_norge.parquet

Én fil, ingen avhengigheter: alle aggregater bakes inn som JSON. Rådataene
(5,6 mill rader) blir værende i parquet — HTML-en bærer nasjonale tall,
alle 20 holdningsgrupper med full statistikk, og detaljer per kommune.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kopi.holdning import AKSER, HOVEDGRUPPE, HOVEDGRUPPER
from kopi.valg import PARTIER

TILB = ["tilb_stemme", "tilb_flytte", "tilb_frivillig", "tilb_protest", "tilb_risiko"]

# Kommunekorrelasjonene (Spearman) slik de var FØR geografien ble målt
# (én sentralitetsvektor for alle akser) — historisk referanse til
# holdningsgeografi-seksjonen. Nå-verdiene regnes ved bygging.
_KORR_FOER = {"innvandring×klima": 0.98, "sentrum×innvandring": 0.90,
              "sentrum×klima": 0.87, "øko×innvandring": 0.40}
_KORR_PAR = [("innvandring", "klima"), ("sentrum_distrikt", "innvandring"),
             ("sentrum_distrikt", "klima"), ("oko_fordeling", "innvandring")]
_KORR_NAVN = ["innvandring×klima", "sentrum×innvandring", "sentrum×klima",
              "øko×innvandring"]


def _geo(v: pd.DataFrame, eu1994: dict[str, float] | None) -> dict:
    """Holdningsgeografi-funnene: kommunekorrelasjoner før/nå per aksepar."""
    g = v.groupby("kommune", observed=True)[AKSER].mean()
    korr = {navn: {"foer": _KORR_FOER[navn],
                   "naa": round(float(g[a].rank().corr(g[b].rank())), 2)}
            for navn, (a, b) in zip(_KORR_NAVN, _KORR_PAR)}
    ut: dict = {"korr": korr}
    if eu1994:
        s = pd.Series(eu1994)
        ut["eu_spenn"] = [round(float(s.min()) * 100, 1),
                          round(float(s.max()) * 100, 1)]
    return ut


def _bokstaver(v: pd.DataFrame, velgere: pd.DataFrame) -> dict:
    """E/N/F/J-andeler (%) per inntektskvintil og per parti — tallene som
    viser de målte ESS-tiltene i praksis."""
    def andeler(g: pd.DataFrame) -> dict:
        t = g["personlighetstype"].astype(str)
        return {"E": round(float(t.str[0].eq("E").mean() * 100), 1),
                "N": round(float(t.str[1].eq("N").mean() * 100), 1),
                "F": round(float(t.str[2].eq("F").mean() * 100), 1),
                "P": round(float(t.str[3].eq("P").mean() * 100), 1)}

    m17 = v[v["alder"] >= 17].copy()
    m17["kvintil"] = pd.qcut(m17["brutto_inntekt"], 5, labels=False)
    kvintil = [andeler(g) for _, g in m17.groupby("kvintil", observed=True)]
    parti = {str(p): andeler(g)
             for p, g in velgere.groupby("parti", observed=True)}
    return {"kvintil": kvintil, "parti": parti}


def gruppestat(v: pd.DataFrame, velgere: pd.DataFrame, kol: str) -> list[dict]:
    ut = []
    for navn, g in v.groupby(kol, observed=True):
        vel = velgere[velgere[kol] == navn]
        topp = vel["parti"].mode()[0]
        rad = {
            "id": str(navn), "n": int(len(g)),
            "parti": f"{topp} ({(vel['parti'] == topp).mean() * 100:.0f} %)",
            "inntekt": int(g["brutto_inntekt"].median() // 1000),
            "alder": int(g["alder"].median()),
            "trygghet": round(float(g["trygghet"].mean())),
            "tillit": round(float(g["institusjonstillit"].mean())),
            "verdi": round(float(g["verdiliberal"].mean())),
            "dnk": round(float(g["medlem_dnk"].mean() * 100)),
        }
        if kol == "holdningsgruppe":
            rad["hoved"] = HOVEDGRUPPE[str(navn)]
        for t in TILB:
            rad[t] = round(float(g[t].mean()), 1)
        ut.append(rad)
    return ut


def main() -> None:
    df = pd.read_parquet("kopi_norge.parquet")
    v = df[df["alder"] >= 16]
    velgere = v[~v["parti"].isin(["stemte ikke", "ikke stemmerett"])]

    try:
        from kopi.eu1994 import hent_eu1994
        eu1994 = hent_eu1994()
    except Exception as e:                     # siden skal kunne bygges offline
        print(f"EU-1994 utilgjengelig ({e}) — kommunekort uten nei-andel.")
        eu1994 = None

    nasjonal = {
        "total": int(len(df)),
        "voksne": int(len(v)),
        "velgere": int(len(velgere)),
        "kommuner": int(df["kommune"].nunique()),
        "parti": {p: round(float((velgere["parti"] == p).mean() * 100), 1)
                  for p in PARTIER},
        "inntekt": {k: int(g["brutto_inntekt"].median())
                    for k, g in df[df["alder"] >= 17].groupby("kjonn", observed=True)},
        "personlighet": {k: round(float(n / len(df) * 100), 1) for k, n in
                         df["personlighetstype"].value_counts().items()},
        "utsatt_vold": round(float(v["utsatt_vold"].mean() * 100), 1),
        "dnk": round(float(df["medlem_dnk"].mean() * 100), 1),
        "helse": {
            "psykisk": round(float(df["psykisk_kontakt"].mean() * 100), 1),
            "muskel": round(float(df["muskel_kontakt"].mean() * 100), 1),
            "helse_god": round(float(v["helse_god"].mean() * 100), 1),
            "levealder_menn": round(float(
                df[df["kjonn"] == "mann"]["forventet_levealder"].mean()), 1),
            "levealder_kvinner": round(float(
                df[df["kjonn"] == "kvinne"]["forventet_levealder"].mean()), 1),
        },
        "bokstaver": _bokstaver(v, velgere),
        "tilb_hoved": {str(h): {t: round(float(g[t].mean()), 1) for t in TILB}
                       for h, g in v.groupby("hovedgruppe", observed=True)},
    }
    grupper = gruppestat(v, velgere, "holdningsgruppe")
    hoved = gruppestat(v, velgere, "hovedgruppe")

    kommuner = []
    aldersbins = list(range(0, 100, 10))
    for kode, g in df.groupby("kommune", observed=True):
        g16 = g[g["alder"] >= 16]
        vel = g16[~g16["parti"].isin(["stemte ikke", "ikke stemmerett"])]
        hg = (g16["holdningsgruppe"].value_counts(normalize=True) * 100).round(1)
        rad_akser = [round(float(g16[a].mean())) for a in AKSER]
        kommuner.append({
            "kode": str(kode),
            "navn": str(g["kommune_navn"].iloc[0]),
            "akser": rad_akser,
            **({"eu1994": round(eu1994[str(kode)] * 100, 1)}
               if eu1994 and str(kode) in eu1994 else {}),
            "n": int(len(g)),
            "inntekt": int(g["brutto_inntekt"].median() // 1000),
            "parti": {p: round(float((vel["parti"] == p).mean() * 100), 1)
                      for p in PARTIER if (vel["parti"] == p).any()},
            "deltakelse": round(float(len(vel) / max(len(g16[g16["alder"] >= 18]), 1) * 100)),
            "trygghet": round(float(g16["trygghet"].mean())),
            "tillit": round(float(g16["institusjonstillit"].mean())),
            "dnk": round(float(g["medlem_dnk"].mean() * 100)),
            "grupper": [[k, float(x)] for k, x in hg.head(5).items()],
            "alder": [int(((g["alder"] >= a) & (g["alder"] < a + 10)).sum())
                      for a in aldersbins] + [int((g["alder"] >= 100).sum())],
            "levealder": round(float(g["forventet_levealder"].mean()), 1),
            "psykisk": round(float(g["psykisk_kontakt"].mean() * 100), 1),
            "helse_god": round(float(g16["helse_god"].mean() * 100), 1),
            "tilb_stemme": round(float(g16["tilb_stemme"].mean()), 1),
            "tilb_protest": round(float(g16["tilb_protest"].mean()), 1),
        })
    kommuner.sort(key=lambda k: k["navn"])

    data = {"nasjonal": nasjonal, "grupper": grupper, "hoved": hoved,
            "hovedliste": HOVEDGRUPPER, "partier": PARTIER,
            "geo": _geo(v, eu1994), "kommuner": kommuner}
    mal = Path("kopi/oversikt_mal.html").read_text(encoding="utf-8")
    html = mal.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                              separators=(",", ":")))
    Path("kopi_norge.html").write_text(html, encoding="utf-8")
    # Publiseringskopi for GitHub Pages (docs/ på main).
    Path("docs").mkdir(exist_ok=True)
    Path("docs/index.html").write_text(html, encoding="utf-8")
    print(f"Skrev kopi_norge.html + docs/index.html ({len(html) / 1e6:.1f} MB)"
          f" — {len(kommuner)} kommuner, {len(grupper)} grupper.")


if __name__ == "__main__":
    main()
