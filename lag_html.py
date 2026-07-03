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

from kopi.holdning import HOVEDGRUPPE, HOVEDGRUPPER
from kopi.valg import PARTIER

TILB = ["tilb_stemme", "tilb_flytte", "tilb_frivillig", "tilb_protest", "tilb_risiko"]


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
    }
    grupper = gruppestat(v, velgere, "holdningsgruppe")
    hoved = gruppestat(v, velgere, "hovedgruppe")

    kommuner = []
    aldersbins = list(range(0, 100, 10))
    for kode, g in df.groupby("kommune", observed=True):
        g16 = g[g["alder"] >= 16]
        vel = g16[~g16["parti"].isin(["stemte ikke", "ikke stemmerett"])]
        hg = (g16["holdningsgruppe"].value_counts(normalize=True) * 100).round(1)
        kommuner.append({
            "kode": str(kode),
            "navn": str(g["kommune_navn"].iloc[0]),
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
        })
    kommuner.sort(key=lambda k: k["navn"])

    data = {"nasjonal": nasjonal, "grupper": grupper, "hoved": hoved,
            "hovedliste": HOVEDGRUPPER, "partier": PARTIER, "kommuner": kommuner}
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
