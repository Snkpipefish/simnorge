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

GEOJSON = Path(".cache/geo/kommuner.geojson")


def _dp(pts: list[tuple[int, int]], tol: float) -> list[tuple[int, int]]:
    """Douglas-Peucker (iterativ) på en punktliste i rutekoordinater."""
    n = len(pts)
    if n < 4:
        return pts
    behold = [False] * n
    behold[0] = behold[-1] = True
    stakk = [(0, n - 1)]
    while stakk:
        i, j = stakk.pop()
        ax, ay = pts[i]
        bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        best, bd = -1, tol
        for k in range(i + 1, j):
            px, py = pts[k]
            if l2 == 0:
                d = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
                d = ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5
            if d > bd:
                best, bd = k, d
        if best >= 0:
            behold[best] = True
            stakk.append((i, best))
            stakk.append((best, j))
    return [p for p, b in zip(pts, behold) if b]


def _geo() -> dict | None:
    """Kommunegrenser som kvantiserte, delta-kodede ringer for kartet.

    Kvantisering til felles rutenett (Q grader) gjør at nabogrenser knekker
    likt; Douglas-Peucker fjerner punkter under synlig størrelse på skjerm."""
    if not GEOJSON.exists():
        return None
    gj = json.loads(GEOJSON.read_text("utf-8"))
    q = 0.004   # grader per rute — ~0,4 km, under én piksel på skjermkartet
    tol = 1.5   # DP-toleranse i ruter

    def ring(r: list) -> list[int] | None:
        pts = [(round(x / q), round(y / q)) for x, y in r]
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        pts = _dp(pts, tol)
        if len(pts) < 3:
            return None
        flat, px, py = [], 0, 0
        for x, y in pts:
            flat += [x - px, y - py]
            px, py = x, y
        return flat

    kommuner = []
    for f in gj["features"]:
        geom = f["geometry"]
        polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        ringer = [fr for poly in polys for r in poly if (fr := ring(r))]
        if ringer:
            kommuner.append({"k": f["properties"]["kommunenummer"], "r": ringer})
    return {"q": q, "kommuner": kommuner}


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
    }
    grupper = gruppestat(v, velgere, "holdningsgruppe")

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

    data = {"nasjonal": nasjonal, "grupper": grupper,
            "hovedliste": HOVEDGRUPPER, "partier": PARTIER,
            "kommuner": kommuner, "geo": _geo()}
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
