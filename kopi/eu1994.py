"""EU-avstemningen 1994 per kommune — MÅLT geo-anker for sentrum/distrikt.

SSB 01187 gir godkjente ja-/nei-stemmer per kommune i 1994-inndelingen
(435 kommuner). Kodene kjedes frem til dagens inndeling med Klass 131-
endringene (KlassClient.code_changes, dato for dato); de få 1994-kommunene
som siden er delt mellom flere nåværende kommuner (8 stk) får stemmene
LIKEDELT mellom målkommunene — en dokumentert tilnærming som kun berører
småkommuner. Kontroll etter mapping: nasjonal nei-andel 52.2 % (= fasit).

Dette gir sentrum/distrikt-aksen sin egen MÅLTE geografi — nei-andelen
spenner 25-94 % og følger et annet kart enn sentralitetsindeksen (kyst/
innland, nord/sør), så aksen slutter å være en ren kopi av by/land-
dimensjonen som de andre aksene også bærer.

ANTAKELSEN som gjør et 31 år gammelt valg til et holdningsanker: sentrum/
periferi-motsetningen (Rokkan-aksen) er den tregeste i norsk politikk, og
nei-andelen antas fortsatt å RANGERE kommunene riktig. Nivå og skala settes
i kopi/holdning.py og er antakelser der.
"""

from __future__ import annotations

import logging

import pandas as pd

from data import SSBClient, KlassClient

logger = logging.getLogger("simnorge.kopi.eu1994")

TABELL = "01187"
NASJONAL_NEI = 0.522          # offisielt resultat 1994 — kontrolltall


def hent_eu1994(ssb: SSBClient | None = None,
                klass: KlassClient | None = None,
                *, til_dato: str = "2025-01-01") -> dict[str, float]:
    """kommune (dagens kode) -> nei-andel 1994 (0-1)."""
    ssb = ssb or SSBClient()
    klass = klass or KlassClient()

    q = {"query": [
        {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "EFEUAvstemning",
         "selection": {"filter": "item", "values": ["01", "02"]}},
        {"code": "Tid", "selection": {"filter": "item", "values": ["1994"]}},
    ]}
    df = ssb.query(TABELL, q)
    piv = df.pivot_table(index="Region", columns="EFEUAvstemning",
                         values="value", aggfunc="sum").fillna(0.0)
    piv.columns = ["ja", "nei"]
    piv = piv[(piv["ja"] + piv["nei"]) > 0]
    piv = piv[piv.index.str.fullmatch(r"\d{4}")]

    # Kjed 1994-kodene fremover, endringsdato for endringsdato.
    ch = klass.code_changes("1994-01-01", til_dato)
    ch = ch[ch["old_code"] != ch["new_code"]]
    vekt: dict[str, dict[str, float]] = {k: {k: 1.0} for k in piv.index}
    for _, g in sorted(ch.groupby("occurred"), key=lambda x: x[0]):
        mapping = g.groupby("old_code")["new_code"].apply(list).to_dict()
        for kilde, maal in vekt.items():
            ny: dict[str, float] = {}
            for kode, andel in maal.items():
                for nk in mapping.get(kode, [kode]):
                    ny[nk] = ny.get(nk, 0.0) + andel / len(mapping.get(kode, [kode]))
            vekt[kilde] = ny

    splitt = sorted(k for k, v in vekt.items() if len(v) > 1)
    if splitt:
        logger.info("EU-1994: %d 1994-kommuner likedelt over flere dagens "
                    "kommuner: %s", len(splitt), splitt)

    ja: dict[str, float] = {}
    nei: dict[str, float] = {}
    for kilde, maal in vekt.items():
        for kode, andel in maal.items():
            ja[kode] = ja.get(kode, 0.0) + float(piv.loc[kilde, "ja"]) * andel
            nei[kode] = nei.get(kode, 0.0) + float(piv.loc[kilde, "nei"]) * andel

    ut = {k: nei[k] / (ja[k] + nei[k]) for k in ja}
    total = sum(nei.values()) / (sum(ja.values()) + sum(nei.values()))
    if abs(total - NASJONAL_NEI) > 0.005:
        raise ValueError(f"EU-1994-mapping avviker fra fasit: {total:.3f} "
                         f"mot {NASJONAL_NEI} — kodekjeding lekker stemmer.")
    logger.info("EU-1994: nei-andel for %d kommuner (nasjonalt %.1f %%, "
                "spenn %.1f-%.1f %%).", len(ut), total * 100,
                min(ut.values()) * 100, max(ut.values()) * 100)
    return ut


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    andeler = hent_eu1994()
    s = pd.Series(andeler).sort_values()
    print("\nMest ja (sentraliseringsvennlig side):")
    print((s.head(8) * 100).round(1).to_string())
    print("\nMest nei (distriktssiden):")
    print((s.tail(8) * 100).round(1).to_string())
