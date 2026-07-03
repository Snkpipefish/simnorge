"""Høynivå-uttrekk bygget på SSBClient.

Disse funksjonene gir rene, navngitte DataFrames per lag, slik at de senere
modulene (populasjon, holdningsberikelse, kalibrering) slipper å kjenne til
dimensjonskodene i hver tabell.
"""

from __future__ import annotations

import pandas as pd

from .ssb_client import SSBClient
from .tables import MAIN_PARTY_CODES, PARTY_CODES


# --------------------------------------------------------------------------- #
# Fasitlag — valg (08092 stortingsvalg, 01180 kommunestyrevalg)               #
# --------------------------------------------------------------------------- #
def election_shares(
    client: SSBClient,
    region: str,
    year: str = "2025",
    *,
    table_id: str = "08092",
    parties: list[str] | None = None,
) -> pd.DataFrame:
    """Partifordeling (godkjente stemmer, prosent) for én region/år.

    Returnerer kolonner: parti_kode, parti, parti_navn (SSB), prosent.
    """
    parties = parties or MAIN_PARTY_CODES
    df = client.fetch(table_id, {
        "Region": [region],
        "PolitParti": parties,
        "ContentsCode": ["GodkjenteProsent"],
        "Tid": [year],
    })
    out = (
        df.rename(columns={"PolitParti": "parti_kode",
                           "PolitParti_label": "parti_navn",
                           "value": "prosent"})
        .assign(parti=lambda d: d["parti_kode"].map(PARTY_CODES))
        [["parti_kode", "parti", "parti_navn", "prosent", "missing"]]
        .sort_values("prosent", ascending=False, ignore_index=True)
    )
    return out


# --------------------------------------------------------------------------- #
# Holdningslag — tillit (13834, nasjonalt × alder × kjønn)                     #
# --------------------------------------------------------------------------- #
def trust(
    client: SSBClient,
    *,
    trust_in: list[str] | None = None,
    age: list[str] | None = None,
    sex: list[str] | None = None,
    measure: str = "Gjsnitt",
    year: str = "2025",
) -> pd.DataFrame:
    """Tillitsmål brutt på demografi (alder × kjønn).

    ``measure``: f.eks. 'Gjsnitt' (skår 0-10), 'AndelHoyTillit', 'AntallSvar'.
    Standard: tillit til politiet (02), alle aldersgrupper, begge kjønn.
    """
    df = client.fetch("13834", {
        "Tillit": trust_in or ["02"],          # 02 = Politiet
        "Alder": age or ["999", "16-24", "25-44", "45-66", "067+"],
        "Kjonn": sex or ["0", "1", "2"],
        "ContentsCode": [measure],
        "Tid": [year],
    })
    return df.rename(columns={
        "Tillit_label": "tillit_til",
        "Alder_label": "aldersgruppe",
        "Kjonn_label": "kjonn",
        "value": measure.lower(),
    })


# --------------------------------------------------------------------------- #
# Strukturlag — husholdningsinntekt (06944, median per kommune)               #
# --------------------------------------------------------------------------- #
def household_income(
    client: SSBClient,
    region: str,
    *,
    household_type: str = "0000",          # 0000 = Alle husholdninger
    measure: str = "SamletInntekt",        # median samlet inntekt (kr)
    year: str | None = None,
) -> pd.DataFrame:
    """Median husholdningsinntekt for én region.

    Hvis ``year`` ikke er gitt, brukes siste tilgjengelige år i tabellen.
    """
    if year is None:
        tid_codes = client.variable_codes("06944")["Tid"]
        year = tid_codes[-1]
    df = client.fetch("06944", {
        "Region": [region],
        "HusholdType": [household_type],
        "ContentsCode": [measure],
        "Tid": [year],
    })
    return df.rename(columns={
        "Region_label": "region_navn",
        "HusholdType_label": "husholdningstype",
        "value": "median_kr",
    })
