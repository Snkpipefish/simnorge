"""Valgdata-paneler for backtest — Modul 7.

Laster valgresultat per kommune for ett valgår, normalisert til ett felles
referanseår via Klass-laget, og regner om til partiandeler på den normaliserte
kommunenøkkelen.

Korrekt håndtering av kodeendringer:
- Vi henter STEMMETALL (Godkjente1), ikke prosent, slik at sammenslåtte
  kommuner kan aggregeres trygt (summere stemmer), før vi regner ut andeler.
- Splittede kildekoder (ikke-reverserbare) droppes av normaliseringslaget og
  rapporteres her som ``dropped_splits`` — de teller som tapt dekning, ikke som
  gjettede tall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data import (
    SSBClient, KlassClient, PARTY_CODES, MAIN_PARTY_CODES,
    build_correspondence, normalize_dataframe,
)

# Koder i Region-dimensjonen som ikke er reelle kommuner.
NON_MUNICIPALITY_CODES = {"9999", "0"}


@dataclass
class ElectionPanel:
    """Normalisert valgresultat for ett valgår på referanse-kommunenøkkelen."""

    year: str
    reference_year: str
    table_id: str
    shares: pd.DataFrame              # kolonner: kommune, party, pct
    total_votes: pd.Series           # stemmer per kommune (vekt til senere bruk)
    dropped_splits: dict[str, list[str]] = field(default_factory=dict)

    @property
    def kommunes(self) -> set[str]:
        return set(self.shares["kommune"])

    def pivot(self) -> pd.DataFrame:
        """kommune × parti matrise av prosent (for scoring)."""
        return self.shares.pivot(index="kommune", columns="party", values="pct")


def municipality_code_set(klass: KlassClient, year: str) -> set[str]:
    """Gyldige kommunekoder for en årgang, uten ikke-kommune-koder."""
    return klass.code_set(year) - NON_MUNICIPALITY_CODES


def load_election_panel(
    ssb: SSBClient,
    klass: KlassClient,
    year: str,
    *,
    table_id: str = "08092",
    reference_year: str = "2024",
    parties: list[str] | None = None,
) -> ElectionPanel:
    """Last ett valgårs resultat, normalisert til ``reference_year``.

    Andeler regnes som parti-stemmer / alle godkjente stemmer i kommunen
    (samme definisjon som SSBs GodkjenteProsent), etter normalisering.
    """
    year = str(year)
    parties = parties or MAIN_PARTY_CODES

    codes = ssb.variable_codes(table_id)
    # Hent stemmetall for ALLE partier (for korrekt nevner per kommune).
    raw = ssb.fetch(table_id, {
        "Region": codes["Region"],
        "PolitParti": codes["PolitParti"],
        "ContentsCode": ["Godkjente1"],
        "Tid": [year],
    })
    keep = municipality_code_set(klass, year)
    raw = (raw[raw["Region"].isin(keep)]
           .rename(columns={"Region": "kommune", "PolitParti": "party_code", "value": "votes"})
           .copy())
    raw["votes"] = raw["votes"].fillna(0)

    # Normaliser stemmetall til referanseårgangen (sammenslåing summeres,
    # splitt droppes + logges).
    source = min(year, reference_year)
    corr = build_correspondence(klass, source, reference_year)
    norm, report = normalize_dataframe(
        raw, corr, code_col="kommune",
        group_cols=["party_code"], additive_cols=["votes"],
    )

    total = norm.groupby("kommune")["votes"].sum().rename("total_votes")
    main = norm[norm["party_code"].isin(parties)].merge(total, on="kommune")
    main["pct"] = main["votes"] / main["total_votes"] * 100.0
    main["party"] = main["party_code"].map(PARTY_CODES)

    shares = main[["kommune", "party", "pct"]].sort_values(
        ["kommune", "party"], ignore_index=True)

    return ElectionPanel(
        year=year,
        reference_year=reference_year,
        table_id=table_id,
        shares=shares,
        total_votes=total,
        dropped_splits=dict(report.split_codes_dropped),
    )
