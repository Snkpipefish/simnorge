"""Byggeren — én rad per bosatt innbygger i Norge.

Kilder: 07459 (befolkning per kommune × kjønn × 1-årsalder — EKTE tellinger,
ikke utvalg), 06655/06944 (inntekt), 08092/13085/13698 (valg), antakelser
(personlighet). Resultatet skrives som parquet med kategoriske kolonner —
hele landet blir en fil på noen titalls MB som grupperes på sekunder.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient, KlassClient
from calibration.election_data import municipality_code_set

from .personlighet import tildel_personlighet, TYPER
from .inntekt import InntektsModell
from .valg import ValgModell, UTFALL, PARTIER

logger = logging.getLogger("simnorge.kopi.bygg")

KJONN_LABELS = ["mann", "kvinne"]


def _hent_befolkning(ssb: SSBClient, klass: KlassClient, year: str,
                     kommuner: list[str] | None) -> pd.DataFrame:
    """07459: tellinger per (kommune, kjønn, 1-årsalder), kun ekte kommuner."""
    koder = ssb.variable_codes("07459")
    regioner = sorted(municipality_code_set(klass, year) & set(koder["Region"]))
    if kommuner:
        regioner = sorted(set(regioner) & set(kommuner))
    aldre = [a for a in koder["Alder"] if a.isdigit()]
    df = ssb.fetch("07459", {
        "Region": regioner, "Kjonn": ["1", "2"], "Alder": aldre,
        "ContentsCode": ["Personer1"], "Tid": [year],
    })
    df = df[~df["missing"] & (df["value"] > 0)].copy()
    df["alder"] = df["Alder"].astype(int)
    df["antall"] = df["value"].astype(int)
    logger.info("07459 (%s): %d kommuner, %d celler, %d personer.",
                year, len(regioner), len(df), int(df["antall"].sum()))
    return df[["Region", "Region_label", "Kjonn", "alder", "antall"]]


def bygg_kopi(
    *,
    befolkningsaar: str = "2026",
    kommuner: list[str] | None = None,
    seed: int = 0,
    ssb: SSBClient | None = None,
) -> pd.DataFrame:
    """Bygg den syntetiske kopien. Deterministisk gitt seed."""
    ssb = ssb or SSBClient()
    klass = KlassClient()

    celler = _hent_befolkning(ssb, klass, befolkningsaar, kommuner)
    rep = celler["antall"].to_numpy()
    kommune = np.repeat(celler["Region"].to_numpy(), rep)
    kjonn_i = np.repeat((celler["Kjonn"] == "2").to_numpy().astype(np.int8), rep)
    alder = np.repeat(celler["alder"].to_numpy().astype(np.int16), rep)
    n = len(alder)
    logger.info("Ekspanderte til %d individer.", n)

    persontype = tildel_personlighet(alder, kjonn_i, seed=seed)
    inntekt = InntektsModell(ssb).trekk(alder, kjonn_i, kommune, seed=seed + 1)
    valg = ValgModell(ssb, klass, kommune_aargang=befolkningsaar)
    parti = valg.tildel(alder, kjonn_i, kommune, inntekt, seed=seed + 2)

    navn = dict(zip(celler["Region"], celler["Region_label"]))
    kom_cat = pd.Categorical(kommune)
    df = pd.DataFrame({
        "kommune": kom_cat,
        "kommune_navn": pd.Categorical.from_codes(
            kom_cat.codes, categories=[navn[k] for k in kom_cat.categories]),
        "kjonn": pd.Categorical.from_codes(kjonn_i, categories=KJONN_LABELS),
        "alder": alder,
        "brutto_inntekt": inntekt,
        "personlighetstype": pd.Categorical.from_codes(persontype, categories=TYPER),
        "parti": pd.Categorical.from_codes(parti, categories=UTFALL),
    })
    _verifiser(df, valg)
    return df


def _verifiser(df: pd.DataFrame, valg: ValgModell) -> None:
    """Sjekk at kopien reproduserer fasit: valgresultat per kommune."""
    velgere = df[df["parti"].isin(PARTIER)]
    sim = (velgere.groupby(["kommune", "parti"], observed=True).size()
           .unstack(fill_value=0))
    sim = sim.div(sim.sum(axis=1), axis=0)[PARTIER]
    felles = sim.index.intersection(valg.shares.index)
    avvik = (sim.loc[felles] - valg.shares.loc[felles]).abs()
    logger.info("Verifikasjon valg: %d kommuner, maks avvik fra faktisk "
                "resultat %.2f pp, snitt %.3f pp (kun trekkstøy).",
                len(felles), avvik.max().max() * 100, avvik.mean().mean() * 100)
