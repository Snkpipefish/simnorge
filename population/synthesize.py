"""Bygg syntetiske personer per kommune — Modul 2.

Knytter marginalbygging (marginals.py) og IPF (ipf.py) sammen til en konkret
syntetisk befolkning: én rad per person, med aksene {kjønn, aldersgruppe,
utdanning, økonomisk status} og kommune-skalar kontekst (median inntekt,
lavinntektsandel).

Reproduserbarhet: hele kjeden er deterministisk (IPF + største-rest-
integrering), så samme kommune/år gir EKSAKT samme befolkning hver kjøring.
``seed`` styrer kun en valgfri stokking av radrekkefølgen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import SSBClient
from .ipf import ipf, integerize
from .marginals import KommuneMarginals, build_marginals
from .schema import AGE_LABELS, ECON_LABELS, EDU_LABELS, SEX_LABELS, SHAPE
from .status import StatusMix, assign_status, build_status_mix

logger = logging.getLogger("simnorge.populasjon")


@dataclass
class PopulationResult:
    kommune: str
    year: str
    n: int
    persons: pd.DataFrame
    marginals: KommuneMarginals
    ipf_iterations: int
    ipf_max_dev: float
    econ_in_model: bool
    issues: list[str] = field(default_factory=list)
    skipped: bool = False


def build_population(
    ssb: SSBClient,
    kommune: str,
    *,
    year: str | None = None,
    seed: int = 0,
    shuffle: bool = False,
    target_n: int | None = None,
) -> PopulationResult:
    km = build_marginals(ssb, kommune, year=year)
    if not km.skipped and target_n and km.n_total > 0:
        # Skaler marginene proporsjonalt til en håndterbar størrelse (brukes
        # f.eks. til en nasjonal konsistens-sjekk uten å materialisere 4,3 mill.
        # personer). Bevarer fordelingene; endrer bare N.
        factor = target_n / km.n_total
        km.sex_age = km.sex_age * factor
        km.sex_edu = km.sex_edu * factor
        km.econ = km.econ * factor
        km.n_total = int(round(km.n_total * factor))
    if km.skipped:
        logger.warning("Kommune %s hoppet over: %s", kommune, "; ".join(km.issues))
        return PopulationResult(
            kommune=kommune, year=km.year, n=0, persons=_empty_persons(),
            marginals=km, ipf_iterations=0, ipf_max_dev=float("nan"),
            econ_in_model=False, issues=list(km.issues), skipped=True,
        )

    # Sett opp IPF-marginene. Økonomisk status tas bare med hvis den finnes;
    # ellers kollapses aksen til én "ukjent"-kategori (ærlig markert).
    if km.econ_available:
        shape = SHAPE
        econ_labels = ECON_LABELS
        margins = [((0, 1), km.sex_age), ((0, 2), km.sex_edu), ((3,), km.econ)]
    else:
        shape = (SHAPE[0], SHAPE[1], SHAPE[2], 1)
        econ_labels = ["ukjent"]
        margins = [((0, 1), km.sex_age), ((0, 2), km.sex_edu)]

    logger.info(
        "Kommune %s (%s): IPF-approksimasjon av fellesfordeling fra %d marginer "
        "(antar betinget uavhengighet gitt marginene — ekstrapolering, ikke målt).",
        kommune, km.year, len(margins),
    )
    fit = ipf(shape, margins)
    counts = integerize(fit.table, km.n_total)
    persons = _expand(counts, kommune, km, econ_labels)

    # Akse 5: arbeidsmarkedsstatus (13563-partisjon, tildelt per aldersbånd —
    # uavhengig av kjønn/utdanning gitt alder; logges i status.py).
    smix = build_status_mix(ssb, kommune, year=km.year)
    persons = assign_status(persons, smix)
    if smix.issues:
        km.issues.extend(smix.issues)

    if shuffle:
        persons = persons.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return PopulationResult(
        kommune=kommune, year=km.year, n=int(counts.sum()), persons=persons,
        marginals=km, ipf_iterations=fit.iterations, ipf_max_dev=fit.max_margin_dev,
        econ_in_model=km.econ_available, issues=list(km.issues),
    )


def _expand(counts: np.ndarray, kommune: str, km: KommuneMarginals,
            econ_labels: list[str]) -> pd.DataFrame:
    flat = counts.flatten()
    idx = np.repeat(np.arange(flat.size), flat)
    s, a, e, c = np.unravel_index(idx, counts.shape)
    df = pd.DataFrame({
        "kommune": kommune,
        "kjonn": np.array(SEX_LABELS)[s],
        "aldersgruppe": np.array(AGE_LABELS)[a],
        "utdanning": np.array(EDU_LABELS)[e],
        "okonomisk_status": np.array(econ_labels)[c],
    })
    # Kommune-skalar kontekst (likt for alle personer i kommunen).
    df["median_husholdningsinntekt"] = km.median_income
    df["lavinntekt_andel"] = km.lowincome_share
    df.insert(0, "person_id", [f"{kommune}-{i}" for i in range(len(df))])
    return df


def _empty_persons() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "person_id", "kommune", "kjonn", "aldersgruppe", "utdanning",
        "okonomisk_status", "arbeidsmarkedsstatus",
        "median_husholdningsinntekt", "lavinntekt_andel",
    ])
