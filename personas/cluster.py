"""Persona-klynging — Modul 4.

Kondenserer den berikede befolkningen til vektede representative personas for å
kutte LLM-kostnad, UTEN å miste det Modul 2/3 bygde inn. To harde begrensninger
ligger i kjernen (se memo `extrapolation-layers` og brukerens brief):

1. KOMMUNE ER EN HARD NØKKEL. En persona representerer alltid folk innen ÉN
   kommune, aldri på tvers — spørretaben må kunne dele per kommune. Det totale
   persona-antallet skaleres derfor med antall kommuner i scope (flere kommuner
   = flere personas = høyere kostnad, men geografisk oppdelbart). Det er en
   bevisst avveining, ikke en svakhet.

2. SPREDNINGEN MÅ OVERLEVE. Klynging frister til å kollapse hver gruppe til sitt
   gjennomsnitt — da kaster vi variasjonen vi nettopp betalte for (særlig Modul
   3s poeng om at to like demografiske personer kan ha ulik holdning). Derfor
   bærer hver persona en LITEN FORDELING (vektet histogram over nivåene), ikke
   ett punkt. Aggregatet blir da ikke kunstig skarpt, og variansen bevares.

Granularitet (kostnad/oppløsning-knapp): I denne byggingen avhenger holdnings-
priorene kun av alder×kjønn (13834, 13790) eller ingenting (13831). Å kollapse
utdanning/økonomi er derfor tilnærmet tapsfritt FOR HOLDNINGER — det grovkorner
bare den demografiske beskrivelsen. Det gir en prinsipiell knapp:
``full`` (alle fire akser) → ``utdanning`` (kollaps økonomi) → ``alder_kjonn``.
Kollapsede akser bæres videre som en sammensetnings-fordeling på personaen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("simnorge.personas")

DEMO_AXES = ["kjonn", "aldersgruppe", "utdanning", "okonomisk_status"]

# Granularitetsforhåndsinnstillinger: hvilke akser klyngenøkkelen bruker.
GRANULARITY = {
    "full": ["kjonn", "aldersgruppe", "utdanning", "okonomisk_status"],
    "utdanning": ["kjonn", "aldersgruppe", "utdanning"],   # kollaps økonomi
    "alder_kjonn": ["kjonn", "aldersgruppe"],              # kollaps utd + økonomi
}


@dataclass
class Persona:
    persona_id: str
    kommune: str
    weight: int                                  # antall innbyggere representert
    demographics: dict[str, str]                 # eksakt verdi for klyngeaksene
    composition: dict[str, dict[str, float]]     # fordeling for kollapsede akser
    median_husholdningsinntekt: float | None
    lavinntekt_andel: float | None
    attitude_dist: dict[str, np.ndarray]         # mål -> sannsynlighet over nivåene
    n_members: int


@dataclass
class PersonaSet:
    personas: list[Persona]
    items: list[str]                             # holdningsmål (kolonnenavn)
    levels: dict[str, list[str]]                 # nivåer per mål
    granularity: list[str]

    @property
    def total_weight(self) -> int:
        return int(sum(p.weight for p in self.personas))

    def kommuner(self) -> set[str]:
        return {p.kommune for p in self.personas}

    def aggregate(self, item: str) -> np.ndarray:
        """Vektet aggregert fordeling over nivåene for ett mål (rekonstruert fra
        personaenes fordelinger — ikke fra punkter)."""
        acc = np.zeros(len(self.levels[item]))
        w = 0.0
        for p in self.personas:
            acc += p.weight * p.attitude_dist[item]
            w += p.weight
        return acc / w if w else acc

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.personas:
            row = {"persona_id": p.persona_id, "kommune": p.kommune,
                   "weight": p.weight, "n_members": p.n_members,
                   **p.demographics,
                   "median_husholdningsinntekt": p.median_husholdningsinntekt,
                   "lavinntekt_andel": p.lavinntekt_andel}
            rows.append(row)
        return pd.DataFrame(rows)


def build_personas(
    enriched: pd.DataFrame,
    items,
    *,
    granularity: str | list[str] = "full",
    seed: int = 0,
) -> PersonaSet:
    """Bygg vektede personas fra en beriket befolkning.

    ``items`` er listen av AttitudeItem fra Modul 3 (gir kolonnenavn + nivåer).
    ``granularity`` velger klyngeaksene (alltid innen kommune). Deterministisk:
    grupperingen er reproduserbar; ``seed`` er med for API-konsistens.
    """
    axes = GRANULARITY[granularity] if isinstance(granularity, str) else list(granularity)
    item_cols = [it.column for it in items]
    levels = {it.column: list(it.levels) for it in items}
    collapsed = [a for a in DEMO_AXES if a not in axes]

    key = ["kommune"] + axes
    personas: list[Persona] = []
    for keyvals, g in enriched.groupby(key, observed=True, sort=True):
        keyvals = keyvals if isinstance(keyvals, tuple) else (keyvals,)
        kommune = keyvals[0]
        demo = dict(zip(axes, keyvals[1:]))

        dist = {}
        for it in items:
            vc = g[it.column].value_counts(normalize=True)
            dist[it.column] = np.array([float(vc.get(lvl, 0.0)) for lvl in it.levels])

        comp = {}
        for ax in collapsed:
            vc = g[ax].value_counts(normalize=True)
            comp[ax] = {str(k): float(v) for k, v in vc.items()}
            # Representativ (modal) verdi for kollapset akse, til beskrivelse.
            demo[ax] = str(vc.idxmax())

        personas.append(Persona(
            persona_id=f"{kommune}#{len(personas)}",
            kommune=kommune,
            weight=int(len(g)),
            demographics={a: str(demo[a]) for a in DEMO_AXES},
            composition=comp,
            median_husholdningsinntekt=_scalar(g, "median_husholdningsinntekt"),
            lavinntekt_andel=_scalar(g, "lavinntekt_andel"),
            attitude_dist=dist,
            n_members=int(len(g)),
        ))

    logger.info(
        "Bygget %d personas over %d kommune(r) (granularitet=%s). Kommune er hard "
        "nøkkel; antall skalerer med kommuner. Hver persona bærer en fordeling, "
        "ikke et punkt — spredningen bevares.",
        len(personas), enriched["kommune"].nunique(), axes,
    )
    return PersonaSet(personas=personas, items=item_cols, levels=levels, granularity=axes)


def _scalar(g: pd.DataFrame, col: str):
    if col not in g.columns:
        return None
    v = g[col].iloc[0]
    return None if pd.isna(v) else float(v)
