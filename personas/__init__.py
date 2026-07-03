"""SimNorge Modul 4 — persona-klynging.

Kondenserer den berikede befolkningen til vektede personas (kommune som hard
nøkkel) som bærer en holdningsfordeling, ikke et punkt — slik at både aggregat
og spredning bevares.
"""

from .cluster import (
    Persona, PersonaSet, build_personas, GRANULARITY, DEMO_AXES,
)

__all__ = ["Persona", "PersonaSet", "build_personas", "GRANULARITY", "DEMO_AXES"]
