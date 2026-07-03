"""SimNorge Modul 3 — holdningsberikelse.

Fester en demografisk holdningsprior (ikke en måling) på hver syntetisk person,
trukket fra SSBs nasjonale holdningsfordelinger med eksplisitt spredning.
"""

from .attitudes import (
    AttitudeItem, load_attitude_items, enrich_population,
    TRUST_ITEMS, MENING_ITEMS, FRIVILLIG_ITEMS,
    TRUST_LEVELS, SCORE_LEVELS, PART_LEVELS,
)

__all__ = [
    "AttitudeItem", "load_attitude_items", "enrich_population",
    "TRUST_ITEMS", "MENING_ITEMS", "FRIVILLIG_ITEMS",
    "TRUST_LEVELS", "SCORE_LEVELS", "PART_LEVELS",
]
