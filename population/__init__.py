"""SimNorge Modul 2 — populasjonsbygger.

Bygger en syntetisk befolkning (16+) per kommune som matcher SSBs
marginalfordelinger, via iterativ proporsjonal tilpasning (IPF).
"""

from .schema import (
    SEX_LABELS, AGE_LABELS, EDU_LABELS, ECON_LABELS, STATUS_LABELS,
    STATUS_TO_OKSTATUS, SHAPE, age_band,
)
from .marginals import KommuneMarginals, build_marginals
from .ipf import IPFResult, ipf, integerize
from .status import StatusMix, assign_status, build_status_mix
from .synthesize import PopulationResult, build_population

# Prøvekommuner: Oslo + store + et par svært små (jf. PLAN: start med 5–10).
SAMPLE_KOMMUNER = {
    "0301": "Oslo",
    "4601": "Bergen",
    "5001": "Trondheim",
    "1103": "Stavanger",
    "3905": "Tønsberg",
    "1151": "Utsira",     # blant landets minste
    "1835": "Træna",      # svært liten
    "1856": "Røst",       # svært liten
}

__all__ = [
    "SEX_LABELS", "AGE_LABELS", "EDU_LABELS", "ECON_LABELS", "STATUS_LABELS",
    "STATUS_TO_OKSTATUS", "SHAPE", "age_band",
    "KommuneMarginals", "build_marginals",
    "IPFResult", "ipf", "integerize",
    "StatusMix", "assign_status", "build_status_mix",
    "PopulationResult", "build_population",
    "SAMPLE_KOMMUNER",
]
