"""SimNorge Modul 1 — datalag.

Eksporterer SSB-klienten, tabellregisteret og høynivå-uttrekkene.
"""

from .ssb_client import SSBClient, SSBQueryError, SuppressionReport
from .tables import TABLES, Layer, Table, by_layer, PARTY_CODES, MAIN_PARTY_CODES
from .loaders import election_shares, trust, household_income
from .klass_client import KlassClient, MUNICIPALITY_CLASSIFICATION
from .municipality import (
    Correspondence, NormalizationReport, build_correspondence,
    normalize_dataframe, DEFAULT_REFERENCE_YEAR,
)

__all__ = [
    "SSBClient", "SSBQueryError", "SuppressionReport",
    "TABLES", "Layer", "Table", "by_layer", "PARTY_CODES", "MAIN_PARTY_CODES",
    "election_shares", "trust", "household_income",
    "KlassClient", "MUNICIPALITY_CLASSIFICATION",
    "Correspondence", "NormalizationReport", "build_correspondence",
    "normalize_dataframe", "DEFAULT_REFERENCE_YEAR",
]
