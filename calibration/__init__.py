"""SimNorge Modul 7 — kalibrering & backtest (skjelett).

Evalueringsrammeverket som lar oss måle fra dag én: pluggbar prediktor,
out-of-sample-håndhevelse, MAE + kommune-korrelasjon, og dekningsrapportering.
"""

from .election_data import ElectionPanel, load_election_panel, municipality_code_set
from .metrics import BacktestResult, CoverageReport, BIG_PARTIES, mae, pearson
from .baselines import (
    Predictor, persistence_baseline, national_mean_baseline,
)
from .backtest import run_backtest
from .engine_predictor import EnginePredictor, ENGINE_PARTIES, party_prompt, VARIANTS
from .engine_backtest import run_variants_backtest, VariantsComparison
from .turnout import load_turnout, turnout_weight
from .correction import AffineCorrection, fit_correction, cv_corrected_mae

__all__ = [
    "ElectionPanel", "load_election_panel", "municipality_code_set",
    "BacktestResult", "CoverageReport", "BIG_PARTIES", "mae", "pearson",
    "Predictor", "persistence_baseline", "national_mean_baseline",
    "run_backtest",
    "EnginePredictor", "ENGINE_PARTIES", "party_prompt", "VARIANTS",
    "run_variants_backtest", "VariantsComparison",
    "load_turnout", "turnout_weight",
    "AffineCorrection", "fit_correction", "cv_corrected_mae",
]
