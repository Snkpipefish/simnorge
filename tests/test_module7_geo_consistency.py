"""Modul 7 — geografisk holdningskonsistens mot SSB 13798/13799 (live SSB, ingen LLM).

Kjør direkte:
    .venv/bin/python -m tests.test_module7_geo_consistency

Sjekker at prediksjonen fra prior × demografisk miks:
1. er gyldige fordelinger per kommune/gruppe,
2. holder nasjonal forankring mot målt nasjonalt nivå (myk grense),
3. rapporterer ærlig hvor mye av målt geo-variasjon demografien forklarer
   (MAE vs flat baseline + korrelasjon) — rapporten er poenget, ikke pynt.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient, KlassClient  # noqa: E402
from enrich import load_attitude_items  # noqa: E402
from population import SAMPLE_KOMMUNER  # noqa: E402
from calibration.geo_consistency import (  # noqa: E402
    kommune_predictions, run_geo_consistency,
)

logging.basicConfig(level=logging.ERROR)


def _clients():
    return SSBClient(), KlassClient()


def test_predictions_are_distributions() -> None:
    ssb, _ = _clients()
    items = load_attitude_items(ssb)
    preds = kommune_predictions(ssb, items, list(SAMPLE_KOMMUNER), pop_year="2024")
    assert len(preds) == len(SAMPLE_KOMMUNER)
    for k, d in preds.items():
        assert d["n"] > 0
        for ax, vec in d["pred"].items():
            assert vec.shape == (3,) and abs(vec.sum() - 1.0) < 1e-9, (k, ax, vec)
            assert (vec >= 0).all()


def test_geo_consistency_report() -> None:
    ssb, klass = _clients()
    items = load_attitude_items(ssb)
    from calibration.election_data import municipality_code_set
    from calibration.geo_consistency import (
        _aggregate, compare, measured_distributions, FYLKE_TABLE,
    )
    from enrich.attitudes import MENING_ITEMS
    kommuner = sorted(municipality_code_set(klass, "2024"))
    res_f, res_s = run_geo_consistency(ssb, klass, items, kommuner, pop_year="2024")
    print("\nMED arbeidsmarkedsstatus-tilt:")
    print(res_f.summary())
    print(res_s.summary())

    # Referanse: ren demografi (uten status) — hva bidrar aksen med geografisk?
    preds0 = kommune_predictions(ssb, items, kommuner, pop_year="2024",
                                 use_status=False)
    meas_f, year_f = measured_distributions(ssb, FYLKE_TABLE, "Region", MENING_ITEMS)
    national = meas_f.pop("0", {})
    res_f0 = compare(_aggregate(preds0, {k: k[:2] for k in preds0}), meas_f,
                     national, "fylke UTEN status", year_f)
    print("\nUTEN status (referanse):")
    print(res_f0.summary())

    for res, min_groups in ((res_f, 10), (res_s, 5)):
        assert len(res.groups) >= min_groups, res.groups
        assert len(res.axes) == 6, res.axes
        # Nasjonal forankring: prediksjonens landsaggregat skal ligge nær målt
        # nasjonalt nivå (prior seedet fra samme undersøkelse; 16+ vs 18+ og
        # årgangsforskjeller gir slark — myk grense, ikke pynt).
        assert res.national_gap_pp <= 5.0, res.national_gap_pp
        # Prediksjonen skal ikke være VESENTLIG verre enn flat baseline —
        # demografisk miks skal ikke ødelegge det nasjonale nivået.
        assert res.mae_pred_pp <= res.mae_flat_pp + 1.0, (res.mae_pred_pp,
                                                          res.mae_flat_pp)
        assert not np.isnan(res.corr_lav) and not np.isnan(res.corr_hoy)


ALL_TESTS = [test_predictions_are_distributions, test_geo_consistency_report]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
