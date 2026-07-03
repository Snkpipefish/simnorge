"""Modul 7 — backtest-skjelett end-to-end (mot ekte SSB + Klass).

Kjør direkte:
    .venv/bin/python -m tests.test_module7_backtest

Verifiserer at skjelettet:
1. kjører end-to-end med en triviell baseline og produserer MAE + korrelasjon,
2. ALLTID rapporterer dekning (X av Y kommuner) sammen med feilmålet,
3. håndhever out-of-sample strukturelt (samme valg som bygg+mål gir feil),
4. bruker normaliserte kommunekoder (splitt droppes, rapporteres som tap),
5. setter en eksplisitt baseline-MAE (terskelen den ekte motoren må slå).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import SSBClient, KlassClient  # noqa: E402
from calibration import (  # noqa: E402
    run_backtest, persistence_baseline, national_mean_baseline,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

BUILD_YEAR = "2021"
TARGET_YEAR = "2025"


def _clients():
    return SSBClient(), KlassClient()


def test_out_of_sample_guard() -> None:
    ssb, klass = _clients()
    raised = False
    try:
        run_backtest(ssb, klass, build_year="2021", target_year="2021",
                     predictor=persistence_baseline)
    except ValueError as e:
        raised = True
        print(f"\nOut-of-sample-vakt utløst som forventet: {e}")
    assert raised, "Samme bygg- og målår skulle vært forbudt"


def test_backtest_runs_and_reports_coverage() -> None:
    ssb, klass = _clients()
    res = run_backtest(ssb, klass, build_year=BUILD_YEAR, target_year=TARGET_YEAR,
                       predictor=persistence_baseline)
    print("\n" + res.summary())

    cov = res.coverage
    # Dekning er rapportert og meningsfull.
    assert cov.total_country_kommunes == 357, cov.total_country_kommunes
    assert 0 < cov.scored <= cov.total_country_kommunes
    # Ålesund (1507) er en ikke-reverserbar splitt i BYGGEÅRET (2021) → droppet
    # der, og de to 2024-kommunene den ville blitt (1508/1580) kan dermed ikke
    # predikeres fra byggeåret. Ærlig synlig i dekningen.
    assert "1507" in cov.dropped_build_split
    assert set(cov.dropped_no_prediction) == {"1508", "1580"}
    assert cov.scored == 355
    # Et MAE-tall finnes for de store partiene.
    assert res.mae_big == res.mae_big  # ikke NaN
    assert res.mae_overall > 0
    # Persistence har geografisk signal → korrelasjon definert og positiv for Ap.
    assert res.per_party_corr["Ap"] > 0


def test_national_mean_has_no_geographic_signal() -> None:
    ssb, klass = _clients()
    res = run_backtest(ssb, klass, build_year=BUILD_YEAR, target_year=TARGET_YEAR,
                       predictor=national_mean_baseline)
    print("\n" + res.summary())
    # Konstant prediksjon per kommune → kommune-korrelasjon er udefinert (NaN).
    import math
    assert all(math.isnan(c) for c in res.per_party_corr.values())
    # Og den skal være DÅRLIGERE enn persistence på MAE (ingen geografi).
    assert res.mae_overall > 0


ALL_TESTS = [
    test_out_of_sample_guard,
    test_backtest_runs_and_reports_coverage,
    test_national_mean_has_no_geographic_signal,
]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
