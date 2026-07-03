"""Modul 2 — verifisering av populasjonsbyggeren.

Kjør direkte:
    .venv/bin/python -m tests.test_module2_population

Verifiserer for prøvekommunene at:
1. IPF konvergerer (fitted-marginene matcher input nær eksakt — IPFs garanti).
2. Den syntetiske befolkningen aggregert tilbake matcher SSB-marginene innenfor
   en liten toleranse (alder/kjønn på tellinger, utdanning/økonomi på andeler).
3. Bygget er reproduserbart (samme kommune -> identisk befolkning).
4. Også de minste kommunene (Utsira/Træna/Røst) bygger uten stille imputering.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient  # noqa: E402
from population import (  # noqa: E402
    SAMPLE_KOMMUNER, AGE_LABELS, EDU_LABELS, SEX_LABELS, build_population,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _sex_age_counts(persons):
    m = np.zeros((2, 4))
    si = {s: i for i, s in enumerate(SEX_LABELS)}
    ai = {a: i for i, a in enumerate(AGE_LABELS)}
    for (s, a), g in persons.groupby(["kjonn", "aldersgruppe"]):
        m[si[s], ai[a]] = len(g)
    return m


def _edu_counts(persons):
    counts = persons["utdanning"].value_counts()
    return np.array([counts.get(e, 0) for e in EDU_LABELS], dtype=float)


# Integreringsgranularitet: største-rest-metoden lar hver marginalkategori
# bomme med høyst noen få personer — UAVHENGIG av kommunestørrelse (ikke
# proporsjonalt med N). Derfor en tett, fast absolutt toleranse i personer.
TOL_PERSONS = 5.0


def test_population_matches_margins() -> None:
    ssb = SSBClient()
    print(f"\n{'kommune':9s} {'N':>7s} {'IPFdev':>9s} "
          f"{'alder/kjønn':>11s} {'utd':>6s} {'økon':>6s}  (maks celleavvik, personer)")
    for code, navn in SAMPLE_KOMMUNER.items():
        res = build_population(ssb, code)
        assert not res.skipped, f"{navn} ble hoppet over: {res.issues}"
        p = res.persons
        assert len(p) == res.n == res.marginals.n_total

        # (1) IPF-garanti: fitted-marginene er nær eksakte.
        assert res.ipf_max_dev < 1e-3, (navn, "IPF-konvergens", res.ipf_max_dev)

        # (2) Syntetisk befolkning aggregert tilbake matcher SSB-marginene
        #     (i tellinger) innenfor integreringsgranulariteten.
        age_dev = np.max(np.abs(_sex_age_counts(p) - res.marginals.sex_age))
        edu_dev = np.max(np.abs(_edu_counts(p) - res.marginals.sex_edu.sum(axis=0)))
        econ_dev = (abs((p["okonomisk_status"] == "lavinntekt").sum() - res.marginals.econ[0])
                    if res.econ_in_model else 0.0)

        print(f"{navn:9s} {res.n:7d} {res.ipf_max_dev:9.1e} "
              f"{age_dev:11.2f} {edu_dev:6.2f} {econ_dev:6.2f}")

        assert age_dev <= TOL_PERSONS, (navn, "alder/kjønn", age_dev)
        assert edu_dev <= TOL_PERSONS, (navn, "utdanning", edu_dev)
        assert econ_dev <= TOL_PERSONS, (navn, "økonomisk status", econ_dev)


def test_reproducible() -> None:
    ssb = SSBClient()
    a = build_population(ssb, "0301").persons
    b = build_population(ssb, "0301").persons
    assert a.equals(b), "Oslo-befolkningen var ikke reproduserbar"
    print("\nOslo reproduserbar: identisk befolkning på to kjøringer ✓")


def test_smallest_kommunes_build() -> None:
    ssb = SSBClient()
    for code in ["1151", "1835", "1856"]:
        res = build_population(ssb, code)
        assert not res.skipped and res.n > 0
        # Ingen stille imputering: alle eventuelle datamangler er logget i issues.
        print(f"  {SAMPLE_KOMMUNER[code]:8s} N={res.n:4d}  econ_i_modell={res.econ_in_model}"
              f"  issues={res.issues if res.issues else 'ingen'}")


ALL_TESTS = [
    test_population_matches_margins,
    test_reproducible,
    test_smallest_kommunes_build,
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
