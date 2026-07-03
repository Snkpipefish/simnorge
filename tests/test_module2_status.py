"""Modul 2+ — arbeidsmarkedsstatus-aksen (live SSB, ingen LLM).

Kjør direkte:
    .venv/bin/python -m tests.test_module2_status

1. Tildelt status per kommune reproduserer 13563-partisjonens totalandeler.
2. Alderslogikk: pensjonister bor i 45-66/67+, studenter i 16-24/25-44.
3. Tilt-retning: arbeidsledige/uføre får målbart LAVERE tillits- og
   livskvalitetsprior enn yrkesaktive (retningen er målt i 13839/13793,
   ikke antatt).
4. Frivillig innsats er nå differensiert på status (ikke flat nasjonal).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient  # noqa: E402
from population import STATUS_LABELS, build_population  # noqa: E402
from population.status import LEAF_MAP, SRC_BANDS, ARBSTATUS_TABLE  # noqa: E402
from enrich import load_attitude_items, enrich_population  # noqa: E402

KOMMUNE = "0301"   # Oslo — stor nok til at ingenting er prikket


def _ssb_total_shares(ssb) -> dict[str, float]:
    year = [t for t in ssb.variable_codes(ARBSTATUS_TABLE)["Tid"] if t <= "2024"][-1]
    df = ssb.fetch(ARBSTATUS_TABLE, {
        "Region": [KOMMUNE], "HovArbStyrkStatus": list(LEAF_MAP),
        "Alder": SRC_BANDS, "ContentsCode": ["Bosatte"], "Tid": [year],
    })
    tot: dict[str, float] = {}
    for _, r in df.iterrows():
        tot[LEAF_MAP[r["HovArbStyrkStatus"]]] = tot.get(
            LEAF_MAP[r["HovArbStyrkStatus"]], 0.0) + float(r["value"])
    s = sum(tot.values())
    return {k: v / s for k, v in tot.items()}


def test_status_shares_match_ssb() -> None:
    ssb = SSBClient()
    pop = build_population(ssb, KOMMUNE, year="2024")
    got = pop.persons["arbeidsmarkedsstatus"].value_counts(normalize=True)
    want = _ssb_total_shares(ssb)
    print("\nStatusandeler (syntetisk vs 13563):")
    for s in STATUS_LABELS:
        g, w = float(got.get(s, 0.0)), want.get(s, 0.0)
        print(f"  {s:12s} {g:.3f} vs {w:.3f}")
        # 13563 er 15+, vi er 16+; alders-reveiing gir slark — men totalene
        # skal ligge tett (±2.5 pp).
        assert abs(g - w) <= 0.025, (s, g, w)
    assert (pop.persons["arbeidsmarkedsstatus"] == "ukjent").sum() == 0


def test_status_age_logic() -> None:
    ssb = SSBClient()
    p = build_population(ssb, KOMMUNE, year="2024").persons
    pens = p[p["arbeidsmarkedsstatus"] == "pensjonist"]["aldersgruppe"]
    stud = p[p["arbeidsmarkedsstatus"] == "student"]["aldersgruppe"]
    assert (pens.isin(["45-66", "67+"])).mean() > 0.95, pens.value_counts()
    assert (stud.isin(["16-24", "25-44"])).mean() > 0.90, stud.value_counts()
    # 67+ er overveiende pensjonister.
    p67 = p[p["aldersgruppe"] == "67+"]["arbeidsmarkedsstatus"]
    assert (p67 == "pensjonist").mean() > 0.5, p67.value_counts(normalize=True)


def _mean_level(sub, col) -> float:
    m = {"lav": 0.0, "middels": 1.0, "hoy": 2.0}
    return float(sub[col].astype(str).map(m).mean())


def test_tilt_direction_measured_not_assumed() -> None:
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    p = enrich_population(build_population(ssb, KOMMUNE, year="2024").persons,
                          items, seed=42)
    # Sammenlign innen samme aldersbånd (25-44) så alderseffekten ikke forstyrrer.
    core = p[p["aldersgruppe"].isin(["25-44", "45-66"])]
    ya = core[core["arbeidsmarkedsstatus"] == "yrkesaktiv"]
    lu = core[core["arbeidsmarkedsstatus"].isin(["arbeidsledig", "ufor"])]
    assert len(lu) > 200, len(lu)
    for col in ("tillit_politisk_system", "livskvalitet_optimisme"):
        m_ya, m_lu = _mean_level(ya, col), _mean_level(lu, col)
        print(f"  {col}: yrkesaktiv={m_ya:.3f} vs ledig/ufør={m_lu:.3f}")
        assert m_lu < m_ya, (col, m_lu, m_ya)


def test_frivillig_now_differentiated() -> None:
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    friv = [i for i in items if i.column.startswith("frivillig_")]
    assert friv and all(i.status_factor is not None for i in friv)
    # Minst én aktivitet skal ha reell differensiering mellom statusene.
    assert any(np.ptp(i.status_factor[0]) > 0.1 for i in friv)


ALL_TESTS = [test_status_shares_match_ssb, test_status_age_logic,
             test_tilt_direction_measured_not_assumed, test_frivillig_now_differentiated]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
