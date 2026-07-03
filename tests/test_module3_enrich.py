"""Modul 3 — verifisering av holdningsberikelsen.

Kjør direkte:
    .venv/bin/python -m tests.test_module3_enrich

Verifiserer:
1. NASJONAL KONSISTENS (holdnings-konsistenssjekken planen nevner): den
   syntetiske befolkningen beriket og aggregert nasjonalt reproduserer SSBs
   nasjonale holdningsfordeling — kilden den ble seedet fra.
2. SPREDNING, ikke punktestimat: innen én demografisk celle reproduserer de
   trukne nivåene cellens SSB-fordeling (to personer i samme celle kan ulike).
3. GEOGRAFISK DIFFERENSIERING: ulike kommuner får ulik aggregert profil fordi
   demografien er ulik — selv om nasjonalt aggregat stemmer.
4. Reproduserbarhet: samme seed -> identisk berikelse.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient  # noqa: E402
from population import build_population  # noqa: E402
from population.schema import AGE_LABELS, SEX_LABELS  # noqa: E402
from enrich import load_attitude_items, enrich_population  # noqa: E402

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

NATIONAL_N = 150_000
TOL_PP = 2.0   # prosentpoeng; dekker aldersbånd-forsoning (13790) + trekkstøy


def _agg(enriched, item):
    return (enriched[item.column].value_counts(normalize=True)
            .reindex(item.levels).fillna(0).to_numpy())


def test_national_consistency() -> None:
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    res = build_population(ssb, "0", target_n=NATIONAL_N)
    enriched = enrich_population(res.persons, items, seed=42)

    print(f"\nNasjonal konsistens (N={res.n}), maks avvik per mål (pp):")
    worst = 0.0
    for it in items:
        dev = float(np.max(np.abs(_agg(enriched, it) - it.national))) * 100
        worst = max(worst, dev)
        flag = " <-- maks" if dev > 0.5 else ""
        if dev > 0.4:
            print(f"  {it.column:30s} {dev:5.2f}{flag}")
        assert dev <= TOL_PP, (it.column, dev)
    print(f"  ... alle {len(items)} mål innenfor {TOL_PP} pp (verste {worst:.2f}).")


def test_spread_within_cell() -> None:
    """Innen én celle reproduserer trekningene cellens SSB-fordeling, og
    profilen er en fordeling (ikke alle like)."""
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    res = build_population(ssb, "0301", target_n=200_000)   # Oslo, nok per celle
    enriched = enrich_population(res.persons, items, seed=7)

    item = next(i for i in items if i.column == "tillit_politiet")
    ai = AGE_LABELS.index("25-44"); si = SEX_LABELS.index("mann")
    cell = enriched[(enriched["aldersgruppe"] == "25-44") & (enriched["kjonn"] == "mann")]
    drawn = (cell["tillit_politiet"].value_counts(normalize=True)
             .reindex(item.levels).fillna(0).to_numpy())
    target = item.lut[ai, si]
    dev = float(np.max(np.abs(drawn - target))) * 100
    print(f"\nCelle 25-44/mann tillit_politiet: trukket={np.round(drawn,3)} "
          f"SSB={np.round(target,3)} dev={dev:.2f} pp")
    assert dev <= 2.0, dev
    # Eksplisitt spredning: cellen har mer enn ett nivå representert.
    assert (cell["tillit_politiet"].nunique() > 1)


def test_geographic_differentiation() -> None:
    """Ulik demografi -> ulik aggregert holdning per kommune (hele poenget)."""
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    item = next(i for i in items if i.column == "tillit_politiet")
    hoy_idx = item.levels.index("hoy")

    shares = {}
    for code, navn in [("0301", "Oslo"), ("1151", "Utsira")]:
        res = build_population(ssb, code)
        enriched = enrich_population(res.persons, items, seed=1)
        shares[navn] = _agg(enriched, item)[hoy_idx]
    print(f"\nHøy tillit til politiet: Oslo={shares['Oslo']:.3f} "
          f"Utsira={shares['Utsira']:.3f}")
    # Ulik alders-/kjønnsmiks gir ulik aggregert holdning. Vi hevder ingen
    # retning — bare at demografien faktisk differensierer kommunene.
    assert abs(shares["Oslo"] - shares["Utsira"]) > 0.01


def test_reproducible() -> None:
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    res = build_population(ssb, "1151")
    a = enrich_population(res.persons, items, seed=5)
    b = enrich_population(res.persons, items, seed=5)
    cols = [it.column for it in items]
    assert all(a[c].astype(str).equals(b[c].astype(str)) for c in cols)
    print("\nBerikelse reproduserbar med fast seed ✓")


ALL_TESTS = [
    test_national_consistency,
    test_spread_within_cell,
    test_geographic_differentiation,
    test_reproducible,
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
