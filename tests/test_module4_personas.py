"""Modul 4 — verifisering av persona-klyngingen.

Kjør direkte:
    .venv/bin/python -m tests.test_module4_personas

Dobbel verifisering (brukerens to bekymringer):
(a) AGGREGATET BEVART: vektede personas aggregert nasjonalt reproduserer
    fortsatt SSBs holdningskonsistens innenfor Modul 3-toleransen — klyngingen
    flytter ikke aggregatet.
(b) SPREDNINGEN OVERLEVER: variansen i en holdningsdimensjon kollapser ikke mot
    null. Vi viser at persona-fordelingene bevarer populasjonsvariansen, mens en
    naiv "kollaps-til-gjennomsnitt" ville mistet en stor del av den.

Pluss: kommune er hard nøkkel (ingen persona krysser kommuner; antall skalerer
med kommuner) og reproduserbarhet.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient  # noqa: E402
from population import build_population  # noqa: E402
from enrich import load_attitude_items, enrich_population  # noqa: E402
from personas import build_personas  # noqa: E402

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

NATIONAL_N = 150_000
TOL_PP = 2.0


@lru_cache(maxsize=1)
def _items():
    return tuple(load_attitude_items(SSBClient()))


def _national_enriched():
    ssb = SSBClient()
    res = build_population(ssb, "0", target_n=NATIONAL_N)
    return enrich_population(res.persons, list(_items()), seed=42)


def _dist_var(probs):
    codes = np.arange(len(probs))
    mean = float((probs * codes).sum())
    return float((probs * (codes - mean) ** 2).sum())


def test_aggregate_preserved_under_clustering() -> None:
    items = list(_items())
    enriched = _national_enriched()
    # Mest aggressiv granularitet (alder×kjønn = 8 personas) — holder det her,
    # holder det for alle finere nivåer.
    pset = build_personas(enriched, items, granularity="alder_kjonn")
    print(f"\n{len(enriched)} personer -> {len(pset.personas)} personas. "
          f"Maks aggregatavvik vs SSB (pp):")
    worst = 0.0
    for it in items:
        dev = float(np.max(np.abs(pset.aggregate(it.column) - it.national))) * 100
        worst = max(worst, dev)
        assert dev <= TOL_PP, (it.column, dev)
    print(f"  alle {len(items)} mål innenfor {TOL_PP} pp (verste {worst:.2f}) — "
          f"aggregatet flyttet seg ikke.")


def test_spread_survives() -> None:
    items = list(_items())
    enriched = _national_enriched()
    pset = build_personas(enriched, items, granularity="alder_kjonn")

    print("\nSpredning før/etter klynging (varians i nivåkoder):")
    print(f"  {'mål':28s} {'pop':>7s} {'persona':>8s} {'naivt punkt':>11s} {'naiv/pop':>8s}")
    for col in ["tillit_politiet", "livskvalitet_optimisme", "frivillig_dugnad"]:
        pop_codes = enriched[col].cat.codes.to_numpy()
        pop_var = float(pop_codes.var())

        persona_var = _dist_var(pset.aggregate(col))   # fra fordelingene

        # Naiv kollaps: hver persona som ETT punkt på sitt gjennomsnittsnivå.
        means, weights = [], []
        for p in pset.personas:
            d = p.attitude_dist[col]
            means.append(float((d * np.arange(len(d))).sum()))
            weights.append(p.weight)
        means, weights = np.array(means), np.array(weights, float)
        weights /= weights.sum()
        gmean = float((weights * means).sum())
        naive_var = float((weights * (means - gmean) ** 2).sum())

        print(f"  {col:28s} {pop_var:7.4f} {persona_var:8.4f} {naive_var:11.4f} "
              f"{naive_var/pop_var if pop_var else 0:8.2f}")

        # (b1) Persona-fordelingene bevarer populasjonsvariansen.
        assert abs(persona_var - pop_var) <= 0.01 * max(pop_var, 1e-6) + 1e-6, col
        # (b2) Testen har tenner: en naiv punkt-kollaps ville mistet spredning.
        assert naive_var < persona_var, (col, "naiv kollaps mistet ikke varians?")


def test_kommune_is_hard_key() -> None:
    items = list(_items())
    ssb = SSBClient()

    def enriched_for(codes):
        import pandas as pd
        frames = [enrich_population(build_population(ssb, c).persons, items, seed=1)
                  for c in codes]
        return pd.concat(frames, ignore_index=True)

    one = build_personas(enriched_for(["0301"]), items)
    two = build_personas(enriched_for(["0301", "4601"]), items)

    # Ingen persona krysser kommuner.
    assert all(isinstance(p.kommune, str) for p in two.personas)
    assert two.kommuner() == {"0301", "4601"}
    # Vekt per kommune = faktisk befolkning (partisjon, ingen lekkasje).
    w_oslo = sum(p.weight for p in two.personas if p.kommune == "0301")
    assert w_oslo == sum(p.weight for p in one.personas)
    # Antall personas skalerer med kommuner (Oslos personas uendret når Bergen
    # legges til).
    assert len([p for p in two.personas if p.kommune == "0301"]) == len(one.personas)
    print(f"\nKommune hard nøkkel: 1 kommune -> {len(one.personas)} personas, "
          f"2 kommuner -> {len(two.personas)} (Oslo-delen uendret) ✓")


def test_reproducible() -> None:
    items = list(_items())
    enriched = _national_enriched()
    a = build_personas(enriched, items, granularity="full")
    b = build_personas(enriched, items, granularity="full")
    assert len(a.personas) == len(b.personas)
    assert all(np.allclose(pa.attitude_dist["tillit_politiet"],
                           pb.attitude_dist["tillit_politiet"]) and pa.weight == pb.weight
               for pa, pb in zip(a.personas, b.personas))
    print(f"\nReproduserbar: {len(a.personas)} personas identiske på to kjøringer ✓")


ALL_TESTS = [
    test_aggregate_preserved_under_clustering,
    test_spread_survives,
    test_kommune_is_hard_key,
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
