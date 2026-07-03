"""Modul 5 — LIVE konsistens- og konformitetstest (ekte LLM-kall via claude-CLI).

Kjør direkte:
    .venv/bin/python -m tests.test_module5_consistency

Krever `claude`-CLI tilgjengelig. Første kjøring gjør ~48 ekte kall (cachet på
disk i .cache/engine, så reruns er momentane).

(4) BLOKKERENDE konsistenssjekk: still motoren tillitsspørsmålet (matcher SSB
    13834 tillit_politiet), aggreger nasjonalt og bekreft at det reproduserer
    SSBs faktiske tillitsfordeling. Hvis ikke, ignorerer modellen prioren.
(5) Konformitets-sjekk: et mer DELT spørsmål (tillit til politisk system) — vis
    at halene overlever, at aggregatet ikke kollapser til et forsiktig midtpunkt.
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
from engine import (  # noqa: E402
    TRUST_POLICE, TRUST_POLITICAL_SYSTEM, expand_by_item, poll,
    weighted_mean, weighted_std, level_means, weighted_position_distribution,
)

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

NATIONAL_N = 120_000


@lru_cache(maxsize=1)
def _setup():
    ssb = SSBClient()
    items = load_attitude_items(ssb)
    enr = enrich_population(build_population(ssb, "0", target_n=NATIONAL_N).persons,
                            items, seed=42)
    pset = build_personas(enr, items, granularity="alder_kjonn")
    return ssb, items, pset


def _ssb_mean(ssb, trust_code):
    g = ssb.fetch("13834", {"Tillit": [trust_code], "Alder": ["999"], "Kjonn": ["0"],
                            "ContentsCode": ["Gjsnitt"],
                            "Tid": [ssb.variable_codes("13834")["Tid"][-1]]})
    return float(g.iloc[0]["value"])


def test_trust_consistency_blocking() -> None:
    ssb, items, pset = _setup()
    inst = expand_by_item(pset, "tillit_politiet")
    out = poll(inst, TRUST_POLICE, geo_context={"0": None}, max_workers=6)
    assert out.n_failed == 0, f"{out.n_failed} svar kunne ikke parses"

    lm = level_means(out.results)
    wm = weighted_mean(out.results)
    ssb_mean = _ssb_mean(ssb, "02")
    dist = weighted_position_distribution(out.results, [(0, 5), (6, 7), (8, 10)])
    item = next(i for i in items if i.column == "tillit_politiet")

    print(f"\n[KONSISTENS] nivå-snitt={ {k: round(v,2) for k,v in lm.items()} }")
    print(f"  LLM aggregat={wm:.2f} vs SSB Gjsnitt={ssb_mean:.2f}")
    print(f"  fordeling {dist} vs SSB {dict(zip(['0-5','6-7','8-10'], np.round(item.national,3)))}")

    # (a) Modellen resonnerer FRA prioren: nivåene er separert (overstyrer ikke).
    assert lm["lav"] < lm["middels"] < lm["hoy"], lm
    assert lm["lav"] <= 4.0 and lm["hoy"] >= 7.5, lm
    assert lm["hoy"] - lm["lav"] >= 4.0, lm
    # (b) Aggregatet reproduserer SSBs nasjonale tillit (mål + fordeling).
    assert abs(wm - ssb_mean) <= 1.0, (wm, ssb_mean)
    ssb_dist = item.national
    got = np.array([dist["0-5"], dist["6-7"], dist["8-10"]])
    assert np.max(np.abs(got - ssb_dist)) <= 0.05, (got, ssb_dist)


def test_conformity_preserves_spread() -> None:
    ssb, items, pset = _setup()
    inst = expand_by_item(pset, "tillit_politisk_system")
    out = poll(inst, TRUST_POLITICAL_SYSTEM, geo_context={"0": None}, max_workers=6)
    assert out.n_failed == 0

    lm = level_means(out.results)
    wstd = weighted_std(out.results)
    tails = weighted_position_distribution(out.results, [(0, 3), (4, 6), (7, 10)])
    print(f"\n[KONFORMITET] nivå-snitt={ {k: round(v,2) for k,v in lm.items()} } "
          f"std={wstd:.2f}")
    print(f"  haler {tails}")

    # Halene overlever: spredning bevart, og begge ender faktisk befolket —
    # IKKE kollapset mot et forsiktig midtpunkt.
    assert wstd >= 1.5, wstd
    assert tails["0-3"] >= 0.08, tails           # lav ende reell
    assert tails["7-10"] >= 0.30, tails          # høy ende reell
    assert lm["lav"] < lm["middels"] < lm["hoy"], lm
    assert lm["hoy"] - lm["lav"] >= 4.0, lm


ALL_TESTS = [test_trust_consistency_blocking, test_conformity_preserves_spread]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
