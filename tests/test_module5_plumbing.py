"""Modul 5 — plumbing-tester (OFFLINE, MockBackend).

Kjør direkte:
    .venv/bin/python -m tests.test_module5_plumbing

Verifiserer RØRENE — instans-ekspansjon, brøkvekt, caching (ingen re-kall),
trygg JSON-parsing og vektet aggregering — UTEN ekte LLM-kall. MockBackend er
lydig mot prioren og kan derfor ALDRI erstatte den ekte konsistenssjekken mot
SSB (test_module5_consistency); den tester kun at maskineriet stemmer.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from personas.cluster import Persona, PersonaSet  # noqa: E402
from engine import (  # noqa: E402
    MockBackend, TRUST_POLICE, expand_by_item, poll, parse_response,
    weighted_mean, level_means,
)


def _toy_personaset():
    personas = [
        Persona("0301#0", "0301", 1000,
                {"kjonn": "mann", "aldersgruppe": "25-44", "utdanning": "uh_lang",
                 "okonomisk_status": "ovrig"}, {}, 800000, 0.1,
                {"tillit_politiet": np.array([0.10, 0.20, 0.70])}, 1000),
        Persona("0301#1", "0301", 500,
                {"kjonn": "kvinne", "aldersgruppe": "67+", "utdanning": "grunnskole",
                 "okonomisk_status": "ovrig"}, {}, 500000, 0.1,
                {"tillit_politiet": np.array([0.05, 0.15, 0.80])}, 500),
    ]
    return PersonaSet(personas, ["tillit_politiet"],
                      {"tillit_politiet": ["lav", "middels", "hoy"]},
                      ["kjonn", "aldersgruppe"])


def test_expand_weights_preserved() -> None:
    inst = expand_by_item(_toy_personaset(), "tillit_politiet")
    assert len(inst) == 6                              # 2 personas × 3 nivåer
    # Brøkvektene per persona summerer til personavekten.
    w0 = inst[inst.persona_id == "0301#0"]["weight"].sum()
    assert abs(w0 - 1000) < 1e-9
    assert abs(inst["weight"].sum() - 1500) < 1e-9


def test_parse_safe() -> None:
    assert parse_response('{"posisjon": 8, "sikkerhet": 0.9}', TRUST_POLICE)["posisjon"] == 8
    # Ekstra tekst rundt JSON.
    p = parse_response('Her er svaret:\n{"posisjon": 3, "begrunnelse":"x"}\nTakk', TRUST_POLICE)
    assert p and p["posisjon"] == 3
    # Klamping til skala.
    assert parse_response('{"posisjon": 99}', TRUST_POLICE)["posisjon"] == 10
    # Søppel -> None (logget, ikke krasj).
    assert parse_response("ikke json", TRUST_POLICE) is None


def test_caching_and_weighting() -> None:
    pset = _toy_personaset()
    inst = expand_by_item(pset, "tillit_politiet")
    backend = MockBackend()
    with tempfile.TemporaryDirectory() as tmp:
        geo = {"0301": "Bosted: et sentralt strøk (storbyområde). "
                       "Kommunens økonomi: rundt landssnittet i inntekt, middels andel lavinntekt."}
        out1 = poll(inst, TRUST_POLICE, geo_context=geo,
                    backend=backend, cache_dir=tmp, max_workers=4)
        # 6 unike prompter (2 demografier × 3 nivåer) -> 6 kall første gang.
        assert out1.n_calls == 6 and out1.n_failed == 0, (out1.n_calls, out1.n_failed)
        calls_after_first = backend.calls

        # Andre kjøring: alt cachet -> ingen nye backend-kall.
        out2 = poll(inst, TRUST_POLICE, geo_context=geo,
                    backend=backend, cache_dir=tmp, max_workers=4)
        assert out2.n_calls == 0
        assert backend.calls == calls_after_first      # ikke økt

    # Mock mapper lav->2, middels->6, høy->9: nivå-snitt skal reflektere det,
    # og aggregatet skal bære fordelingen (ikke kollapse).
    lm = level_means(out1.results)
    assert lm["lav"] < lm["middels"] < lm["hoy"]
    assert abs(lm["lav"] - 2) < 0.01 and abs(lm["hoy"] - 9) < 0.01
    wm = weighted_mean(out1.results)
    print(f"\nMock nivå-snitt: {lm}; vektet aggregat={wm:.2f}")
    assert 7.0 < wm < 9.0          # tung mot høy, men halene drar ned -> ikke 9


ALL_TESTS = [test_expand_weights_preserved, test_parse_safe, test_caching_and_weighting]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
