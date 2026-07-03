"""Modul 8 — spørretab-plumbing (offline: mock-backend, isolert cache).

Kjør direkte:
    .venv/bin/python -m tests.test_module8_app

Tester rørene, IKKE modelladferd (jf. cache-disiplinen: mock-svar skrives
til en midlertidig cache og rører aldri .cache/):
1. qspec: gyldig tolkning valideres og caches; ugyldig akse -> ærlig feil.
2. service.ask nasjonalt: aggregat + fordeling + ærlighetsmetadata komplett.
3. service.ask kommune: geo-linje uten kommunenavn i prompten.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.qspec import QSpecError, interpret_question  # noqa: E402
from app.service import SimNorgeService  # noqa: E402

QSPEC_JSON = {
    "axis": "tillit_politisk_system", "axis_reason": "handler om tillit til politikere",
    "domain": "politikk",
    "scale_low_label": "stoler ikke i det hele tatt",
    "scale_high_label": "stoler fullstendig",
    "level_phrasing": {"lav": "LAV tillit til politikere.",
                       "middels": "MIDDELS tillit til politikere.",
                       "hoy": "HØY tillit til politikere."},
}


class ScriptedBackend:
    """Svarer qspec-JSON på tolknings-prompter og posisjons-JSON på polling.
    Registrerer prompter så testen kan inspisere dem."""

    name = "scripted"

    def __init__(self, qspec_json: dict):
        self.qspec_json = qspec_json
        self.poll_prompts: list[str] = []

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str:
        if "målbar spesifikasjon" in prompt:
            return json.dumps(self.qspec_json, ensure_ascii=False)
        self.poll_prompts.append(prompt)
        m = re.search(r"holdning: (LAV|MIDDELS|HØY)", prompt)
        pos = {"LAV": 2, "MIDDELS": 6, "HØY": 9}.get(m.group(1) if m else "", 5)
        return json.dumps({"posisjon": pos, "begrunnelse": "mock", "sikkerhet": 0.8})


def test_qspec_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iq = interpret_question("Hvor mye stoler du på politikerne?",
                                backend=ScriptedBackend(QSPEC_JSON), cache_dir=tmp)
        assert iq.spec.focus_item == "tillit_politisk_system"
        assert iq.domain == "politikk"
        # Cachet: nytt kall uten backend-svar (backend som feiler) skal treffe cache.
        class Boom:
            name = "boom"
            def complete(self, *a, **kw):
                raise RuntimeError("skal ikke kalles — cachet")
        iq2 = interpret_question("Hvor mye stoler du på politikerne?",
                                 backend=Boom(), cache_dir=tmp)
        assert iq2.spec.level_phrasing == iq.spec.level_phrasing

        bad = dict(QSPEC_JSON, axis="tillit_svigermor")
        try:
            interpret_question("Tull?", backend=ScriptedBackend(bad), cache_dir=tmp)
            raise AssertionError("ugyldig akse skulle gitt QSpecError")
        except QSpecError:
            pass


def _service(tmp: str, backend) -> SimNorgeService:
    return SimNorgeService(backend=backend, cache_root=tmp)


def test_ask_national() -> None:
    backend = ScriptedBackend(QSPEC_JSON)
    with tempfile.TemporaryDirectory() as tmp:
        svc = _service(tmp, backend)
        res = svc.ask("Hvor mye stoler du på politikerne?", "0")
    d = res.as_dict()
    assert d["geografi"]["navn"] == "Hele Norge"
    assert 0 <= d["resultat"]["snitt"] <= 10 and d["resultat"]["spredning"] > 0
    assert abs(sum(v["andel"] for v in d["resultat"]["fordeling"].values()) - 1) < 1e-9
    assert d["aerlighet"]["geo_validitet"] == "validert"
    assert "EKSTRAPOLERT" in d["aerlighet"]["ekstrapolert"]
    assert set(d["resultat"]["per_nivaa"]) == {"lav", "middels", "hoy"}
    assert d["dekning"]["feilede"] == 0
    # Nasjonal poll: ingen bosted-linje i noen prompt.
    assert not any("Bosted:" in p for p in backend.poll_prompts)


def test_ask_kommune_geo_line_without_name() -> None:
    backend = ScriptedBackend(QSPEC_JSON)
    with tempfile.TemporaryDirectory() as tmp:
        svc = _service(tmp, backend)
        res = svc.ask("Hvor mye stoler du på politikerne?", "1151")   # Utsira
    assert res.geography_label == "Utsira"
    assert backend.poll_prompts and all("Bosted:" in p for p in backend.poll_prompts)
    # Kommunenavn skal ALDRI stå i prompten — kun sted-type + økonominivå.
    assert not any("Utsira" in p for p in backend.poll_prompts)


ALL_TESTS = [test_qspec_validation, test_ask_national, test_ask_kommune_geo_line_without_name]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
