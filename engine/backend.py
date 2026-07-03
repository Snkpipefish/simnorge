"""LLM-backender for spørremotoren — Modul 5.

Standard backend kaller `claude`-CLI-en ikke-interaktivt (`claude -p --model …`),
som bruker eksisterende auth — ingen API-nøkkel nødvendig i dette miljøet. En
valgfri API-backend (anthropic-SDK) kan brukes når en nøkkel finnes.

`MockBackend` er KUN for plumbing-tester (caching/vekting/parsing/aggregering).
Den er deterministisk og «lydig» mot prioren, og kan derfor ALDRI erstatte den
ekte konsistenssjekken mot SSB — den tester rør, ikke modelladferd.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import Protocol


class BackendError(RuntimeError):
    pass


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str: ...


class ClaudeCLIBackend:
    """Kaller `claude -p` som subprosess. stdin fra /dev/null for å unngå
    3-sekunders stdin-venting per kall."""

    name = "claude-cli"

    def __init__(self, timeout: int = 120, retries: int = 6, backoff: float = 10.0) -> None:
        # retries/backoff dimensjonert for VEDVARENDE last: rate-vinduer hos
        # CLI-en varer i minutter, så kort backoff (~30 s totalt) ga 2000+
        # unødvendige dekningstap i batch-kjøringen 2026-07-02. Nå ventes
        # 10+20+…+50 s ≈ 2,5 min før en celle regnes som tapt.
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str:
        last = ""
        for attempt in range(self.retries):
            try:
                proc = subprocess.run(
                    ["claude", "-p", "--model", model, prompt],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=self.timeout,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
                last = f"rc={proc.returncode} stderr={proc.stderr[:200]!r}"
            except subprocess.TimeoutExpired:
                last = f"timeout {self.timeout}s"
            # Transient (rate/usage/avbrudd): vent økende og prøv igjen.
            time.sleep(self.backoff * (attempt + 1))
        raise BackendError(f"claude CLI feilet etter {self.retries} forsøk: {last}")


class APIBackend:
    """Anthropic-SDK-backend (krever ANTHROPIC_API_KEY og `anthropic`)."""

    name = "anthropic-api"

    def __init__(self) -> None:
        import anthropic  # lazy: ikke nødvendig når CLI brukes
        self.client = anthropic.Anthropic()

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str:
        msg = self.client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


class MockBackend:
    """KUN plumbing. Leser dispoisjonsnivået ut av prompten og svarer lydig.
    Kan ikke teste om en ekte modell respekterer prioren."""

    name = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str:
        self.calls += 1
        level = "middels"
        m = re.search(r"din etablerte .*?:\s*(LAV|MIDDELS|HØY|DELTOK|IKKE)", prompt, re.I)
        if m:
            level = m.group(1).lower()
        pos = {"lav": 2, "middels": 6, "høy": 9, "deltok": 9, "ikke": 1}.get(level, 5)
        return f'{{"posisjon": {pos}, "begrunnelse": "mock {level}", "sikkerhet": 0.8}}'
