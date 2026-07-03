"""Kjør valg-backtesten (motorvarianter vs baseliner) — Modul 6, reproduserbart.

LIVE: ekte LLM-kall via claude-CLI (cachet i .cache/engine_party/<variant>/).

    .venv/bin/python run_engine_backtest.py [build_year] [target_year]

Standard: bygg 2021 → mål 2025, prøvekommunene, tre varianter (stedløs /
geo-uten-navn / full geografi) mot persistence + national_mean. En MÅLING.
"""

from __future__ import annotations

import logging
import sys

from data import SSBClient, KlassClient
from enrich import load_attitude_items
from population import SAMPLE_KOMMUNER
from calibration import run_variants_backtest

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    build = sys.argv[1] if len(sys.argv) > 1 else "2021"
    target = sys.argv[2] if len(sys.argv) > 2 else "2025"
    pop_year = "2024" if target >= "2024" else target

    ssb, klass = SSBClient(), KlassClient()
    items = load_attitude_items(ssb)
    cmp = run_variants_backtest(
        ssb, klass, items, build_year=build, target_year=target,
        communes=list(SAMPLE_KOMMUNER), pop_year=pop_year,
        granularity="full", max_workers=5,
    )
    print("\n" + cmp.summary())


if __name__ == "__main__":
    main()
