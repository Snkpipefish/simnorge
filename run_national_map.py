"""Batch: nasjonal poll av ett spørsmål + kommune-choropleth — Modul 8.

    .venv/bin/python run_national_map.py ["spørsmål"]

Standard spørsmål: tillit til det politiske systemet (politikk-domene =
geografisk validert akse). ~900 unike celler første gang (~20-25 min via
claude-CLI, 6 tråder); alt diskcachet og delt med spørretaben, så re-kjøring
og nye kart over samme celler er billige. Kartet legges i app/static/kart/
og dukker opp i spørretabens kart-seksjon.
"""

from __future__ import annotations

import logging
import sys

from app.batch import national_poll, render_map

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_QUESTION = "Hvor mye stoler du på det politiske systemet i Norge?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    res = national_poll(question)
    html, meta = render_map(res)
    print(f"\nKart: {html}\nMetadata: {meta}")
    print(f"Celler: {res.n_cells}, nye kall: {res.n_calls}, "
          f"feilede: {res.n_failed_cells}, kommuner: {len(res.values)}")
    if res.n_failed_cells:
        print("NB: feilede celler = dekningstap — kjør på nytt for å fylle hullene.")


if __name__ == "__main__":
    main()
