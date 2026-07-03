"""Modul 7 — deltakelsesvekting fra 13360+13085 (live SSB + offline vektlogikk).

Kjør direkte:
    .venv/bin/python -m tests.test_module7_turnout

Verifiserer at:
1. LUT-en dekker hele (kjønn × aldersbånd × utdanning)-rommet og har begge
   kjente gradienter: utdanning (grunnskole < vgs < UH) og alder (16-24 er
   laveste bånd — trukket ned av både lav ung-deltakelse og 16-17 uten
   stemmerett),
2. manglende valgår gir None (uniform fallback), aldri et gjettet tall,
3. turnout_weight er 1.0 uten LUT og slår opp riktig med,
4. vekting i det økologiske cellerommet flytter masse fra grunnskole- til
   UH-celler og fra 16-24- til 45-66-bånd (retningene vi vet fra fasit).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient  # noqa: E402
from population.schema import AGE_LABELS, EDU_LABELS, SEX_LABELS  # noqa: E402
from calibration.turnout import load_turnout, turnout_weight  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def test_lut_complete_and_monotone() -> None:
    ssb = SSBClient()
    lut = load_turnout(ssb, "2021")
    assert lut is not None
    # Full dekning av schema-rommet.
    for s in SEX_LABELS:
        for a in AGE_LABELS:
            for e in EDU_LABELS:
                assert (s, a, e) in lut, (s, a, e)
                assert 0.01 <= lut[(s, a, e)] <= 0.99, lut[(s, a, e)]
    for s in SEX_LABELS:
        # Utdanningsgradienten, innen samme aldersbånd.
        for a in AGE_LABELS:
            assert lut[(s, a, "grunnskole")] < lut[(s, a, "videregaaende")] \
                < lut[(s, a, "uh_lang")], (s, a)
        # Aldersgradienten, innen samme utdanning: 16-24 lavest (inkl. 16-17
        # uten stemmerett), 45-66 høyere enn 16-24 og 25-44.
        for e in EDU_LABELS:
            assert lut[(s, "16-24", e)] < lut[(s, "25-44", e)], (s, e)
            assert lut[(s, "25-44", e)] < lut[(s, "45-66", e)], (s, e)
    # Fagskole arver vgs-nivået, uh_kort/uh_lang deler UH-nivået.
    assert lut[("mann", "45-66", "fagskole")] == lut[("mann", "45-66", "videregaaende")]
    assert lut[("mann", "45-66", "uh_kort")] == lut[("mann", "45-66", "uh_lang")]
    print("\nDeltakelse 2021 (mann, videregående): " + ", ".join(
        f"{a} {lut[('mann', a, 'videregaaende')]*100:.1f}%" for a in AGE_LABELS))


def test_missing_year_returns_none() -> None:
    ssb = SSBClient()
    assert load_turnout(ssb, "2017") is None   # 13360 har bare 2021/2025


def test_weight_fallback_and_lookup() -> None:
    assert turnout_weight(None, "mann", "16-24", "grunnskole") == 1.0
    lut = {("mann", "16-24", "grunnskole"): 0.35}
    assert turnout_weight(lut, "mann", "16-24", "grunnskole") == 0.35
    assert turnout_weight(lut, "kvinne", "67+", "uh_lang") == 1.0   # hull -> uniform


def test_cell_weights_shift_direction() -> None:
    """Vekting skal flytte masse fra lav- til høy-deltakelsesceller."""
    ssb = SSBClient()
    from calibration.ecological import CELL_INDEX, cell_weights

    lut = load_turnout(ssb, "2021")
    w0 = cell_weights(ssb, "0301", year="2024")
    w1 = cell_weights(ssb, "0301", year="2024", turnout=lut)
    assert w0 is not None and w1 is not None
    assert np.isclose(w0.sum(), 1.0) and np.isclose(w1.sum(), 1.0)

    def mass(w, axis, label):
        return sum(w[i] for cell, i in CELL_INDEX.items() if cell[axis] == label)

    assert mass(w1, 2, "grunnskole") < mass(w0, 2, "grunnskole")
    assert mass(w1, 2, "uh_lang") > mass(w0, 2, "uh_lang")
    assert mass(w1, 1, "16-24") < mass(w0, 1, "16-24")
    assert mass(w1, 1, "45-66") > mass(w0, 1, "45-66")
    print("Oslo 16-24-masse: %.3f -> %.3f, 45-66: %.3f -> %.3f" % (
        mass(w0, 1, "16-24"), mass(w1, 1, "16-24"),
        mass(w0, 1, "45-66"), mass(w1, 1, "45-66")))


if __name__ == "__main__":
    test_lut_complete_and_monotone()
    test_missing_year_returns_none()
    test_weight_fallback_and_lookup()
    test_cell_weights_shift_direction()
    print("\nAlle deltakelsestester grønne.")
