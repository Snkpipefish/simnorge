"""Modul 7 — deltakelsesvekting fra 13360 (live SSB + offline vektlogikk).

Kjør direkte:
    .venv/bin/python -m tests.test_module7_turnout

Verifiserer at:
1. LUT-en dekker hele (kjønn × utdanning)-rommet og har den kjente gradienten
   (grunnskole < videregående < UH) for begge kjønn,
2. manglende valgår gir None (uniform fallback), aldri et gjettet tall,
3. turnout_weight er 1.0 uten LUT og slår opp riktig med,
4. vekting i det økologiske cellerommet flytter masse fra grunnskole- til
   UH-celler (retningen vi vet fra fasit).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data import SSBClient  # noqa: E402
from population.schema import EDU_LABELS, SEX_LABELS  # noqa: E402
from calibration.turnout import load_turnout, turnout_weight  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def test_lut_complete_and_monotone() -> None:
    ssb = SSBClient()
    lut = load_turnout(ssb, "2021")
    assert lut is not None
    # Full dekning av schema-rommet.
    for s in SEX_LABELS:
        for e in EDU_LABELS:
            assert (s, e) in lut, (s, e)
            assert 0.3 < lut[(s, e)] < 1.0, lut[(s, e)]
    # Utdanningsgradienten (den kjente, dokumenterte skjevheten).
    for s in SEX_LABELS:
        assert lut[(s, "grunnskole")] < lut[(s, "videregaaende")] < lut[(s, "uh_lang")], s
    # Fagskole arver vgs-nivået, uh_kort/uh_lang deler UH-nivået.
    assert lut[("mann", "fagskole")] == lut[("mann", "videregaaende")]
    assert lut[("mann", "uh_kort")] == lut[("mann", "uh_lang")]
    print("\nDeltakelse 2021 (mann): grunnskole %.1f%%, vgs %.1f%%, UH %.1f%%" % (
        lut[("mann", "grunnskole")] * 100, lut[("mann", "videregaaende")] * 100,
        lut[("mann", "uh_lang")] * 100))


def test_missing_year_returns_none() -> None:
    ssb = SSBClient()
    assert load_turnout(ssb, "2017") is None   # 13360 har bare 2021/2025


def test_weight_fallback_and_lookup() -> None:
    assert turnout_weight(None, "mann", "grunnskole") == 1.0
    lut = {("mann", "grunnskole"): 0.589}
    assert turnout_weight(lut, "mann", "grunnskole") == 0.589
    assert turnout_weight(lut, "kvinne", "uh_lang") == 1.0   # hull -> uniform


def test_cell_weights_shift_direction() -> None:
    """Vekting skal flytte masse fra lav- til høy-deltakelsesceller."""
    ssb = SSBClient()
    from calibration.ecological import CELL_INDEX, cell_weights

    lut = load_turnout(ssb, "2021")
    w0 = cell_weights(ssb, "0301", year="2024")
    w1 = cell_weights(ssb, "0301", year="2024", turnout=lut)
    assert w0 is not None and w1 is not None
    assert np.isclose(w0.sum(), 1.0) and np.isclose(w1.sum(), 1.0)

    def mass(w, edu):
        return sum(w[i] for cell, i in CELL_INDEX.items() if cell[2] == edu)

    assert mass(w1, "grunnskole") < mass(w0, "grunnskole")
    assert mass(w1, "uh_lang") > mass(w0, "uh_lang")
    print("Oslo grunnskole-masse: %.3f -> %.3f, uh_lang: %.3f -> %.3f" % (
        mass(w0, "grunnskole"), mass(w1, "grunnskole"),
        mass(w0, "uh_lang"), mass(w1, "uh_lang")))


if __name__ == "__main__":
    test_lut_complete_and_monotone()
    test_missing_year_returns_none()
    test_weight_fallback_and_lookup()
    test_cell_weights_shift_direction()
    print("\nAlle deltakelsestester grønne.")
