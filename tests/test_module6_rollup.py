"""Modul 6 — offline-test av vektet opprulling med usikkerhetsbånd (del 1).

Kjør direkte:
    .venv/bin/python -m tests.test_module6_rollup
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from aggregate import bootstrap_distribution, rollup, fylke_of  # noqa: E402


def test_bootstrap_point_and_band() -> None:
    cats = ["a", "b", "c"]
    # 3 personas (enheter) med ulik fordeling, ulik vekt.
    per_unit = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    w = np.array([700.0, 200.0, 100.0])
    d = bootstrap_distribution(per_unit, w, cats, n_boot=300, seed=1)
    # Punktestimat = vektet snitt.
    expect = (per_unit * (w / w.sum())[:, None]).sum(0)
    assert np.allclose(d.point, expect, atol=1e-9)
    assert abs(d.point.sum() - 1.0) < 1e-9
    # Båndet omslutter punktet.
    assert np.all(d.lo <= d.point + 1e-9) and np.all(d.hi >= d.point - 1e-9)
    assert d.weight == 1000.0
    print(f"\nbootstrap point={np.round(d.point,3)} lo={np.round(d.lo,3)} hi={np.round(d.hi,3)}")


def test_rollup_hierarchy() -> None:
    cats = ["x", "y"]

    def dist(p):
        arr = np.array([[p, 1 - p]])
        return bootstrap_distribution(arr, np.array([1.0]), cats, n_boot=50, seed=0)

    # To kommuner i fylke 03, én i fylke 31.
    commune_dists = {"0301": dist(0.8), "0303": dist(0.6), "3101": dist(0.2)}
    weight = {"0301": 600.0, "0303": 400.0, "3101": 1000.0}
    out = rollup(commune_dists, weight)

    assert set(out["kommune"]) == {"0301", "0303", "3101"}
    # Fylke 03 = vektet snitt av 0301,0303: (0.8*600+0.6*400)/1000 = 0.72.
    assert abs(out["fylke"]["03"].point[0] - 0.72) < 1e-9
    assert fylke_of("0301") == "03"
    # Nasjonalt = vektet over alle: (0.8*600+0.6*400+0.2*1000)/2000 = 0.46.
    assert abs(out["nasjonalt"]["alle"].point[0] - 0.46) < 1e-9
    print(f"\nfylke03 x={out['fylke']['03'].point[0]:.2f}  "
          f"nasjonalt x={out['nasjonalt']['alle'].point[0]:.2f}")


ALL_TESTS = [test_bootstrap_point_and_band, test_rollup_hierarchy]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
