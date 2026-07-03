"""Modul 7 — kalibreringskorreksjon (offline, syntetiske data).

Kjør direkte:
    .venv/bin/python -m tests.test_module7_correction

Verifiserer at:
1. korreksjonen gjenoppretter en kjent demping-mot-midten (b > 1 læres),
2. identitetsdata gir ~identitetskorreksjon (ingen skade),
3. apply klipper negative andeler og renormaliserer til 100,
4. monotoni-vakten holder b >= 0 selv med antikorrelerte data,
5. kommune-CV reduserer MAE på dempede data og er ærlig (fit aldri på
   evaluerte kommuner).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from calibration.correction import (  # noqa: E402
    AffineCorrection, cv_corrected_mae, fit_correction,
)

PARTIES = ["Ap", "Høyre", "FrP", "Sp"]
NAT = {"Ap": 30.0, "Høyre": 30.0, "FrP": 25.0, "Sp": 15.0}


def _synthetic(damp: float, n: int = 40, noise: float = 0.5, seed: int = 1):
    """Fasit varierer rundt nasjonalt snitt; prediksjonen er DEMPET mot
    midten med faktor ``damp`` (< 1) + støy. Kolonneformat som i backtesten."""
    rng = np.random.default_rng(seed)
    rows_a, rows_p = [], []
    for i in range(n):
        k = f"K{i:03d}"
        dev = rng.normal(0, 6, len(PARTIES))
        dev -= dev.mean()
        for j, p in enumerate(PARTIES):
            y = NAT[p] + dev[j]
            x = NAT[p] + damp * dev[j] + rng.normal(0, noise)
            rows_a.append({"kommune": k, "party": p, "pct": y})
            rows_p.append({"kommune": k, "party": p, "pred_pct": x})
    return pd.DataFrame(rows_p), pd.DataFrame(rows_a)


def test_recovers_damping() -> None:
    pred, actual = _synthetic(damp=0.5)
    corr = fit_correction(pred, actual, PARTIES)
    for p in PARTIES:
        assert 1.6 < corr.b[p] < 2.4, (p, corr.b[p])   # ~1/0.5
    print("\n" + corr.summary())


def test_identity_is_harmless() -> None:
    pred, actual = _synthetic(damp=1.0, noise=0.01)
    corr = fit_correction(pred, actual, PARTIES)
    for p in PARTIES:
        assert abs(corr.b[p] - 1.0) < 0.05, (p, corr.b[p])
        assert abs(corr.a[p]) < 1.0, (p, corr.a[p])


def test_apply_clips_and_renormalizes() -> None:
    corr = AffineCorrection(parties=tuple(PARTIES),
                            a={p: -20.0 for p in PARTIES},
                            b={p: 1.0 for p in PARTIES}, n_fit=1)
    pred = pd.DataFrame([{"kommune": "K", "party": p, "pred_pct": v}
                         for p, v in [("Ap", 50), ("Høyre", 30), ("FrP", 15), ("Sp", 5)]])
    out = corr.apply(pred)
    assert (out["pred_pct"] >= 0).all()
    assert np.isclose(out["pred_pct"].sum(), 100.0)
    assert out.loc[out["party"] == "Sp", "pred_pct"].iloc[0] == 0.0   # 5-20 klippet


def test_monotonicity_guard() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(10, 40, 30)
    pred = pd.DataFrame({"kommune": [f"K{i}" for i in range(30)],
                         "party": "Ap", "pred_pct": x})
    actual = pd.DataFrame({"kommune": [f"K{i}" for i in range(30)],
                           "party": "Ap", "pct": 60 - x})   # perfekt antikorrelert
    corr = fit_correction(pred, actual, ["Ap"])
    assert corr.b["Ap"] == 0.0                               # aldri negativ


def test_cv_reduces_mae_on_damped_data() -> None:
    pred, actual = _synthetic(damp=0.5, n=60, seed=7)
    res = cv_corrected_mae(pred, actual, PARTIES, folds=4)
    print(f"CV på dempede data: raw MAE {res['raw']:.2f} -> "
          f"korrigert {res['corrected']:.2f} (n={res['n']})")
    assert res["corrected"] < res["raw"] * 0.75


if __name__ == "__main__":
    test_recovers_damping()
    test_identity_is_harmless()
    test_apply_clips_and_renormalizes()
    test_monotonicity_guard()
    test_cv_reduces_mae_on_damped_data()
    print("\nAlle korreksjonstester grønne.")
