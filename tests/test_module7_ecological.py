"""Modul 7 økologisk inferens — matematikk-tester (ingen nettverk).

Kjør direkte:
    .venv/bin/python -m tests.test_module7_ecological

1) Den håndkodede gradienten matcher numerisk differensiering.
2) Modellen gjenvinner kjente celle-stereotyper fra aggregater alene
   (det er hele påstanden i økologisk inferens — verifisert på syntetisk
   data der fasiten er kjent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.ecological import (  # noqa: E402
    EcologicalModel, _loss_grad, _softmax, fit_ecological,
)


def _rand_problem(seed=0, K=12, C=6, P=4, G=2):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(C), size=K)
    S = rng.dirichlet(np.ones(P), size=K)
    X = rng.normal(0, 1, (K, G))
    beta = rng.normal(0, 0.5, P)
    delta = rng.normal(0, 0.5, (C, P))
    gamma = rng.normal(0, 0.5, (G, P))
    return W, S, X, beta, delta, gamma


def test_gradient_matches_numeric():
    W, S, X, beta, delta, gamma = _rand_problem()
    l2c, l2g = 1e-3, 1e-3
    _, db, dd, dg = _loss_grad(beta, delta, gamma, W, S, X, l2c, l2g)

    eps = 1e-6

    def loss_at(b, d, g):
        return _loss_grad(b, d, g, W, S, X, l2c, l2g)[0]

    for analytic, param, name in [(db, beta, "beta"), (dd, delta, "delta"),
                                  (dg, gamma, "gamma")]:
        flat = param.reshape(-1)
        idxs = np.random.default_rng(1).choice(flat.size, size=min(8, flat.size),
                                               replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            up = loss_at(beta, delta, gamma)
            flat[i] = orig - eps
            dn = loss_at(beta, delta, gamma)
            flat[i] = orig
            num = (up - dn) / (2 * eps)
            ana = analytic.reshape(-1)[i]
            assert abs(ana - num) < 1e-6, f"{name}[{i}]: {ana} vs {num}"


def test_recovers_known_stereotypes():
    """Syntetisk fasit: kjente celle-fordelinger + geo-effekt -> aggregér ->
    fit skal komme nær fasiten out-of-sample."""
    rng = np.random.default_rng(42)
    K, C, P, G = 120, 8, 5, 2
    true_beta = rng.normal(0, 0.3, P)
    true_delta = rng.normal(0, 1.0, (C, P))
    true_gamma = rng.normal(0, 0.8, (G, P))

    W = rng.dirichlet(np.ones(C) * 2, size=K)
    X = rng.normal(0, 1, (K, G))
    logits = true_beta[None, None, :] + true_delta[None, :, :] + (X @ true_gamma)[:, None, :]
    S = np.einsum("kc,kcp->kp", W, _softmax(logits))

    tr, te = slice(0, 90), slice(90, K)
    model = fit_ecological(W[tr], S[tr], X[tr], parties=[f"p{i}" for i in range(P)],
                           l2_cell=1e-6, l2_geo=1e-6, iters=4000, lr=0.05)
    pred = model.predict_shares(W[te], X[te])
    mae_pp = np.abs(pred - S[te]).mean() * 100
    assert mae_pp < 0.5, f"out-of-sample MAE {mae_pp:.2f} pp — gjenvinner ikke fasiten"

    # Uten geo-ledd skal samme data gi KLART dårligere fit (den økologiske
    # feilslutningen gjort synlig).
    X0 = np.zeros((K, 0))
    demo = fit_ecological(W[tr], S[tr], X0[tr], parties=[f"p{i}" for i in range(P)],
                          l2_cell=1e-6, l2_geo=1e-6, iters=4000, lr=0.05)
    mae_demo = np.abs(demo.predict_shares(W[te], X0[te]) - S[te]).mean() * 100
    assert mae_demo > mae_pp * 2


def test_predict_shares_sums_to_one():
    W, S, X, beta, delta, gamma = _rand_problem(seed=3)
    model = EcologicalModel(parties=list("abcd"), beta=beta, delta=delta,
                            gamma=gamma, geo_cols=["g1", "g2"],
                            l2_cell=0, l2_geo=0)
    pred = model.predict_shares(W, X)
    assert np.allclose(pred.sum(axis=1), 1.0)


ALL_TESTS = [
    test_gradient_matches_numeric,
    test_recovers_known_stereotypes,
    test_predict_shares_sums_to_one,
]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
