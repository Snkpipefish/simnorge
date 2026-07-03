"""Økologisk inferens: lær celle-stereotypene fra ALLE kommunene — Modul 7.

Idéen (fra brukeren): hver kommune er en ligning. Valgresultatet er en vektet
sum av hvordan de demografiske cellene stemmer, og cellevektene kjenner vi fra
IPF-populasjonen. Med ~357 kommuner × 9 partier × flere valg er systemet
overbestemt — stereotypene kan ESTIMERES fra norsk fasit i stedet for å
hentes fra LLM-ens magefølelse. Null LLM-kall → alle kommuner er gratis.

Modell (mikstur av multinomiske logit-er):
    andel_kp = Σ_c w_kc · softmax_p( β_p + δ_cp + Σ_g x_kg γ_gp )

- w_kc: kommunens IPF-cellevekter (fraksjonelle — ingen personekspansjon).
- β_p:  nasjonalt parti-intercept (uregularisert).
- δ_cp: celle-stereotypen (L2-regularisert mot 0 = mot landssnittet).
- γ_gp: geografiske ledd (sentralitet, inntektsnivå, lavinntektsandel).

DEN ØKOLOGISKE FEILSLUTNINGEN, eksplisitt: uten geo-leddene vil regresjonen
tilskrive stedseffekter til demografiske grupper (bønder «får skylden» for at
hele Sp-kommunen stemmer Sp). Backtesten i Modul 6 påviste at sted forklarer
mye utover demografi (geo_no_name >> demographic_only), så γ er ikke pynt —
den er det som holder δ ærlig. Vi fitter begge varianter for å SE effekten.

Protokoll: fit på ett valgår, valider out-of-sample på et annet (aldri samme).
λ velges med k-fold CV på byggeåret alene — målåret røres aldri før scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import SSBClient, KlassClient
from population.ipf import ipf
from population.marginals import build_marginals
from population.schema import AGE_LABELS, EDU_LABELS, SEX_LABELS, SHAPE
from .engine_predictor import ENGINE_PARTIES, SENT_MARKER, load_centrality

logger = logging.getLogger("simnorge.okologisk")

ECON3 = ["lavinntekt", "ovrig", "ukjent"]     # felles celle-rom også når 12944 mangler
CELLS: list[tuple[str, str, str, str]] = [
    (s, a, e, c)
    for s in SEX_LABELS for a in AGE_LABELS for e in EDU_LABELS for c in ECON3
]
CELL_INDEX = {cell: i for i, cell in enumerate(CELLS)}
N_CELLS = len(CELLS)                          # 2×4×6×3 = 144

# Sentralitet 01..06 -> tre grupper (samme grovhet som prompt-markørene).
SENT_GROUP = {"01": 0, "02": 0, "03": 1, "04": 1, "05": 2, "06": 2}
GEO_FEATURES = ["sentral", "mellomsentral", "distrikt", "inntektsratio", "lavinntektsandel"]


# --------------------------------------------------------------------------- #
# Cellevekter og geo-features per kommune                                      #
# --------------------------------------------------------------------------- #
def cell_weights(ssb: SSBClient, kommune: str, *, year: str,
                 turnout: dict[tuple[str, str], float] | None = None) -> np.ndarray | None:
    """IPF-fellesfordelingen som normaliserte vekter i det felles 144-cellerommet.

    Fraksjonell (ingen integerisering/personekspansjon) — vi trenger andeler,
    ikke individer. None hvis kommunen mangler kritiske marginer (logget der).

    ``turnout``: valgfri (kjønn, aldersbånd, utdanning) -> P(stemmer) fra
    13360+13085 (byggeåret — lekkasjedisiplin håndheves av kallende kode).
    Valgresultatet er en deltakelsesvektet sum av cellene, så vektene bør være
    det også: uten dette må δ-stereotypene selv absorbere at unge og lavt
    utdannede møter sjeldnere opp.
    """
    km = build_marginals(ssb, kommune, year=year)
    if km.skipped or km.n_total <= 0:
        return None
    if km.econ_available:
        shape = SHAPE
        margins = [((0, 1), km.sex_age), ((0, 2), km.sex_edu), ((3,), km.econ)]
        econ_labels = ["lavinntekt", "ovrig"]
    else:
        shape = (SHAPE[0], SHAPE[1], SHAPE[2], 1)
        margins = [((0, 1), km.sex_age), ((0, 2), km.sex_edu)]
        econ_labels = ["ukjent"]
    table = ipf(shape, margins).table
    w = np.zeros(N_CELLS)
    for si, s in enumerate(SEX_LABELS):
        for ai, a in enumerate(AGE_LABELS):
            for ei, e in enumerate(EDU_LABELS):
                for ci, c in enumerate(econ_labels):
                    t = turnout.get((s, a, e), 1.0) if turnout is not None else 1.0
                    w[CELL_INDEX[(s, a, e, c)]] = table[si, ai, ei, ci] * t
    return w / w.sum()


def geo_features(ssb: SSBClient, klass: KlassClient, kommuner: list[str],
                 *, year: str) -> pd.DataFrame:
    """Geo-feature-matrise per kommune: sentralitetsgruppe (one-hot),
    inntektsratio mot landet (sentrert) og lavinntektsandel (sentrert)."""
    sent = load_centrality(klass)
    nat = ssb.fetch("06944", {"Region": ["0"], "HusholdType": ["0000"],
                              "ContentsCode": ["SamletInntekt"],
                              "Tid": [ssb.variable_codes("06944")["Tid"][-1]]})
    national_median = float(nat.iloc[0]["value"])

    rows = []
    for k in kommuner:
        km = build_marginals(ssb, k, year=year)   # diskcachet — gratis re-oppslag
        g = SENT_GROUP.get(sent.get(k, ""), None)
        onehot = [0.0, 0.0, 0.0]
        if g is not None:
            onehot[g] = 1.0
        ratio = ((km.median_income / national_median) - 1.0
                 if km.median_income else 0.0)
        low = (km.lowincome_share - 0.10) if km.lowincome_share is not None else 0.0
        rows.append([*onehot, ratio, low])
    return pd.DataFrame(rows, index=kommuner, columns=GEO_FEATURES)


# --------------------------------------------------------------------------- #
# Modellen                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class EcologicalModel:
    parties: list[str]
    beta: np.ndarray                    # (P,)   nasjonalt intercept
    delta: np.ndarray                   # (C,P)  celle-stereotyper
    gamma: np.ndarray                   # (G,P)  geo-effekter
    geo_cols: list[str]
    l2_cell: float
    l2_geo: float
    fit_loss: float = float("nan")

    def cell_dist(self, cell: tuple[str, str, str, str],
                  x_geo: np.ndarray | None = None) -> dict[str, float]:
        """Den lærte stereotypen: partifordeling (%) for én demografisk celle."""
        logits = self.beta + self.delta[CELL_INDEX[cell]]
        if x_geo is not None:
            logits = logits + x_geo @ self.gamma
        p = _softmax(logits[None, :])[0]
        return {pt: float(v * 100) for pt, v in zip(self.parties, p)}

    def predict_shares(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        """(K,C) vekter + (K,G) geo -> (K,P) predikerte andeler (sum 1)."""
        logits = (self.beta[None, None, :] + self.delta[None, :, :]
                  + (X @ self.gamma)[:, None, :])
        pi = _softmax(logits)
        return np.einsum("kc,kcp->kp", W, pi)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _loss_grad(beta, delta, gamma, W, S, X, l2_cell, l2_geo):
    """Vektet MSE + L2. Returnerer (loss, dβ, dδ, dγ). Gradient håndkodet
    (verifisert numerisk i test_module7_ecological)."""
    K = W.shape[0]
    logits = beta[None, None, :] + delta[None, :, :] + (X @ gamma)[:, None, :]
    pi = _softmax(logits)                                   # (K,C,P)
    pred = np.einsum("kc,kcp->kp", W, pi)                   # (K,P)
    err = pred - S
    loss = float((err ** 2).sum() / K
                 + l2_cell * (delta ** 2).sum() + l2_geo * (gamma ** 2).sum())

    R = 2.0 * err / K                                        # dLoss/dPred (K,P)
    A = np.einsum("kp,kcp->kc", R, pi)                       # (K,C)
    G = W[:, :, None] * pi * (R[:, None, :] - A[:, :, None])  # (K,C,P)
    d_beta = G.sum(axis=(0, 1))
    d_delta = G.sum(axis=0) + 2.0 * l2_cell * delta
    d_gamma = X.T @ G.sum(axis=1) + 2.0 * l2_geo * gamma
    return loss, d_beta, d_delta, d_gamma


def fit_ecological(W: np.ndarray, S: np.ndarray, X: np.ndarray, *,
                   parties: list[str] | None = None,
                   geo_cols: list[str] | None = None,
                   l2_cell: float = 1e-4, l2_geo: float = 1e-4,
                   iters: int = 3000, lr: float = 0.05,
                   seed: int = 0) -> EcologicalModel:
    """Adam på mikstur-modellen. W:(K,C) vekter, S:(K,P) fasitandeler (sum 1),
    X:(K,G) geo-features (G kan være 0 → ren demografisk variant)."""
    parties = parties or ENGINE_PARTIES
    K, C = W.shape
    P = S.shape[1]
    Gn = X.shape[1]
    rng = np.random.default_rng(seed)

    # Start i landssnittet: β = logit av nasjonal fordeling, δ=γ=0.
    nat = S.mean(axis=0).clip(1e-4)
    beta = np.log(nat / nat.sum())
    delta = rng.normal(0, 1e-3, (C, P))
    gamma = np.zeros((Gn, P))

    params = [beta, delta, gamma]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss = float("nan")
    for t in range(1, iters + 1):
        loss, *grads = _loss_grad(beta, delta, gamma, W, S, X, l2_cell, l2_geo)
        for i, g in enumerate(grads):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            params[i] -= lr * mh / (np.sqrt(vh) + eps)
    logger.info("Økologisk fit: K=%d kommuner, C=%d celler, G=%d geo, loss=%.6f",
                K, C, Gn, loss)
    return EcologicalModel(parties=parties, beta=beta, delta=delta, gamma=gamma,
                           geo_cols=geo_cols or [], l2_cell=l2_cell, l2_geo=l2_geo,
                           fit_loss=loss)


def cv_lambda(W: np.ndarray, S: np.ndarray, X: np.ndarray, *,
              grid: list[float] | None = None, folds: int = 5,
              iters: int = 1500, seed: int = 0) -> float:
    """k-fold CV på BYGGEÅRET for å velge felles λ (l2_cell = l2_geo).
    Målåret røres aldri her. Returnerer λ med lavest kommune-MAE."""
    grid = grid or [1e-5, 1e-4, 1e-3, 1e-2]
    K = W.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(K)
    fold_of = np.zeros(K, dtype=int)
    for i, idx in enumerate(order):
        fold_of[idx] = i % folds

    best_lam, best_mae = grid[0], float("inf")
    for lam in grid:
        errs = []
        for f in range(folds):
            tr, te = fold_of != f, fold_of == f
            model = fit_ecological(W[tr], S[tr], X[tr], l2_cell=lam, l2_geo=lam,
                                   iters=iters, seed=seed)
            pred = model.predict_shares(W[te], X[te])
            errs.append(np.abs(pred - S[te]).mean() * 100)
        m = float(np.mean(errs))
        logger.info("CV λ=%g: MAE=%.3f pp", lam, m)
        if m < best_mae:
            best_lam, best_mae = lam, m
    logger.info("CV valgte λ=%g (MAE %.3f pp)", best_lam, best_mae)
    return best_lam


# --------------------------------------------------------------------------- #
# Prediktor med samme kall-konvensjon som EnginePredictor                      #
# --------------------------------------------------------------------------- #
@dataclass
class EcologicalPredictor:
    """Fittet modell + datalag -> prediksjons-DataFrame (kommune, party, pred_pct)."""

    model: EcologicalModel
    weights: dict[str, np.ndarray]      # kommune -> (C,) cellevekter
    geo: pd.DataFrame                   # kommune-indeksert feature-matrise
    name: str = "ecological"
    skipped: list[str] = field(default_factory=list)

    def predict(self, kommuner: list[str]) -> pd.DataFrame:
        rows = []
        for k in kommuner:
            w = self.weights.get(k)
            if w is None:
                self.skipped.append(k)
                continue
            x = (self.geo.loc[[k]].to_numpy() if k in self.geo.index
                 else np.zeros((1, len(self.model.geo_cols))))
            shares = self.model.predict_shares(w[None, :], x)[0]
            for p, v in zip(self.model.parties, shares):
                rows.append({"kommune": k, "party": p, "pred_pct": float(v * 100)})
        return pd.DataFrame(rows)
