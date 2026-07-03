"""Økologisk backtest: lær stereotypene fra ALLE kommunene — Modul 7.

Fit på build_year (alle kommuner med data), valider out-of-sample på
target_year. Null LLM-kall — ren SSB-data + regresjon. λ velges med 5-fold
CV på byggeåret alene; målåret røres aldri før scoring.

    .venv/bin/python run_ecological_backtest.py [build_year] [target_year]

Standard: 2021 → 2025. Rapporterer to nivåer:
- alle scorbare kommuner (den egentlige testen — dette kan motoren ikke),
- de 8 prøvekommunene (for direkte sammenligning med motorvariantene).

Populasjonsvekter bruker pop_year (2024) for BEGGE år — demografi flytter seg
sakte, og valgpanelene er normalisert til 2024-koder. Ærlig forenkling, logget.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from data import SSBClient, KlassClient
from population import SAMPLE_KOMMUNER
from calibration.election_data import load_election_panel, municipality_code_set
from calibration.engine_backtest import _national_mean9, _renorm9, _score, _wide
from calibration.ecological import (
    CELLS, GEO_FEATURES, EcologicalPredictor, cell_weights, cv_lambda,
    fit_ecological, geo_features,
)
from calibration.engine_predictor import ENGINE_PARTIES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("simnorge.ssb").setLevel(logging.ERROR)
logging.getLogger("simnorge.populasjon").setLevel(logging.ERROR)
logging.getLogger("simnorge.kommune").setLevel(logging.ERROR)
log = logging.getLogger("okologisk.runner")

POP_YEAR = "2024"


def main() -> None:
    build = sys.argv[1] if len(sys.argv) > 1 else "2021"
    target = sys.argv[2] if len(sys.argv) > 2 else "2025"

    ssb, klass = SSBClient(), KlassClient()
    fit_panel = load_election_panel(ssb, klass, build)
    test_panel = load_election_panel(ssb, klass, target)
    all_codes = sorted(municipality_code_set(klass, "2024"))
    candidates = [k for k in all_codes if k in fit_panel.kommunes and k in test_panel.kommunes]

    # Deltakelsesvekting fra BYGGEÅRET (13360+13085). None -> uniform (logget).
    # MÅLT (2021→2025, n=355): eksakt nøytral for MAE (5.01/9.24 begge veier),
    # både med og uten aldersgradienten — fitten absorberer omvektingen.
    # Beholdes fordi δ-stereotypene da leses som VELGERE (ikke bosatte), som er
    # det ligningen faktisk beskriver.
    from calibration.turnout import load_turnout
    turnout = load_turnout(ssb, build)

    log.info("Bygger IPF-cellevekter for %d kommuner (pop %s) …", len(candidates), POP_YEAR)
    weights: dict[str, np.ndarray] = {}
    no_weights = []
    for i, k in enumerate(candidates):
        try:
            w = cell_weights(ssb, k, year=POP_YEAR, turnout=turnout)
        except Exception as e:  # noqa: BLE001 — én kommune skal ikke velte kjøringen
            log.warning("Kommune %s: cellevekter feilet (%s) — teller som tapt dekning.", k, e)
            w = None
        if w is None:
            no_weights.append(k)
        else:
            weights[k] = w
        if (i + 1) % 50 == 0:
            log.info("  … %d/%d", i + 1, len(candidates))
    scored = sorted(weights)
    log.info("Cellevekter klare: %d kommuner (%d uten data: %s)",
             len(scored), len(no_weights), no_weights[:8])

    geo = geo_features(ssb, klass, scored, year=POP_YEAR)

    # Fasit-matriser på 9-parti-basis, kommune-justert rekkefølge.
    S_fit = (_wide(_renorm9(fit_panel.shares, scored), "pct")
             .reindex(scored).to_numpy() / 100.0)
    S_test_wide = _wide(_renorm9(test_panel.shares, scored), "pct").reindex(scored)
    W = np.vstack([weights[k] for k in scored])
    X_geo = geo.loc[scored].to_numpy()
    X_none = np.zeros((len(scored), 0))

    # NaN-vakt: kommuner der fasit mangler helt droppes fra fit.
    ok = ~np.isnan(S_fit).any(axis=1)
    if (~ok).any():
        log.warning("%d kommuner uten komplett %s-fasit droppet fra fit.",
                    int((~ok).sum()), build)

    variants = {}
    for vname, X in [("eco_demographic", X_none), ("eco_geo", X_geo)]:
        lam = cv_lambda(W[ok], S_fit[ok], X[ok], iters=1200)
        model = fit_ecological(W[ok], S_fit[ok], X[ok],
                               geo_cols=GEO_FEATURES if X.shape[1] else [],
                               l2_cell=lam, l2_geo=lam, iters=4000)
        pred = EcologicalPredictor(model=model, weights=weights,
                                   geo=geo if X.shape[1] else geo[[]], name=vname)
        variants[vname] = pred

    # ------------------------------------------------------------------ #
    # Scoring — alle kommuner og prøvekommune-delsettet                    #
    # ------------------------------------------------------------------ #
    natvec = _national_mean9(fit_panel)

    def preds_for(kommuner: list[str]) -> dict[str, pd.DataFrame]:
        actual = S_test_wide.reindex(kommuner)
        out = {"persistence": _wide(_renorm9(fit_panel.shares, kommuner), "pct").reindex(kommuner),
               "national_mean": pd.DataFrame({p: natvec[p] for p in ENGINE_PARTIES},
                                             index=actual.index)}
        for vname, pred in variants.items():
            long = pred.predict(kommuner)
            out[vname] = (_wide(long.rename(columns={"pred_pct": "pct"}), "pct")
                          .reindex(kommuner))
        return out

    for label, kommuner in [
        (f"ALLE scorbare kommuner (n={len(scored)})", scored),
        (f"prøvekommunene (n={len([k for k in SAMPLE_KOMMUNER if k in scored])})",
         [k for k in SAMPLE_KOMMUNER if k in scored]),
    ]:
        actual = S_test_wide.reindex(kommuner)
        print(f"\nØkologisk backtest — {label}  bygg {build} → mål {target}")
        print(f"  {'metode':18s} {'alle':>6s} {'store':>6s}")
        rows = {}
        for m, p in preds_for(kommuner).items():
            per_party, overall = _score(p, actual)
            rows[m] = per_party
            print(f"  {m:18s} {overall['all']:6.2f} {overall['big']:6.2f}")
        print("  Per parti (MAE pp): " + " / ".join(rows))
        for p in ENGINE_PARTIES:
            print(f"    {p:8s} " + " / ".join(f"{rows[m][p]:5.2f}" for m in rows))

    # ------------------------------------------------------------------ #
    # De lærte stereotypene — noen illustrerende celler                    #
    # ------------------------------------------------------------------ #
    model = variants["eco_geo"].model
    print("\nLærte stereotyper (eco_geo, uten geo-ledd = 'nasjonal' celle):")
    examples = [
        ("mann", "45-66", "grunnskole", "ovrig"),
        ("mann", "25-44", "uh_lang", "ovrig"),
        ("kvinne", "16-24", "uh_kort", "ovrig"),
        ("kvinne", "67+", "videregaaende", "ovrig"),
        ("mann", "45-66", "fagskole", "lavinntekt"),
    ]
    for cell in examples:
        d = model.cell_dist(cell)
        top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {'/'.join(cell):42s} " + "  ".join(f"{p} {v:.0f}%" for p, v in top))


if __name__ == "__main__":
    main()
