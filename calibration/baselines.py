"""Trivielle baseline-prediktorer — Modul 7 (skjelett).

Disse er midlertidige stubber som lar backtest-løkka kjøre end-to-end før den
ekte spørremotoren finnes. De setter terskelen motoren senere må slå:

- ``persistence_baseline``: "ingenting har endret seg siden forrige valg" —
  bruk byggeårets kommuneresultat direkte. Den STERKE baselinen: en modell som
  ikke slår denne, tilfører ingenting.
- ``national_mean_baseline``: gi hver kommune det nasjonale snittet fra
  byggeåret. NULL-modellen uten geografi — viser verdien av geografisk signal
  (kommune-korrelasjon blir per definisjon udefinert).

Prediktor-kontrakten tar BARE byggeåret (``fit``). Den ser aldri målåret. Det
er den strukturelle garantien for out-of-sample (se backtest.py).
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .election_data import ElectionPanel


class Predictor(Protocol):
    """En prediktor får byggeårets panel + hvilke kommuner/partier som skal
    predikeres, og returnerer en DataFrame med kolonner [kommune, party, pred_pct].

    Den får BEVISST ikke målårets data — det gjør det strukturelt umulig å
    validere på treningssettet."""

    name: str

    def __call__(
        self, fit: ElectionPanel, target_kommunes: list[str], parties: list[str]
    ) -> pd.DataFrame: ...


class _Persistence:
    name = "persistence (forrige valg per kommune)"

    def __call__(self, fit, target_kommunes, parties):
        df = fit.shares.rename(columns={"pct": "pred_pct"})
        # Bare kommuner vi faktisk skal predikere (snitt-join skjer i harnessen).
        return df[df["kommune"].isin(set(target_kommunes))][["kommune", "party", "pred_pct"]]


class _NationalMean:
    name = "national_mean (nasjonalt snitt, ingen geografi)"

    def __call__(self, fit, target_kommunes, parties):
        # Nasjonalt snitt = stemmevektet andel i byggeåret (parti-stemmer over
        # alle kommuner / alle godkjente stemmer over alle kommuner).
        nat_total = float(fit.total_votes.sum())
        votes = fit.shares.merge(fit.total_votes, on="kommune")
        votes["party_votes"] = votes["pct"] / 100.0 * votes["total_votes"]
        nat = votes.groupby("party")["party_votes"].sum() / nat_total * 100.0
        rows = [{"kommune": k, "party": p, "pred_pct": float(nat.get(p, float("nan")))}
                for k in target_kommunes for p in nat.index]
        return pd.DataFrame(rows)


persistence_baseline: Predictor = _Persistence()
national_mean_baseline: Predictor = _NationalMean()
