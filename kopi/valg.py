"""Stemmegivning per person — forankret i FAKTISKE valgresultater.

Tre kilder, alle ekte:
- 08092: stortingsvalget 2025, godkjente stemmer per parti per kommune
  (via eksisterende panel-laster med Klass-normalisering av kommunekoder).
- 13085: valgdeltakelse per 1-årsalder × kjønn (unge stemmer sjeldnere).
- 13698: velgerundersøkelsen — partioppslutning etter kjønn × alder ×
  bruttoinntekt, nasjonalt.

Sammensetting:
1. Hver person 18+ blir velger med sannsynlighet fra 13085 (alder × kjønn).
2. Velgerens partisannsynlighet = kommunens FAKTISKE resultat, skråstilt
   med den nasjonale demografitilten fra 13698 — og deretter RAKET per
   kommune slik at kommuneaggregatet fortsatt matcher det faktiske
   resultatet eksakt (opp til trekkstøy). Tilten flytter altså partier
   MELLOM demografiske grupper innen kommunen, aldri kommunens totaler.
3. Under 18: «ikke stemmerett». Ikke-velgere: «stemte ikke».

Antakelsen som ligger igjen: tilten fra 13698 (nasjonal) gjelder i alle
kommuner. Det er samme type dokumenterte uavhengighetsantakelse som resten
av prosjektet — men totalnivåene er målt, ikke antatt.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import SSBClient, KlassClient, PARTY_CODES
from calibration.election_data import load_election_panel

logger = logging.getLogger("simnorge.kopi.valg")

PARTIER = list(PARTY_CODES.values()) + ["Andre"]        # 10 utfall for velgere
IKKE_VELGER = ["stemte ikke", "ikke stemmerett"]
UTFALL = PARTIER + IKKE_VELGER

_ALDER3_GRENSER = [30, 50]                 # 00-29, 30-49, 50+
_ALDER3 = ["00-29", "30-49", "50+"]
_INNT5_GRENSER = [300_000, 500_000, 700_000, 900_000]
_INNT5 = ["01", "02", "03", "04", "05"]
_KJONN = ["1", "2"]
VALGBAR_ALDER = 18


class ValgModell:
    def __init__(self, ssb: SSBClient, klass: KlassClient, *,
                 valgaar: str = "2025", kommune_aargang: str = "2026"):
        self.valgaar = valgaar
        panel = load_election_panel(ssb, klass, valgaar,
                                    reference_year=kommune_aargang)
        self.shares = self._til_shares(panel)                # kommune × 10
        # Kommuner som mistet resultatet i Klass-normaliseringen (splitt
        # 2025→2026) kan fortsatt finnes under 2025-nøkkelen — bruk deres
        # EGET resultat der, ikke nasjonale andeler.
        if panel.dropped_splits:
            selv = self._til_shares(load_election_panel(
                ssb, klass, valgaar, reference_year=valgaar))
            redning = selv.loc[selv.index.intersection(panel.dropped_splits)]
            if len(redning):
                self.shares = pd.concat([self.shares, redning])
                logger.info("Hentet eget %s-resultat for %d splittkommuner "
                            "under valgårets kommunenøkkel: %s", valgaar,
                            len(redning), sorted(redning.index))
        self.turnout = self._last_deltakelse(ssb)            # (2, 106) 0..1
        self.tilt = self._last_tilt(ssb)                     # (2,3,5,10)
        logger.info("Valgmodell: %d kommuner med 2025-resultat, deltakelse "
                    "13085, demografitilt 13698.", len(self.shares))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _til_shares(panel) -> pd.DataFrame:
        piv = panel.pivot()                                  # kommune × 9, pct
        piv["Andre"] = (100.0 - piv.sum(axis=1)).clip(lower=0.0)
        return piv[PARTIER] / 100.0

    def _last_deltakelse(self, ssb: SSBClient) -> np.ndarray:
        koder = [c for c in ssb.variable_codes("13085")["Alder"] if c != "999A"]
        df = ssb.fetch("13085", {
            "Valgtype": ["2"], "Kjonn": _KJONN, "Alder": koder,
            "ContentsCode": ["Valgdeltakelse"], "Tid": [self.valgaar],
        })
        t = np.zeros((2, 106))
        sett = np.zeros((2, 106), dtype=bool)
        for _, r in df.iterrows():
            if bool(r["missing"]) or np.isnan(r["value"]):
                continue
            a = int("".join(ch for ch in r["Alder"] if ch.isdigit()))
            a = min(a, 105)
            si = _KJONN.index(r["Kjonn"])
            t[si, a] = r["value"] / 100.0
            sett[si, a] = True
        # Aldre over kildens maks arver siste kjente verdi; under 18 = 0.
        for si in range(2):
            siste = 0.0
            for a in range(VALGBAR_ALDER, 106):
                if sett[si, a]:
                    siste = t[si, a]
                else:
                    t[si, a] = siste
            t[si, :VALGBAR_ALDER] = 0.0
        return t

    def _last_tilt(self, ssb: SSBClient) -> np.ndarray:
        """(kjønn, alder3, innt5, parti) -> P(parti|celle)/P(parti|alle),
        klippet [0.2, 5]. Prikkede celler får tilt 1 (logget)."""
        rev = {v: k for k, v in PARTY_CODES.items()}
        pkoder = [rev[p] for p in PARTY_CODES.values()] + ["92"]
        df = ssb.fetch("13698", {
            "PolitParti": pkoder, "Kjonn": ["0"] + _KJONN,
            "Alder": ["999A"] + _ALDER3, "BruttoInnte": ["00"] + _INNT5,
            "ContentsCode": ["Velgere"], "Tid": [self.valgaar],
        })

        def vec(kj: str, al: str, inn: str) -> np.ndarray:
            sub = df[(df["Kjonn"] == kj) & (df["Alder"] == al)
                     & (df["BruttoInnte"] == inn)]
            v = np.array([_val(sub, "PolitParti", k) for k in pkoder])
            return v

        nat = vec("0", "999A", "00")
        nat = nat / nat.sum()
        tilt = np.ones((2, 3, 5, len(pkoder)))
        for si, kj in enumerate(_KJONN):
            for ai, al in enumerate(_ALDER3):
                for ii, inn in enumerate(_INNT5):
                    v = vec(kj, al, inn)
                    if np.isnan(v).any() or v.sum() <= 0:
                        logger.warning("13698 celle (%s,%s,%s): prikket — "
                                       "tilt 1 (logget).", kj, al, inn)
                        continue
                    tilt[si, ai, ii] = np.clip((v / v.sum()) / nat, 0.2, 5.0)
        return tilt

    # ------------------------------------------------------------------ #
    def tildel(self, alder: np.ndarray, kjonn_i: np.ndarray,
               kommune: np.ndarray, inntekt: np.ndarray,
               *, seed: int = 0) -> np.ndarray:
        """Utfallskoder inn i UTFALL per person. Deterministisk gitt seed."""
        n = len(alder)
        rng = np.random.default_rng(seed)
        ut = np.full(n, UTFALL.index("ikke stemmerett"), dtype=np.int8)

        myndig = alder >= VALGBAR_ALDER
        a = np.clip(alder, 0, 105)
        stemmer = myndig & (rng.random(n) < self.turnout[kjonn_i, a])
        ut[myndig & ~stemmer] = UTFALL.index("stemte ikke")

        # Cellenøkkel for velgere: kjønn × alder3 × inntektsgruppe.
        a3 = np.searchsorted(_ALDER3_GRENSER, alder, side="right")
        i5 = np.searchsorted(_INNT5_GRENSER,
                             np.nan_to_num(inntekt, nan=0.0), side="right")
        celle = kjonn_i * 15 + a3 * 5 + i5                    # 0..29

        nasjonal = self._nasjonal_share()
        df = pd.DataFrame({"kommune": kommune[stemmer],
                           "celle": celle[stemmer],
                           "idx": np.flatnonzero(stemmer)})
        mangler: set[str] = set()
        for kom, g in df.groupby("kommune", sort=False):
            if kom in self.shares.index:
                s = self.shares.loc[kom].to_numpy()
            else:
                mangler.add(kom); s = nasjonal
            teller = np.bincount(g["celle"], minlength=30)
            q = self._rake(s, teller)                         # (30, 10)
            for c in np.unique(g["celle"]):
                rader = g.loc[g["celle"] == c, "idx"].to_numpy()
                ut[rader] = rng.choice(len(PARTIER), size=len(rader), p=q[c])
        if mangler:
            logger.warning("%d kommuner uten 2025-resultat fikk nasjonale "
                           "andeler (logget): %s", len(mangler), sorted(mangler))
        logger.info("Tildelte stemmegivning: %d velgere, %d hjemmesittere, "
                    "%d uten stemmerett.", int(stemmer.sum()),
                    int((myndig & ~stemmer).sum()), int((~myndig).sum()))
        return ut

    def _nasjonal_share(self) -> np.ndarray:
        return self.shares.mean(axis=0).to_numpy() / self.shares.mean(axis=0).sum()

    def _rake(self, s: np.ndarray, teller: np.ndarray,
              *, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
        """Partisannsynlighet per celle slik at (a) cellene følger tilten og
        (b) det VEKTEDE kommuneaggregatet matcher det faktiske resultatet s."""
        s = np.maximum(s, 0)
        s = s / s.sum()
        tiltm = self.tilt.reshape(30, -1)                     # (30, 10)
        m = s[None, :] * tiltm
        f = np.ones(len(s))
        n_tot = teller.sum()
        if n_tot == 0:
            return m / m.sum(axis=1, keepdims=True)
        for _ in range(iters):
            q = m * f[None, :]
            q = q / q.sum(axis=1, keepdims=True)
            agg = (teller[:, None] * q).sum(axis=0) / n_tot
            if np.abs(agg - s).max() < tol:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                f = f * np.where(agg > 0, s / agg, 1.0)
        return q


def _val(sub: pd.DataFrame, dim: str, kode: str) -> float:
    r = sub[sub[dim] == kode]
    if r.empty or bool(r.iloc[0]["missing"]):
        return np.nan
    return float(r.iloc[0]["value"])
