"""Holdningsberikelse — Modul 3.

For hver syntetisk person festes en HOLDNINGSPROFIL ved oppslag i SSBs nasjonale
holdningstabeller, basert på personens demografiske akser. Profilen er en
DEMOGRAFISK PRIOR, ikke en målt holdning (se memo `extrapolation-layers`):
den er ment som myk kontekst for LLM-en i Modul 5 — ikke et tall motoren senere
skal tvinges til å reprodusere.

Avgjørende designvalg:
- Profilen trekkes fra en FORDELING med eksplisitt spredning (SSBs andel
  lav/middels/høy skår), ikke fra gjennomsnittet. To personer i samme
  demografiske celle kan derfor få ulik profil — trukket fra gruppens fordeling
  med fast seed. Aggregert reproduserer trekningene SSBs fordeling.
- Aksene krysses ikke i kildedata. Vi kombinerer dem under en eksplisitt
  uavhengighetsantakelse og logger den. Vi fabrikkerer ALDRI en sammenheng som
  ikke finnes i kilden:
    * 13834 tillit      -> personens (aldersgruppe × kjønn)   [ren match]
    * 13790 livskvalitet-> personens (kjønn × aldersgruppe)   [aldersbånd forsonet]
    * 13831 frivillig   -> personens arbeidsmarkedsstatus     [ren match fra 2026-07]

STATUS-TILT (fra 2026-07, da personene fikk arbeidsmarkedsstatus fra 13563):
13839 (tillit) og 13793 (livskvalitet) måler de samme aksene brutt på
økonomisk status. Vi kombinerer med (alder×kjønn)-prioren multiplikativt:
P(nivå | alder, kjønn, status) ∝ P(nivå | alder, kjønn) × P(nivå | status)/P(nivå)
— en eksplisitt rake/odds-justering. Kildene krysser ikke akser vi ikke har;
faktoren er målt nasjonalt per status og logges der den er prikket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import SSBClient
from population.schema import AGE_LABELS, SEX_LABELS, STATUS_LABELS, STATUS_TO_OKSTATUS

logger = logging.getLogger("simnorge.holdning")

# Innholdsaksene vi henter fordelingen fra (lav/middels/høy).
TRUST_LEVELS = ["lav", "middels", "hoy"]
SCORE_LEVELS = ["lav", "middels", "hoy"]
PART_LEVELS = ["deltok", "ikke"]

# Tillit (13834): 6 mål for tillit.
TRUST_ITEMS = {
    "01": "kommunestyret", "02": "politiet", "03": "rettsvesenet",
    "04": "politisk_system", "05": "nyhetsmediene", "06": "ukjente_personer",
}
# Livskvalitet (13790): mening og mestring.
MENING_ITEMS = {
    "01": "optimisme", "02": "mening", "03": "engasjement",
    "04": "mestring", "05": "sosiale_relasjoner", "06": "bidrag_andre",
}
# Frivillig innsats (13831): aktivitetstyper.
FRIVILLIG_ITEMS = {
    "01": "styrearbeid", "02": "instruksjon", "03": "dugnad",
    "04": "informasjonsarbeid", "90": "annet",
}

# Vår aldersgruppe -> kildens aldersband.
TILLIT_AGE = {"16-24": "16-24", "25-44": "25-44", "45-66": "45-66", "67+": "067+"}
# 13790 mangler 16-17 (bruker 18-24) og deler 67+ i 67-79/80+ (forsones).
MENING_AGE = {"16-24": ["18-24"], "25-44": ["25-44"], "45-66": ["45-66"],
              "67+": ["67-79", "80+"]}
SEX_TO_CODE = {"mann": "1", "kvinne": "2"}


@dataclass
class AttitudeItem:
    """Ett holdningsmål med en oppslagstabell fra demografisk celle til
    sannsynlighetsvektor over nivåene."""

    column: str                 # kolonnenavn på personen, f.eks. "tillit_politiet"
    levels: list[str]
    lut: np.ndarray             # form (4_alder, 2_kjonn, L) ELLER (1,1,L) for flat
    keyed_by_demography: bool   # False = flat nasjonal (frivillig)
    national: np.ndarray        # SSBs nasjonale fordeling (L,) for konsistens-sjekk
    status_factor: np.ndarray | None = None   # (L, 6) odds-tilt per arbeidsmarkedsstatus


def _latest(ssb: SSBClient, table_id: str) -> str:
    return ssb.variable_codes(table_id)["Tid"][-1]


def _normalize(vec: np.ndarray) -> np.ndarray:
    s = vec.sum()
    return vec / s if s > 0 else np.full_like(vec, 1.0 / len(vec))


# --------------------------------------------------------------------------- #
# Status-tilt: P(nivå|status)/P(nivå|alt) fra 13839/13793                      #
# --------------------------------------------------------------------------- #
def _status_factors(ssb: SSBClient, table: str, item_dim: str, item_codes: list[str],
                    cc: list[str], year: str) -> dict[str, np.ndarray]:
    """item-kode -> (L, 6) multiplikativ faktor per STATUS_LABELS.

    Basen er det AntallSvar-VEKTEDE snittet over statusene (selv-konsistent:
    vektet sum av tiltede fordelinger reproduserer basen; 13793 har heller
    ingen 'status i alt'-kode). Prikkede celler -> faktor 1 (logget)."""
    if year not in ssb.variable_codes(table)["Tid"]:
        year = _latest(ssb, table)
    codes = [STATUS_TO_OKSTATUS[s] for s in STATUS_LABELS]
    df = ssb.fetch(table, {item_dim: item_codes, "OkonomiskStatus": codes,
                           "ContentsCode": cc + ["AntallSvar"], "Tid": [year]})
    out = {}
    for code in item_codes:
        dists, weights = {}, {}
        for s in STATUS_LABELS:
            vec = _cell_vec(df, cc, **{item_dim: code,
                                       "OkonomiskStatus": STATUS_TO_OKSTATUS[s]})
            w = _cell_scalar(df, "AntallSvar", **{item_dim: code,
                                                  "OkonomiskStatus": STATUS_TO_OKSTATUS[s]})
            if np.isnan(vec).any() or np.isnan(w) or w <= 0:
                logger.warning("%s %s status=%s: prikket — ingen status-tilt for "
                               "denne cellen (faktor 1, logget).", table, code, s)
                continue
            dists[s], weights[s] = _normalize(vec), w
        fac = np.ones((len(cc), len(STATUS_LABELS)))
        if dists:
            wsum = sum(weights.values())
            base = sum(dists[s] * weights[s] for s in dists) / wsum
            for si, s in enumerate(STATUS_LABELS):
                if s not in dists:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.where(base > 0, dists[s] / base, 1.0)
                fac[:, si] = np.clip(f, 0.05, 20.0)
        out[code] = fac
    return out


# --------------------------------------------------------------------------- #
# 13834 — tillit etter alder × kjønn                                          #
# --------------------------------------------------------------------------- #
def _load_tillit(ssb: SSBClient, year: str) -> list[AttitudeItem]:
    cc = ["AndelLavTillit", "AndelMiddelsTillit", "AndelHoyTillit"]
    df = ssb.fetch("13834", {
        "Tillit": list(TRUST_ITEMS), "Alder": list(TILLIT_AGE.values()),
        "Kjonn": ["1", "2"], "ContentsCode": cc, "Tid": [year],
    })
    nat = ssb.fetch("13834", {
        "Tillit": list(TRUST_ITEMS), "Alder": ["999"], "Kjonn": ["0"],
        "ContentsCode": cc, "Tid": [year],
    })
    factors = _status_factors(ssb, "13839", "Tillit", list(TRUST_ITEMS), cc, year)
    items = []
    for code, name in TRUST_ITEMS.items():
        lut = np.zeros((len(AGE_LABELS), len(SEX_LABELS), 3))
        for ai, age in enumerate(AGE_LABELS):
            for si, sex in enumerate(SEX_LABELS):
                vec = _cell_vec(df, cc, Tillit=code, Alder=TILLIT_AGE[age],
                                Kjonn=SEX_TO_CODE[sex])
                lut[ai, si] = _resolve(vec, name, f"{age}/{sex}", logger)
        nvec = _cell_vec(nat, cc, Tillit=code, Alder="999", Kjonn="0")
        items.append(AttitudeItem(f"tillit_{name}", TRUST_LEVELS, lut, True,
                                  _normalize(nvec), status_factor=factors.get(code)))
    return items


# --------------------------------------------------------------------------- #
# 13790 — livskvalitet etter kjønn × alder (aldersbånd forsones)              #
# --------------------------------------------------------------------------- #
def _load_mening(ssb: SSBClient, year: str) -> list[AttitudeItem]:
    cc = ["AndelLavSkar", "AndelMidSkar", "AndelHoySkar", "AntallSvar"]
    src_ages = sorted({a for v in MENING_AGE.values() for a in v})
    df = ssb.fetch("13790", {
        "Mening": list(MENING_ITEMS), "Kjonn": ["1", "2"], "Alder": src_ages,
        "ContentsCode": cc, "Tid": [year],
    })
    nat = ssb.fetch("13790", {
        "Mening": list(MENING_ITEMS), "Kjonn": ["0"], "Alder": ["18+"],
        "ContentsCode": cc, "Tid": [year],
    })
    logger.info("13790: aldersbånd forsonet (16-24←18-24; 67+←vektet 67-79+80+) "
                "— eksplisitt ekstrapolering.")
    factors = _status_factors(ssb, "13793", "Mening", list(MENING_ITEMS), cc[:3], year)
    items = []
    dist_cc = cc[:3]
    for code, name in MENING_ITEMS.items():
        lut = np.zeros((len(AGE_LABELS), len(SEX_LABELS), 3))
        for ai, age in enumerate(AGE_LABELS):
            for si, sex in enumerate(SEX_LABELS):
                # Vektet sammenslåing av kildebånd via AntallSvar.
                acc = np.zeros(3); wsum = 0.0
                for sage in MENING_AGE[age]:
                    vec = _cell_vec(df, dist_cc, Mening=code, Kjonn=SEX_TO_CODE[sex],
                                    Alder=sage)
                    w = _cell_scalar(df, "AntallSvar", Mening=code,
                                     Kjonn=SEX_TO_CODE[sex], Alder=sage)
                    if not np.isnan(vec).any() and w and not np.isnan(w):
                        acc += vec * w; wsum += w
                lut[ai, si] = _resolve(acc / wsum if wsum else np.array([np.nan]*3),
                                       name, f"{age}/{sex}", logger)
        nvec = _cell_vec(nat, dist_cc, Mening=code, Kjonn="0", Alder="18+")
        items.append(AttitudeItem(f"livskvalitet_{name}", SCORE_LEVELS, lut, True,
                                  _normalize(nvec), status_factor=factors.get(code)))
    return items


# --------------------------------------------------------------------------- #
# 13831 — frivillig innsats: flat nasjonal (ingen matchende personakse)        #
# --------------------------------------------------------------------------- #
def _load_frivillig(ssb: SSBClient, year: str) -> list[AttitudeItem]:
    codes = ["00"] + [STATUS_TO_OKSTATUS[s] for s in STATUS_LABELS]
    df = ssb.fetch("13831", {
        "TypeFrivilligInnsats": list(FRIVILLIG_ITEMS), "OkonomiskStatus": codes,
        "ContentsCode": ["Andel"], "Tid": [year],
    })
    logger.info("13831 frivillig innsats: kobles nå DIREKTE på personens "
                "arbeidsmarkedsstatus (13563-aksen) — den gamle ærlighetsgrensen "
                "(flat nasjonal) gjelder bare personer med ukjent status.")

    def share(code: str, status: str) -> float | None:
        row = df[(df["TypeFrivilligInnsats"] == code) & (df["OkonomiskStatus"] == status)]
        if row.empty or bool(row.iloc[0]["missing"]):
            return None
        return float(row.iloc[0]["value"]) / 100.0

    items = []
    for code, name in FRIVILLIG_ITEMS.items():
        tot = share(code, "00") or 0.0
        vec = np.array([tot, 1.0 - tot])
        fac = np.ones((2, len(STATUS_LABELS)))
        for si, s in enumerate(STATUS_LABELS):
            sh = share(code, STATUS_TO_OKSTATUS[s])
            if sh is None or not (0 < tot < 1):
                logger.warning("13831 %s status=%s: prikket — faktor 1 (logget).",
                               name, s)
                continue
            fac[:, si] = np.clip([sh / tot, (1 - sh) / (1 - tot)], 0.05, 20.0)
        items.append(AttitudeItem(f"frivillig_{name}", PART_LEVELS,
                                  vec.reshape(1, 1, 2), False, vec,
                                  status_factor=fac))
    return items


# --------------------------------------------------------------------------- #
# Hjelpere for celleoppslag                                                   #
# --------------------------------------------------------------------------- #
def _cell_vec(df: pd.DataFrame, content_codes: list[str], **dims) -> np.ndarray:
    mask = pd.Series(True, index=df.index)
    for k, v in dims.items():
        mask &= df[k] == v
    sub = df[mask]
    out = []
    for cc in content_codes:
        r = sub[sub["ContentsCode"] == cc]
        out.append(np.nan if r.empty or bool(r.iloc[0]["missing"]) else float(r.iloc[0]["value"]))
    return np.array(out)


def _cell_scalar(df: pd.DataFrame, content_code: str, **dims) -> float:
    v = _cell_vec(df, [content_code], **dims)
    return float(v[0])


def _resolve(vec: np.ndarray, item: str, cell: str, log) -> np.ndarray:
    """Normaliser en celles nivåfordeling; håndter prikking eksplisitt."""
    if np.isnan(vec).any():
        log.warning("Holdning %s celle %s: prikket/manglende fordeling — "
                    "faller tilbake på uniform (logget, ikke stille imputert).", item, cell)
        return np.full(len(vec), 1.0 / len(vec))
    return _normalize(vec)


# --------------------------------------------------------------------------- #
# Berik en befolkning                                                         #
# --------------------------------------------------------------------------- #
def load_attitude_items(ssb: SSBClient, *, year: str | None = None) -> list[AttitudeItem]:
    return (
        _load_tillit(ssb, year or _latest(ssb, "13834"))
        + _load_mening(ssb, year or _latest(ssb, "13790"))
        + _load_frivillig(ssb, year or _latest(ssb, "13831"))
    )


def enrich_population(
    persons: pd.DataFrame,
    items: list[AttitudeItem],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Fest en holdningsprofil på hver person ved å TREKKE et nivå per mål fra
    personens demografiske celles fordeling. Deterministisk gitt ``seed``."""
    out = persons.copy()
    n = len(out)
    if n == 0:
        for it in items:
            out[it.column] = pd.Series(dtype="object")
        return out

    age_i = out["aldersgruppe"].map({a: i for i, a in enumerate(AGE_LABELS)}).to_numpy()
    sex_i = out["kjonn"].map({s: i for i, s in enumerate(SEX_LABELS)}).to_numpy()
    rng = np.random.default_rng(seed)

    # Status-tilt: kun for personer med kjent arbeidsmarkedsstatus.
    status_i = None
    if "arbeidsmarkedsstatus" in out.columns:
        m = out["arbeidsmarkedsstatus"].map(
            {s: i for i, s in enumerate(STATUS_LABELS)})
        status_i = m.to_numpy(dtype="float")       # NaN for 'ukjent'

    for it in items:
        if it.keyed_by_demography:
            probs = it.lut[age_i, sex_i]          # (n, L)
        else:
            probs = np.repeat(it.lut[0, 0][None, :], n, axis=0)
        if it.status_factor is not None and status_i is not None:
            known = ~np.isnan(status_i)
            if known.any():
                fac = it.status_factor.T[status_i[known].astype(int)]   # (k, L)
                tilted = probs[known] * fac
                probs = probs.copy()
                probs[known] = tilted / tilted.sum(axis=1, keepdims=True)
        u = rng.random(n)
        cum = np.cumsum(probs, axis=1)
        idx = (u[:, None] >= cum).sum(axis=1)
        idx = np.clip(idx, 0, len(it.levels) - 1)
        out[it.column] = pd.Categorical.from_codes(idx, categories=it.levels)
    logger.info("Beriket %d personer med %d holdningsmål (demografisk prior, "
                "ikke måling).", n, len(items))
    return out
