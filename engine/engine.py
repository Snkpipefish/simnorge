"""Spørremotoren — Modul 5.

Ekspanderer personas til pollings-instanser langs den relevante holdningsaksen
(med brøkvekt = persona.weight × P(nivå)), slik at HELE fordelingen — inkludert
halene — bæres inn i LLM-kallene. Poller Claude per instans, cacher på disk
(samme persona-kontekst + spørsmål kaller aldri API på nytt), parser JSON trygt,
og aggregerer vektet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from personas import PersonaSet
from .backend import LLMBackend, ClaudeCLIBackend
from .prompt import QuestionSpec, build_prompt

logger = logging.getLogger("simnorge.motor")

DEFAULT_MODEL = "claude-sonnet-4-6"


# --------------------------------------------------------------------------- #
# Ekspander personas -> pollings-instanser                                    #
# --------------------------------------------------------------------------- #
def expand_by_item(pset: PersonaSet, focus_item: str) -> pd.DataFrame:
    """Én instans per (persona × nivå) på fokus-aksen, brøkvekt = vekt × P(nivå).

    Dette bærer fordelingen (med haler) inn i pollingen i stedet for å kollapse
    personaen til sitt mest sannsynlige nivå."""
    levels = pset.levels[focus_item]
    rows = []
    for p in pset.personas:
        probs = p.attitude_dist[focus_item]
        for lvl, pr in zip(levels, probs):
            if pr <= 0:
                continue
            rows.append({
                "persona_id": p.persona_id, "kommune": p.kommune,
                **p.demographics, "level": lvl, "weight": p.weight * float(pr),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Trygg parsing                                                               #
# --------------------------------------------------------------------------- #
def parse_response(text: str, spec: QuestionSpec) -> dict | None:
    raw = None
    try:
        raw = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                raw = json.loads(m.group(0))
            except Exception:
                raw = None
    if not isinstance(raw, dict) or "posisjon" not in raw:
        logger.warning("Kunne ikke parse LLM-svar: %r", text[:120])
        return None
    try:
        pos = float(raw["posisjon"])
    except Exception:
        return None
    pos = float(np.clip(pos, spec.scale_min, spec.scale_max))
    try:
        cert = float(raw.get("sikkerhet", 0.5))
    except Exception:
        cert = 0.5
    return {"posisjon": pos, "sikkerhet": float(np.clip(cert, 0.0, 1.0)),
            "begrunnelse": str(raw.get("begrunnelse", ""))[:200]}


# --------------------------------------------------------------------------- #
# Poll-motoren                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class PollOutput:
    spec: QuestionSpec
    results: pd.DataFrame          # per instans: + posisjon, sikkerhet, begrunnelse
    n_calls: int                   # faktiske backend-kall (cache-miss)
    n_cached: int
    n_failed: int


def poll(
    instances: pd.DataFrame,
    spec: QuestionSpec,
    *,
    geo_context: dict[str, str | None],
    backend: LLMBackend | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: str | Path = ".cache/engine",
    max_workers: int = 6,
) -> PollOutput:
    """``geo_context``: kommune -> geo-linje uten navn (engine.geo.GeoContext),
    eller None for nasjonal poll. Kommunenavn skal ALDRI inn i prompten."""
    backend = backend or ClaudeCLIBackend()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Bygg prompt + cache-nøkkel per instans.
    prompts, keys = [], []
    for _, row in instances.iterrows():
        geo = geo_context.get(row["kommune"])
        demo = {k: row[k] for k in ("kjonn", "aldersgruppe", "utdanning", "okonomisk_status")
                if k in row}
        p = build_prompt(geo, demo, spec, row["level"])
        prompts.append(p)
        keys.append(hashlib.sha1(f"{model}\n{p}".encode("utf-8")).hexdigest()[:20])

    # Hvilke trenger kall?
    def cache_path(k): return cache / f"{k}.json"
    todo = [(i, prompts[i], keys[i]) for i in range(len(prompts))
            if not cache_path(keys[i]).exists()]

    n_calls = 0

    def run(job):
        i, prompt, key = job
        # Permanent feilet kall = tapt dekning for cellen, IKKE krasj for hele
        # spørringen (samme disiplin som valg-backtesten; cellen forblir
        # ucachet så neste kjøring fyller hullet).
        try:
            text = backend.complete(prompt, model=model)
        except Exception as e:  # noqa: BLE001
            logger.warning("Instans-kall feilet permanent (dekningstap): %s", e)
            return None
        cache_path(key).write_text(json.dumps({"text": text}, ensure_ascii=False), "utf-8")
        return key

    if todo:
        logger.info("Poller %d instanser via %s/%s (%d cachet).",
                    len(todo), backend.name, model, len(prompts) - len(todo))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for key in ex.map(run, todo):
                if key is not None:
                    n_calls += 1

    # Les + parse alle. Manglende cachefil = permanent feilet kall.
    out = instances.copy().reset_index(drop=True)
    positions, certs, just, failed = [], [], [], 0
    for i in range(len(out)):
        path = cache_path(keys[i])
        parsed = None
        if path.exists():
            text = json.loads(path.read_text("utf-8"))["text"]
            parsed = parse_response(text, spec)
        if parsed is None:
            failed += 1
            positions.append(np.nan); certs.append(np.nan); just.append("")
        else:
            positions.append(parsed["posisjon"]); certs.append(parsed["sikkerhet"])
            just.append(parsed["begrunnelse"])
    out["posisjon"] = positions
    out["sikkerhet"] = certs
    out["begrunnelse"] = just
    return PollOutput(spec=spec, results=out, n_calls=n_calls,
                      n_cached=len(prompts) - len(todo), n_failed=failed)


# --------------------------------------------------------------------------- #
# Vektet aggregering (lett — full opprulling er Modul 6)                       #
# --------------------------------------------------------------------------- #
def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["posisjon"].notna()]


def weighted_mean(df: pd.DataFrame) -> float:
    d = _valid(df)
    w = d["weight"].to_numpy()
    return float(np.average(d["posisjon"].to_numpy(), weights=w)) if w.sum() else float("nan")


def weighted_std(df: pd.DataFrame) -> float:
    d = _valid(df)
    w = d["weight"].to_numpy(); x = d["posisjon"].to_numpy()
    if w.sum() == 0:
        return float("nan")
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def level_means(df: pd.DataFrame) -> dict[str, float]:
    d = _valid(df)
    return {lvl: float(np.average(g["posisjon"], weights=g["weight"]))
            for lvl, g in d.groupby("level") if g["weight"].sum() > 0}


def weighted_position_distribution(df: pd.DataFrame, bins: list[tuple[float, float]]) -> dict:
    d = _valid(df)
    w = d["weight"].to_numpy(); x = d["posisjon"].to_numpy()
    tot = w.sum()
    return {f"{lo}-{hi}": float(w[(x >= lo) & (x <= hi)].sum() / tot) if tot else 0.0
            for lo, hi in bins}
