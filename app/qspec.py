"""Fritt spørsmål -> QuestionSpec — Modul 8.

LLM-en fyller gapet mellom de grove SSB-aksene og et fritt spørsmål (jf.
PLAN): ett kall tolker spørsmålet, velger NÆRMESTE holdningsakse personaens
disposisjon skal hentes fra, og formulerer nivå-frasene. Streng validering;
kan ikke spørsmålet tolkes, sier vi det ÆRLIG (feil, ikke stille fallback).

Domeneklassifisering (politikk/livskvalitet/annet) følger med fordi den
styrer ærlighetsbanneret: geografisk differensiering er VALIDERT for
politiske spørsmål (valg-backtest) men målt til ~flat for livskvalitet
(SSB 13798/13799) — UI-et må si det.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from engine.backend import LLMBackend, ClaudeCLIBackend
from engine.engine import DEFAULT_MODEL
from engine.prompt import QuestionSpec
from enrich import TRUST_ITEMS, MENING_ITEMS

logger = logging.getLogger("simnorge.qspec")

# Akser LLM-en kan velge blant (3-nivå: lav/middels/hoy) — avledet fra
# berikelseslaget så listen aldri drifter fra hva personaene faktisk bærer.
ALLOWED_AXES = ([f"tillit_{n}" for n in TRUST_ITEMS.values()]
                + [f"livskvalitet_{n}" for n in MENING_ITEMS.values()])
DOMAINS = ["politikk", "livskvalitet", "annet"]


class QSpecError(ValueError):
    """Spørsmålet kunne ikke tolkes til en målbar spesifikasjon."""


@dataclass
class InterpretedQuestion:
    spec: QuestionSpec
    domain: str                  # politikk | livskvalitet | annet
    axis_reason: str             # LLM-ens begrunnelse for aksevalget


ANALYSIS_PROMPT = """Du oversetter et fritt meningsmålingsspørsmål til en målbar spesifikasjon
for en syntetisk befolkningsundersøkelse i Norge.

Spørsmål: «{question}»

Tilgjengelige holdningsakser (velg den som er NÆRMEST tematisk beslektet —
personens etablerte disposisjon på denne aksen blir utgangspunktet for svaret):
{axes}

Svar KUN med JSON:
{{
 "axis": "<én av aksene over>",
 "axis_reason": "<maks 12 ord: hvorfor denne aksen er nærmest>",
 "domain": "<politikk | livskvalitet | annet>",
 "scale_low_label": "<hva 0 betyr for DETTE spørsmålet, maks 6 ord>",
 "scale_high_label": "<hva 10 betyr, maks 6 ord>",
 "level_phrasing": {{
   "lav": "<persona-instruks, f.eks. 'LAV tillit til X.' tilpasset aksen>",
   "middels": "<...>",
   "hoy": "<...>"
 }}
}}"""


def _cache_path(cache_dir: Path, model: str, question: str) -> Path:
    key = hashlib.sha1(f"{model}\n{question}".encode()).hexdigest()[:20]
    return cache_dir / f"{key}.json"


def interpret_question(
    question: str,
    *,
    backend: LLMBackend | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: str | Path = ".cache/qspec",
) -> InterpretedQuestion:
    question = question.strip()
    if len(question) < 5:
        raise QSpecError("Spørsmålet er for kort til å tolkes.")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache, model, question)

    if path.exists():
        raw = json.loads(path.read_text("utf-8"))
    else:
        backend = backend or ClaudeCLIBackend()
        text = backend.complete(
            ANALYSIS_PROMPT.format(question=question, axes="\n".join(f"- {a}" for a in ALLOWED_AXES)),
            model=model)
        raw = _parse(text)
        if raw is None:
            raise QSpecError(f"Kunne ikke tolke spørsmålet (uparsbart LLM-svar): {text[:120]!r}")
        path.write_text(json.dumps(raw, ensure_ascii=False), "utf-8")

    axis = raw.get("axis")
    phr = raw.get("level_phrasing") or {}
    if axis not in ALLOWED_AXES:
        raise QSpecError(f"Ugyldig akse fra tolkningen: {axis!r}")
    if sorted(phr) != ["hoy", "lav", "middels"] or not all(
            isinstance(v, str) and v.strip() for v in phr.values()):
        raise QSpecError(f"Ufullstendige nivå-fraser: {phr!r}")
    domain = raw.get("domain") if raw.get("domain") in DOMAINS else "annet"

    spec = QuestionSpec(
        question=question, scale_min=0, scale_max=10,
        scale_low_label=str(raw.get("scale_low_label") or "helt uenig/negativ"),
        scale_high_label=str(raw.get("scale_high_label") or "helt enig/positiv"),
        focus_item=axis,
        level_phrasing={k: str(v) for k, v in phr.items()},
    )
    return InterpretedQuestion(spec=spec, domain=domain,
                               axis_reason=str(raw.get("axis_reason", "")))


def _parse(text: str) -> dict | None:
    try:
        raw = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            raw = json.loads(m.group(0))
        except Exception:
            return None
    return raw if isinstance(raw, dict) else None
