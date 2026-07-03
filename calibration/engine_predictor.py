"""Spørremotoren som valg-prediktor — Modul 6 (lukker backtest-loopen).

Den ekte, uavhengige testen: parti-preferanse er det eneste kommunale fasit-
signalet vi ALDRI har matet inn i prioren.

TRE VARIANTER (for å skille ekte resonnement fra memorering):
- ``demographic_only``: kun demografi + holdningsprior. INGEN sted/inntekt.
  Lekkasjefritt referansepunkt (celle-agnostisk → ~antall celler kall).
- ``geo_no_name``: + sentralitet (sentrum/distrikt) og kommunens inntektsnivå,
  men UTEN kommunenavn. Tester om ekte demografisk-geografisk resonnement løfter.
- ``with_geography``: + kommunenavn. Kan løfte via resonnement ELLER via at
  modellen HUSKER hvordan kommunen stemte (memorering) — derfor egen variant.

Hvis geo_no_name løfter like mye som with_geography, kommer gevinsten fra
resonnement; hvis bare navn-varianten løfter, lukter det memorering.

Kostnad: demographic_only og geo_no_name er celle-agnostiske → kall skalerer med
(celler × kontekst-nivåer), ikke kommuner. with_geography har navn → kall
skalerer med celler × kommuner (en full nasjonal kjøring blir dyr der).
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

from data import SSBClient, KlassClient, PARTY_CODES
from population import build_population
from population.schema import AGE_LABELS, SEX_LABELS
from enrich import enrich_population
from personas import build_personas
from engine.backend import LLMBackend, ClaudeCLIBackend
from engine.engine import DEFAULT_MODEL
from engine.geo import (  # delt geo-kontekst (flyttet hit fra denne modulen)
    SENT_MARKER, income_tier, load_centrality, lowinc_tier,
)

logger = logging.getLogger("simnorge.motor.valg")

ENGINE_PARTIES = ["Ap", "Høyre", "FrP", "Sp", "SV", "Venstre", "KrF", "MDG", "Rødt"]
NAME_TO_CODE = {v: k for k, v in PARTY_CODES.items()}

AGE_T = {"16-24": "16–24 år", "25-44": "25–44 år", "45-66": "45–66 år", "67+": "67+ år"}
EDU_T = {"grunnskole": "grunnskole", "videregaaende": "videregående", "fagskole": "fagskole",
         "uh_kort": "kort høyere utdanning", "uh_lang": "lang høyere utdanning",
         "uoppgitt": "uoppgitt utdanning"}
ECON_T = {"lavinntekt": "lav husholdningsinntekt", "ovrig": "vanlig/høyere inntekt",
          "ukjent": "uoppgitt inntekt"}
TRUST_T = {"lav": "lav", "middels": "middels", "hoy": "høy"}
STATUS_T = {"yrkesaktiv": "yrkesaktiv", "arbeidsledig": "arbeidsledig",
            "student": "student/elev", "ufor": "arbeidsufør",
            "pensjonist": "pensjonist", "annet": "utenfor arbeidsstyrken (annet)"}
# Statuser med lavere andel enn dette i personens aldersbånd ekspanderes ikke
# (kostnadskontroll: haler på <1 % gir celler som ikke flytter aggregatet).
STATUS_MIN_SHARE = 0.01


@dataclass(frozen=True)
class Variant:
    key: str
    income: bool
    sentralitet: bool
    name: bool


VARIANTS = {
    "demographic_only": Variant("demographic_only", False, False, False),
    "geo_no_name": Variant("geo_no_name", True, True, False),
    "with_geography": Variant("with_geography", True, True, True),
}


def _modal_polsys_trust(item, aldersgruppe: str, kjonn: str,
                        status: str | None = None) -> str:
    """Modal tillit for cellen — status-tiltet når personen har status
    (samme rake som Modul 3, så prompt-prioren er konsistent med berikelsen)."""
    ai = AGE_LABELS.index(aldersgruppe); si = SEX_LABELS.index(kjonn)
    probs = item.lut[ai, si]
    if status is not None and getattr(item, "status_factor", None) is not None:
        from population.schema import STATUS_LABELS
        if status in STATUS_LABELS:
            probs = probs * item.status_factor[:, STATUS_LABELS.index(status)]
    return ["lav", "middels", "hoy"][int(np.argmax(probs))]


def party_prompt(demo: dict, trust: str, *, status: str, geo_line: str | None) -> str:
    bosted = f"\n{geo_line}" if geo_line else ""
    return f"""Du estimerer stemmegivning ved norsk stortingsvalg for en demografisk GRUPPE (ikke én person).

Gruppe i Norge: {demo['kjonn']}, {AGE_T.get(demo['aldersgruppe'], demo['aldersgruppe'])}, \
{EDU_T.get(demo['utdanning'], demo['utdanning'])}, {ECON_T.get(demo['okonomisk_status'], demo['okonomisk_status'])}, \
{STATUS_T.get(status, status)}.{bosted}
Holdningsprior for gruppen: {TRUST_T.get(trust, trust)} tillit til det politiske systemet.

Anslå hvordan gruppen fordeler stemmene på partiene i prosent (sum = 100):
{", ".join(ENGINE_PARTIES)}.
Tenk på hvem OG hvor personen er. Reflekter reell variasjon i gruppen — IKKE legg alt på ett parti.

Svar KUN med JSON, uten annen tekst:
{{{", ".join(f'"{p}": <prosent>' for p in ENGINE_PARTIES)}}}"""


def _parse_party(text: str) -> dict | None:
    raw = None
    try:
        raw = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                raw = json.loads(m.group(0))
            except Exception:
                return None
    if not isinstance(raw, dict):
        return None
    vec = {}
    for p in ENGINE_PARTIES:
        try:
            vec[p] = max(0.0, float(raw.get(p, 0.0)))
        except Exception:
            vec[p] = 0.0
    s = sum(vec.values())
    return {p: v / s * 100.0 for p, v in vec.items()} if s > 0 else None


class EnginePredictor:
    name = "simnorge-engine"

    def __init__(self, ssb: SSBClient, klass: KlassClient, items, *,
                 variant: str = "demographic_only",
                 pop_year: str = "2024", granularity: str = "full",
                 communes: list[str] | None = None,
                 backend: LLMBackend | None = None, model: str = DEFAULT_MODEL,
                 cache_dir: str | Path = ".cache/engine_party", max_workers: int = 8,
                 turnout_weighting: bool = False):
        # turnout_weighting=False som standard: MÅLT (2021→2025, 8 kommuner) at
        # eksplisitt vekting gjør motoren marginalt svakere — også med full
        # kjønn×alder×utdanning-LUT (3.71→3.81 pp MAE geo_no_name; etter affin
        # korreksjon eksakt likt, 2.01). LLM-ens celleanslag er allerede
        # implisitt deltakelsesvektet; vekting dobbelteller gradientene.
        self.ssb, self.klass, self.items = ssb, klass, items
        self.variant = VARIANTS[variant]
        self.pop_year, self.granularity = pop_year, granularity
        self.communes = set(communes) if communes else None
        self.backend = backend or ClaudeCLIBackend()
        self.model = model
        self.cache_dir = Path(cache_dir) / self.variant.key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.turnout_weighting = turnout_weighting
        self.name = f"engine_{self.variant.key}"

        self._polsys = next(i for i in items if i.column == "tillit_politisk_system")
        self._national_median = self._fetch_national_median()
        self._kommune_navn = {}
        self._sentralitet = load_centrality(klass) if (self.variant.sentralitet or True) else {}
        self.last_personas: dict[str, object] = {}
        self._status_mix: dict[str, object] = {}
        self.n_unique_cells = 0
        self.n_failed_cells = 0

    def _fetch_national_median(self) -> float:
        df = self.ssb.fetch("06944", {"Region": ["0"], "HusholdType": ["0000"],
                                      "ContentsCode": ["SamletInntekt"],
                                      "Tid": [self.ssb.variable_codes("06944")["Tid"][-1]]})
        return float(df.iloc[0]["value"])

    def _geo_line(self, kommune: str, persona) -> str | None:
        if not (self.variant.income or self.variant.sentralitet or self.variant.name):
            return None
        parts = []
        if self.variant.name:
            navn = self._kommune_navn.get(kommune, kommune)
            sent = SENT_MARKER.get(self._sentralitet.get(kommune, ""), "")
            parts.append(f"Bosted: {navn} ({sent})" if sent else f"Bosted: {navn}")
        elif self.variant.sentralitet:
            parts.append(f"Bosted: {SENT_MARKER.get(self._sentralitet.get(kommune, ''), 'et sted i Norge')}")
        if self.variant.income:
            parts.append(f"kommunens økonomi: {income_tier(persona.median_husholdningsinntekt, self._national_median)}, "
                         f"{lowinc_tier(persona.lavinntekt_andel)}")
        return ". ".join(parts) + "." if parts else None

    def _cellkey(self, kommune: str, persona, status: str) -> tuple:
        trust = _modal_polsys_trust(self._polsys, persona.demographics["aldersgruppe"],
                                    persona.demographics["kjonn"], status)
        d = persona.demographics
        base = (d["kjonn"], d["aldersgruppe"], d["utdanning"], d["okonomisk_status"],
                status, trust)
        if self.variant.name:
            return base + (kommune,)
        if self.variant.income or self.variant.sentralitet:
            return base + (self._sentralitet.get(kommune, ""),
                           income_tier(persona.median_husholdningsinntekt, self._national_median),
                           lowinc_tier(persona.lavinntekt_andel))
        return base

    def _status_shares(self, kommune: str, aldersgruppe: str) -> list[tuple[str, float]]:
        """[(status, andel)] i personens aldersbånd, kuttet ved STATUS_MIN_SHARE
        og renormalisert (kostnadskontroll, logget antakelse i status.py)."""
        from population.schema import STATUS_LABELS
        mix = self._status_mix[kommune]
        probs = mix.probs[AGE_LABELS.index(aldersgruppe)]
        pairs = [(s, float(p)) for s, p in zip(STATUS_LABELS, probs)
                 if p >= STATUS_MIN_SHARE]
        tot = sum(p for _, p in pairs) or 1.0
        return [(s, p / tot) for s, p in pairs]

    def __call__(self, fit_panel, target_kommunes, parties) -> pd.DataFrame:
        from population import build_status_mix
        from .turnout import load_turnout, turnout_weight

        # Deltakelse fra BYGGEÅRET (aldri målåret — lekkasjedisiplin).
        # None -> uniform vekt (logget i turnout.py).
        tlut = load_turnout(self.ssb, fit_panel.year) if self.turnout_weighting else None

        communes = [k for k in target_kommunes if self.communes is None or k in self.communes]
        # Bygg personas + statusmiks + kommunenavn.
        cell_prompt: dict[tuple, str] = {}
        self._status_mix = {}
        for k in communes:
            pop = build_population(self.ssb, k, year=self.pop_year)
            if pop.skipped:
                continue
            enr = enrich_population(pop.persons, self.items)
            pset = build_personas(enr, self.items, granularity=self.granularity)
            self.last_personas[k] = pset
            self._status_mix[k] = build_status_mix(self.ssb, k, year=self.pop_year)

        # Kommunenavn-oppslag (én gang).
        navn_df = self.klass.codes_at(self.pop_year)
        self._kommune_navn = dict(zip(navn_df["code"], navn_df["name"]))

        # Persona × arbeidsmarkedsstatus (vekt = personavekt × P(status|alder)).
        for k, pset in self.last_personas.items():
            for p in pset.personas:
                for status, _ in self._status_shares(k, p.demographics["aldersgruppe"]):
                    ck = self._cellkey(k, p, status)
                    if ck not in cell_prompt:
                        trust = _modal_polsys_trust(
                            self._polsys, p.demographics["aldersgruppe"],
                            p.demographics["kjonn"], status)
                        cell_prompt[ck] = party_prompt(
                            p.demographics, trust, status=status,
                            geo_line=self._geo_line(k, p))

        self.n_unique_cells = len(cell_prompt)
        dists = self._poll(cell_prompt)

        rows = []
        for k, pset in self.last_personas.items():
            agg = {p: 0.0 for p in ENGINE_PARTIES}; W = 0.0
            for persona in pset.personas:
                for status, share in self._status_shares(k, persona.demographics["aldersgruppe"]):
                    d = dists.get(self._cellkey(k, persona, status))
                    if d is None:
                        continue
                    demo = persona.demographics
                    w = (persona.weight * share
                         * turnout_weight(tlut, demo["kjonn"], demo["aldersgruppe"],
                                          demo["utdanning"]))
                    for party in ENGINE_PARTIES:
                        agg[party] += w * d[party] / 100.0
                    W += w
            if W <= 0:
                continue
            for party in ENGINE_PARTIES:
                rows.append({"kommune": k, "party": party, "pred_pct": agg[party] / W * 100.0})
        return pd.DataFrame(rows)

    def _poll(self, cell_prompt: dict[tuple, str]) -> dict[tuple, dict]:
        items = list(cell_prompt.items())
        paths = {ck: self.cache_dir / f"{hashlib.sha1((self.model + chr(10) + pr).encode()).hexdigest()[:20]}.json"
                 for ck, pr in items}
        todo = [(ck, pr, paths[ck]) for ck, pr in items if not paths[ck].exists()]
        if todo:
            logger.info("Valg-poll [%s]: %d unike kontekst-celler (%d cachet).",
                        self.variant.key, len(todo), len(items) - len(todo))

            def run(job):
                ck, pr, path = job
                try:
                    text = self.backend.complete(pr, model=self.model)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Celle-kall feilet permanent, hopper over (dekningstap): %s", e)
                    return
                path.write_text(json.dumps({"text": text}, ensure_ascii=False), "utf-8")

            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                list(ex.map(run, todo))
        out = {}
        missing = 0
        for ck, _ in items:
            if not paths[ck].exists():           # feilet permanent, hoppet over
                missing += 1
                continue
            parsed = _parse_party(json.loads(paths[ck].read_text("utf-8"))["text"])
            if parsed is not None:
                out[ck] = parsed
        self.n_failed_cells = missing
        if missing:
            logger.warning("Valg-poll [%s]: %d/%d celler uten svar (dekningstap) — "
                           "kjør på nytt for å fylle hullene.",
                           self.variant.key, missing, len(items))
        return out
