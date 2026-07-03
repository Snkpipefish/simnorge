"""Spørretjenesten — Modul 8 (orkestrerer Modul 1-6 bak ett kall).

ask(spørsmål, geografi) -> strukturert resultat med:
- vektet aggregat + spredning + fordeling over skalaen (med bootstrap-bånd),
- oppdeling per disposisjonsnivå, aldersgruppe og kjønn,
- ÆRLIGHETSMETADATA som UI-et er forpliktet til å vise: at kommunale svar er
  ekstrapolert (ikke målt), hvilken SSB-akse prioren kom fra, dekningstap, og
  om geografisk differensiering er validert for spørsmålets domene
  (politikk: ja via valg-backtest; livskvalitet: målt ~flat, SSB 13798/13799).

Kostnadsdisiplin: granularitet 'alder_kjonn' som standard (24 unike prompter
per geografi — interaktivt), 'full' for batch. Alt LLM-cachet på disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from data import SSBClient, KlassClient
from population import build_population
from population.marginals import build_marginals
from enrich import load_attitude_items, enrich_population
from personas import build_personas
from engine import GeoContext, expand_by_item, poll
from engine.backend import LLMBackend
from engine.engine import DEFAULT_MODEL
from aggregate.rollup import bootstrap_distribution
from .qspec import InterpretedQuestion, interpret_question

logger = logging.getLogger("simnorge.app")

NATIONAL_N = 120_000
POSITION_BINS = [(0, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

GEO_VALIDITY = {
    "politikk": ("validert", "Geografisk differensiering er validert mot faktiske "
                             "valgresultat per kommune (backtest 2021→2025)."),
    "livskvalitet": ("delvis", "SSB har målt livskvalitet per fylke/sentralitet "
                               "(13798/13799): variasjonen er liten, og modellen "
                               "(demografi + arbeidsmarkedsstatus) fanger bare en "
                               "del av den (korrelasjon ~0.4 for andel med lav "
                               "skår). Forvent svar nær landssnittet; geografiske "
                               "utslag er svakt, ikke fullt, validert."),
    "annet": ("uvalidert", "For dette temaet finnes ingen geografisk fasit. Svaret er "
                           "ekstrapolert fra nasjonale demografi-koblinger og kan ikke "
                           "valideres geografisk."),
}


@dataclass
class AskResult:
    question: str
    geography: str
    geography_label: str
    interpreted: InterpretedQuestion
    mean: float
    std: float
    n_represented: float
    distribution: dict            # bin -> {andel, lo, hi}
    by_level: dict[str, float]
    by_age: dict[str, float]
    by_sex: dict[str, float]
    n_prompts: int
    n_calls: int
    n_failed: int
    certainty_mean: float = float("nan")   # LLM-ens selvrapporterte sikkerhet (vektet)
    low_certainty_share: float = 0.0       # andel av vekten med sikkerhet < 0.5
    honesty: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sporsmal": self.question,
            "geografi": {"kode": self.geography, "navn": self.geography_label},
            "tolkning": {"akse": self.interpreted.spec.focus_item,
                         "begrunnelse": self.interpreted.axis_reason,
                         "domene": self.interpreted.domain,
                         "skala": {"lav": self.interpreted.spec.scale_low_label,
                                   "hoy": self.interpreted.spec.scale_high_label}},
            "resultat": {"snitt": round(self.mean, 2), "spredning": round(self.std, 2),
                         "representerer_personer": int(self.n_represented),
                         "fordeling": self.distribution,
                         "per_nivaa": {k: round(v, 2) for k, v in self.by_level.items()},
                         "per_alder": {k: round(v, 2) for k, v in self.by_age.items()},
                         "per_kjonn": {k: round(v, 2) for k, v in self.by_sex.items()}},
            "dekning": {"unike_prompter": self.n_prompts, "nye_kall": self.n_calls,
                        "feilede": self.n_failed},
            "llm_sikkerhet": {"snitt": round(self.certainty_mean, 2),
                              "andel_lav": round(self.low_certainty_share, 2)},
            "aerlighet": self.honesty,
        }


class SimNorgeService:
    def __init__(self, ssb: SSBClient | None = None, klass: KlassClient | None = None,
                 *, backend: LLMBackend | None = None, model: str = DEFAULT_MODEL,
                 pop_year: str = "2024", cache_root: str = ".cache"):
        # cache_root MÅ pekes et annet sted i tester med mock-backend —
        # mock-svar skal aldri forurense den ekte LLM-cachen.
        self.ssb = ssb or SSBClient()
        self.klass = klass or KlassClient()
        self.backend = backend
        self.model = model
        self.pop_year = pop_year
        self.cache_root = cache_root
        self.items = load_attitude_items(self.ssb)
        self.geo = GeoContext(self.ssb, self.klass)
        navn = self.klass.codes_at(pop_year)
        self._navn = dict(zip(navn["code"], navn["name"]))
        self._psets: dict[tuple[str, str], object] = {}

    # ------------------------------------------------------------------ #
    def kommuner(self) -> list[dict]:
        return [{"kode": k, "navn": v} for k, v in sorted(self._navn.items())
                if len(k) == 4]

    def _pset(self, geography: str, granularity: str):
        key = (geography, granularity)
        if key not in self._psets:
            if geography == "0":
                pop = build_population(self.ssb, "0", target_n=NATIONAL_N)
            else:
                pop = build_population(self.ssb, geography, year=self.pop_year)
            if pop.skipped or pop.n == 0:
                raise ValueError(f"Kommune {geography}: ingen populasjon ({'; '.join(pop.issues)})")
            enr = enrich_population(pop.persons, self.items, seed=42)
            self._psets[key] = build_personas(enr, self.items, granularity=granularity)
        return self._psets[key]

    def _geo_context(self, geography: str) -> dict[str, str | None]:
        if geography == "0":
            return {"0": None}
        km = build_marginals(self.ssb, geography, year=self.pop_year)
        return {geography: self.geo.line_for(geography, km.median_income,
                                             km.lowincome_share)}

    # ------------------------------------------------------------------ #
    def ask(self, question: str, geography: str = "0", *,
            granularity: str = "alder_kjonn", max_workers: int = 6) -> AskResult:
        iq = interpret_question(question, backend=self.backend, model=self.model,
                                cache_dir=f"{self.cache_root}/qspec")
        if iq.spec.focus_item not in {i.column for i in self.items}:
            raise ValueError(f"Aksen {iq.spec.focus_item} finnes ikke i holdningslaget.")

        pset = self._pset(geography, granularity)
        inst = expand_by_item(pset, iq.spec.focus_item)
        out = poll(inst, iq.spec, geo_context=self._geo_context(geography),
                   backend=self.backend, model=self.model, max_workers=max_workers,
                   cache_dir=f"{self.cache_root}/engine")
        res = out.results[out.results["posisjon"].notna()]
        if res.empty:
            raise RuntimeError("Ingen gyldige svar fra motoren (fullt dekningstap).")

        w = res["weight"].to_numpy()
        x = res["posisjon"].to_numpy()
        mean = float(np.average(x, weights=w))
        std = float(np.sqrt(np.average((x - mean) ** 2, weights=w)))

        # LLM-ens selvrapporterte sikkerhet RAPPORTERES men vektes aldri inn:
        # selvrapportert sikkerhet er ukalibrert, og å nedvekte lav-sikkerhet-
        # celler ville stille skjevfordele representativiteten (cellene med
        # usikker LLM er fortsatt like mange virkelige mennesker).
        cert = res["sikkerhet"].to_numpy()
        certainty_mean = float(np.average(cert, weights=w))
        low_certainty_share = float(w[cert < 0.5].sum() / w.sum())

        # Fordeling over skalabins med bootstrap-bånd (instans = enhet).
        cats = [f"{lo}-{hi}" for lo, hi in POSITION_BINS]
        onehot = np.zeros((len(res), len(POSITION_BINS)))
        for j, (lo, hi) in enumerate(POSITION_BINS):
            onehot[:, j] = ((x >= lo) & (x <= hi)).astype(float)
        dist = bootstrap_distribution(onehot, w, cats)

        def group_means(col: str) -> dict[str, float]:
            return {str(g): float(np.average(s["posisjon"], weights=s["weight"]))
                    for g, s in res.groupby(col, observed=True) if s["weight"].sum() > 0}

        validity, validity_text = GEO_VALIDITY[iq.domain]
        honesty = {
            "ekstrapolert": ("Kommunale svar er EKSTRAPOLERT fra nasjonale "
                             "demografi-holdningskoblinger (SSB) — ikke målt lokalt."),
            "prior_akse": (f"Personenes utgangspunkt er SSB-aksen "
                           f"'{iq.spec.focus_item}'; LLM-en fyller gapet til "
                           f"spørsmålet derfra."),
            "geo_validitet": validity,
            "geo_validitet_tekst": validity_text,
        }
        if out.n_failed:
            honesty["dekningstap"] = (f"{out.n_failed} av {len(out.results)} instanser "
                                      f"uten gyldig svar — tolkes som tapt dekning.")

        label = "Hele Norge" if geography == "0" else self._navn.get(geography, geography)
        return AskResult(
            question=question, geography=geography, geography_label=label,
            interpreted=iq, mean=mean, std=std, n_represented=float(w.sum()),
            distribution=dist.as_dict(),
            by_level=group_means("level"), by_age=group_means("aldersgruppe"),
            by_sex=group_means("kjonn"),
            n_prompts=len(inst),
            n_calls=out.n_calls, n_failed=out.n_failed,
            certainty_mean=certainty_mean, low_certainty_share=low_certainty_share,
            honesty=honesty,
        )
