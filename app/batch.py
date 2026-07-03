"""Nasjonal batch-poll + kommune-choropleth — Modul 8 (kartet fra PLAN).

Kart er en BATCH-jobb, ikke et klikk: ett spørsmål polles én gang per unik
celle (demografi × disposisjonsnivå × geo-linje, ~39 geo-linjer × 24 ≈ 900
celler for hele landet), og hver kommunes verdi beregnes som

    snitt_k = Σ_{alder,kjønn} w_k(alder,kjønn) · Σ_nivå P(nivå|alder,kjønn)
              · posisjon(alder, kjønn, nivå, geo-linje_k)

— altså prior-LUT × kommunens kjønn/alder-miks × cellens LLM-svar. Ingen
populasjonsekspansjon; marginalene er diskcachet. Prompter er identiske med
spørretabens (samme build_prompt + geo-linjer), så cellene DELES med
interaktive spørringer.

Resultat: {slug}.json (per-kommune-verdier + metadata) og {slug}.html
(plotly-choropleth med ærlighetsbanner) i app/static/kart/.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import SSBClient, KlassClient
from population.marginals import build_marginals
from population.schema import AGE_LABELS, SEX_LABELS
from engine import GeoContext, poll
from engine.backend import LLMBackend
from engine.engine import DEFAULT_MODEL
from .qspec import interpret_question
from .service import GEO_VALIDITY

logger = logging.getLogger("simnorge.kart")

KART_DIR = Path(__file__).parent / "static" / "kart"
GEOJSON_PATH = Path(".cache/geo/kommuner.geojson")
LEVELS = ["lav", "middels", "hoy"]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9æøå]+", "-", text.lower()).strip("-")
    return s[:70] or "kart"


@dataclass
class NationalMapResult:
    question: str
    slug: str
    values: pd.DataFrame          # kommune, navn, mean, n_pop
    domain: str
    axis: str
    n_cells: int
    n_calls: int
    n_failed_cells: int
    year: str


def national_poll(
    question: str,
    ssb: SSBClient | None = None,
    klass: KlassClient | None = None,
    *,
    backend: LLMBackend | None = None,
    model: str = DEFAULT_MODEL,
    pop_year: str = "2024",
    cache_root: str = ".cache",
    max_workers: int = 6,
) -> NationalMapResult:
    from calibration.election_data import municipality_code_set
    from enrich import load_attitude_items

    ssb = ssb or SSBClient()
    klass = klass or KlassClient()
    iq = interpret_question(question, backend=backend, model=model,
                            cache_dir=f"{cache_root}/qspec")
    items = load_attitude_items(ssb)
    item = next(i for i in items if i.column == iq.spec.focus_item)

    geo = GeoContext(ssb, klass)
    navn_df = klass.codes_at(pop_year)
    navn = dict(zip(navn_df["code"], navn_df["name"]))

    # Kommunens kjønn/alder-miks + geo-linje (alt diskcachet i Modul 1/2).
    kommuner = sorted(municipality_code_set(klass, pop_year))
    mix, line_of, n_pop = {}, {}, {}
    for k in kommuner:
        km = build_marginals(ssb, k, year=pop_year)
        if km.skipped or km.n_total <= 0:
            continue
        mix[k] = (km.sex_age / km.sex_age.sum()).T      # (4 alder, 2 kjønn)
        line_of[k] = geo.line_for(k, km.median_income, km.lowincome_share)
        n_pop[k] = km.n_total

    # Én instans per (geo-linje × alder × kjønn × nivå) — kommunefeltet er en
    # representant-kommune for linjen (kun brukt til geo_context-oppslag).
    repr_of = {}
    for k, line in line_of.items():
        repr_of.setdefault(line, k)
    rows, geo_context = [], {}
    for line, rk in repr_of.items():
        geo_context[rk] = line
        for a in AGE_LABELS:
            for s in SEX_LABELS:
                for lvl in LEVELS:
                    rows.append({"kommune": rk, "kjonn": s, "aldersgruppe": a,
                                 "level": lvl, "weight": 1.0, "geo_line": line})
    inst = pd.DataFrame(rows)
    logger.warning("Nasjonal poll: %d celler (%d geo-linjer × %d demografi×nivå).",
                   len(inst), len(repr_of), len(AGE_LABELS) * len(SEX_LABELS) * 3)

    out = poll(inst, iq.spec, geo_context=geo_context, backend=backend,
               model=model, max_workers=max_workers,
               cache_dir=f"{cache_root}/engine")

    # posisjon-LUT: (geo-linje, alder, kjønn, nivå) -> posisjon.
    res = out.results
    pos = {}
    failed_cells = 0
    for _, r in res.iterrows():
        if np.isnan(r["posisjon"]):
            failed_cells += 1
            continue
        pos[(r["geo_line"], r["aldersgruppe"], r["kjonn"], r["level"])] = float(r["posisjon"])

    # Kommuneverdier: w(a,s) × P(nivå|a,s) × posisjon.
    vals = []
    for k, w in mix.items():
        line = line_of[k]
        num = den = 0.0
        for ai, a in enumerate(AGE_LABELS):
            for si, s in enumerate(SEX_LABELS):
                probs = item.lut[ai, si]
                for li, lvl in enumerate(LEVELS):
                    p = pos.get((line, a, s, lvl))
                    if p is None or probs[li] <= 0:
                        continue
                    wgt = w[ai, si] * probs[li]
                    num += wgt * p
                    den += wgt
        if den > 0:
            vals.append({"kommune": k, "navn": navn.get(k, k),
                         "mean": num / den, "n_pop": n_pop[k]})
    values = pd.DataFrame(vals)
    return NationalMapResult(
        question=question, slug=slugify(question), values=values,
        domain=iq.domain, axis=iq.spec.focus_item, n_cells=len(inst),
        n_calls=out.n_calls, n_failed_cells=failed_cells, year=pop_year,
    )


# --------------------------------------------------------------------------- #
# Choropleth-HTML                                                              #
# --------------------------------------------------------------------------- #
def render_map(result: NationalMapResult, *,
               geojson_path: Path = GEOJSON_PATH) -> tuple[Path, Path]:
    import plotly.graph_objects as go

    KART_DIR.mkdir(parents=True, exist_ok=True)
    gj = json.loads(Path(geojson_path).read_text("utf-8"))
    v = result.values
    validity, validity_text = GEO_VALIDITY[result.domain]

    # Choroplethmap (MapLibre-fliser): robust mot GeoJSON-winding, i motsetning
    # til geo-akse-varianten som fylte hele flaten.
    fig = go.Figure(go.Choroplethmap(
        geojson=gj, featureidkey="properties.kommunenummer",
        locations=v["kommune"], z=v["mean"].round(2),
        colorscale="RdYlBu",
        marker_line_width=0.3, marker_line_color="#888", marker_opacity=0.85,
        colorbar_title="snitt (0–10)",
        customdata=np.stack([v["navn"], v["n_pop"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>snitt %{z:.2f}"
                      "<br>~%{customdata[1]:,} personer 16+<extra></extra>",
    ))
    spread = float(v["mean"].max() - v["mean"].min())
    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=65.0, lon=15.0), zoom=3.6),
        title=dict(text=(f"{result.question}<br><sup>Ekstrapolert, ikke målt — "
                         f"{validity_text} Kommunespenn: {spread:.2f} skalapoeng."
                         f"</sup>"), x=0.02),
        margin=dict(l=0, r=0, t=90, b=0), height=760,
    )
    html_path = KART_DIR / f"{result.slug}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

    meta = {
        "sporsmal": result.question, "slug": result.slug, "akse": result.axis,
        "domene": result.domain, "geo_validitet": validity,
        "geo_validitet_tekst": validity_text, "aar": result.year,
        "celler": result.n_cells, "nye_kall": result.n_calls,
        "feilede_celler": result.n_failed_cells,
        "kommunespenn": round(spread, 3),
        "verdier": {r["kommune"]: round(r["mean"], 3) for _, r in v.iterrows()},
    }
    json_path = KART_DIR / f"{result.slug}.json"
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return html_path, json_path


def list_maps() -> list[dict]:
    """Tilgjengelige kart (for spørretabens kart-seksjon)."""
    out = []
    if KART_DIR.exists():
        for p in sorted(KART_DIR.glob("*.json")):
            m = json.loads(p.read_text("utf-8"))
            out.append({"sporsmal": m["sporsmal"], "slug": m["slug"],
                        "geo_validitet": m["geo_validitet"],
                        "kommunespenn": m.get("kommunespenn"),
                        "feilede_celler": m.get("feilede_celler", 0)})
    return out
