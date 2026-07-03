"""SSB API v0 client for SimNorge — Modul 1 (datalag).

Henter tabeller fra Statistisk sentralbyrå sitt åpne API
(`https://data.ssb.no/api/v0/no/table/{id}`) og returnerer rene, tidy
DataFrames. Rådata caches til disk som parquet, slik at hele datalaget er
reproduserbart og refetch er valgfritt.

Designvalg (forankret i PLAN.md):
- Metadata hentes med GET, datauttrekk med POST (json-stat2).
- Celleprikking (SSB anonymiserer celler < 3 personer) håndteres EKSPLISITT:
  prikkede/manglende celler logges og markeres med kolonnen ``missing`` —
  aldri stille imputering.
- Alt rådata caches; samme spørring gir samme fil. Fast og deterministisk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import requests

logger = logging.getLogger("simnorge.ssb")

BASE_URL = "https://data.ssb.no/api/v0/no/table"

# SSB-API-et er rate-begrenset (~30 spørringer/60 s). 429 og transiente 5xx
# håndteres med venting + nytt forsøk i stedet for å feile — nødvendige for
# hele-landet-kjøringer (355 kommuner × flere tabeller).
RETRYABLE_STATUS = {429, 502, 503, 504}
MAX_RETRIES = 6
DEFAULT_RETRY_WAIT = 25.0  # sekunder, når Retry-After mangler

# Statussymboler SSB bruker i json-stat2 for celler uten ordinær verdi.
# Kilde: SSB / PxWeb. ".." og ":" dekker prikking/konfidensialitet.
SUPPRESSION_STATUS = {
    ":": "konfidensielt/prikket (celle < 3 personer)",
    ".": "ikke mulig å oppgi",
    "..": "oppgave mangler",
    "...": "tall finnes ikke",
    "-": "null (eksakt)",
}


@dataclass
class SuppressionReport:
    """Oppsummering av prikkede/manglende celler i ett uttrekk."""

    table_id: str
    total_cells: int
    missing_cells: int
    status_counts: dict[str, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)

    @property
    def missing_fraction(self) -> float:
        return self.missing_cells / self.total_cells if self.total_cells else 0.0

    def log(self) -> None:
        if self.missing_cells == 0:
            logger.info("Tabell %s: ingen prikkede/manglende celler (%d celler).",
                        self.table_id, self.total_cells)
            return
        logger.warning(
            "Tabell %s: %d/%d celler (%.1f%%) er prikket/mangler. Status: %s",
            self.table_id, self.missing_cells, self.total_cells,
            100 * self.missing_fraction, self.status_counts,
        )
        for ex in self.examples[:5]:
            logger.warning("   prikket eksempel: %s", ex)


class SSBClient:
    """Tynn, cachet klient mot SSB-tabell-API-et."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/ssb",
        lang: str = "no",
        timeout: int = 60,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lang = lang
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "SimNorge/0.1 (SSB open data)")
        # Siste prikkerapport per tabell, tilgjengelig for inspeksjon/test.
        self.last_report: dict[str, SuppressionReport] = {}

    # ------------------------------------------------------------------ #
    # Metadata (GET, cachet som json)                                    #
    # ------------------------------------------------------------------ #
    def metadata(self, table_id: str, *, force: bool = False) -> dict:
        """Hent og cache tabellmetadata (dimensjoner og gyldige koder)."""
        path = self.cache_dir / f"{table_id}.meta.json"
        if path.exists() and not force:
            return json.loads(path.read_text(encoding="utf-8"))
        url = f"{BASE_URL}/{table_id}"
        logger.info("GET metadata %s", url)
        resp = self._request("get", url)
        resp.raise_for_status()
        meta = resp.json()
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def variables(self, table_id: str) -> list[dict]:
        return self.metadata(table_id)["variables"]

    def variable_codes(self, table_id: str) -> dict[str, list[str]]:
        return {v["code"]: list(v["values"]) for v in self.variables(table_id)}

    # ------------------------------------------------------------------ #
    # Datauttrekk (POST json-stat2, cachet som parquet)                  #
    # ------------------------------------------------------------------ #
    def query(self, table_id: str, query: dict, *, force: bool = False) -> pd.DataFrame:
        """Kjør en rå json-stat2-spørring og returner tidy DataFrame.

        ``query`` er PxWeb-spørringsobjektet (uten ``response``-feltet;
        det settes til json-stat2 her).
        """
        body = {"query": query.get("query", query), "response": {"format": "json-stat2"}}
        key = self._cache_key(table_id, body)
        path = self.cache_dir / f"{table_id}.{key}.parquet"
        if path.exists() and not force:
            df = pd.read_parquet(path)
            # Rekonstruer prikkerapport fra cachet data så logging er konsistent.
            self._report_from_df(table_id, df).log()
            return df

        url = f"{BASE_URL}/{table_id}"
        logger.info("POST %s  query-dims=%s", url, [q["code"] for q in body["query"]])
        resp = self._request("post", url, json=body)
        if resp.status_code >= 400:
            raise SSBQueryError(table_id, body, resp)
        df = self._jsonstat2_to_tidy(resp.json())
        report = self._report_from_df(table_id, df)
        self.last_report[table_id] = report
        report.log()
        df.to_parquet(path, index=False)
        return df

    def fetch(
        self,
        table_id: str,
        filters: Mapping[str, Sequence[str]] | None = None,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        """Bekvemmelighetsuttrekk: bygg json-stat2-spørring fra et enkelt dict.

        Regler per dimensjon i tabellen:
        - finnes i ``filters``  -> item-filter med de oppgitte kodene
        - ellers, eliminerbar   -> utelates (kollapser til total)
        - ellers                -> velg alle verdier ("all", "*")

        Dette gjør at man bare trenger å spesifisere det man bryr seg om.
        """
        filters = dict(filters or {})
        query: list[dict] = []
        for var in self.variables(table_id):
            code = var["code"]
            if code in filters:
                values = list(filters[code])
                self._validate_codes(table_id, var, values)
                query.append({"code": code, "selection": {"filter": "item", "values": values}})
            elif var.get("elimination"):
                continue
            else:
                query.append({"code": code, "selection": {"filter": "all", "values": ["*"]}})
        return self.query(table_id, {"query": query}, force=force)

    # ------------------------------------------------------------------ #
    # json-stat2 -> tidy                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _jsonstat2_to_tidy(js: dict) -> pd.DataFrame:
        """Konverter et json-stat2-datasett til en lang/tidy DataFrame.

        Én rad per celle. Kolonner: <dim>=kode og <dim>_label=tekst for hver
        dimensjon, pluss ``value`` (float, NaN ved prikking) og ``status``
        (statussymbol eller None) og ``missing`` (bool).
        """
        dim_ids: list[str] = js["id"]
        sizes: list[int] = js["size"]
        dims = js["dimension"]

        # Ordnet kodeliste + etikettkart per dimensjon.
        ordered_codes: dict[str, list[str]] = {}
        labels: dict[str, dict[str, str]] = {}
        for d in dim_ids:
            cat = dims[d]["category"]
            index = cat["index"]
            if isinstance(index, list):
                codes = list(index)
            else:  # dict kode->posisjon
                codes = sorted(index, key=lambda k: index[k])
            ordered_codes[d] = codes
            labels[d] = cat.get("label", {})

        values = js["value"]
        status = js.get("status", {})
        n = 1
        for s in sizes:
            n *= s

        # Hjelp til å hente verdi/status enten de er liste eller sparse dict.
        def get_value(flat: int):
            if isinstance(values, dict):
                return values.get(str(flat))
            return values[flat]

        def get_status(flat: int):
            if isinstance(status, dict):
                return status.get(str(flat))
            if isinstance(status, list):
                return status[flat] if flat < len(status) else None
            return None

        # Row-major iterasjon: siste dimensjon varierer raskest.
        # Forhåndsberegn "strides" så vi kan dekode flat-indeks til koordinater.
        strides = [1] * len(sizes)
        for i in range(len(sizes) - 2, -1, -1):
            strides[i] = strides[i + 1] * sizes[i + 1]

        rows = []
        for flat in range(n):
            row: dict = {}
            rem = flat
            for i, d in enumerate(dim_ids):
                pos = rem // strides[i]
                rem = rem % strides[i]
                code = ordered_codes[d][pos]
                row[d] = code
                row[f"{d}_label"] = labels[d].get(code, code)
            val = get_value(flat)
            st = get_status(flat)
            row["value"] = pd.NA if val is None else val
            row["status"] = st
            row["missing"] = val is None
            rows.append(row)

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    # ------------------------------------------------------------------ #
    # Prikking / manglende verdier                                       #
    # ------------------------------------------------------------------ #
    def _report_from_df(self, table_id: str, df: pd.DataFrame) -> SuppressionReport:
        missing_mask = df["missing"].astype(bool) if "missing" in df else df["value"].isna()
        missing = df[missing_mask]
        status_counts: dict[str, int] = {}
        if "status" in df.columns:
            for st, cnt in missing["status"].fillna("(ingen)").value_counts().items():
                desc = SUPPRESSION_STATUS.get(st, st)
                status_counts[f"{st} ({desc})"] = int(cnt)
        dim_cols = [c for c in df.columns
                    if c not in ("value", "status", "missing") and not c.endswith("_label")]
        examples = missing[dim_cols].head(5).to_dict("records") if not missing.empty else []
        report = SuppressionReport(
            table_id=table_id,
            total_cells=len(df),
            missing_cells=int(missing_mask.sum()),
            status_counts=status_counts,
            examples=examples,
        )
        self.last_report[table_id] = report
        return report

    # ------------------------------------------------------------------ #
    # Internt                                                            #
    # ------------------------------------------------------------------ #
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """HTTP med retry på rate-limit (429) og transiente 5xx."""
        resp = None
        for attempt in range(MAX_RETRIES):
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if resp.status_code not in RETRYABLE_STATUS:
                return resp
            wait = DEFAULT_RETRY_WAIT
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(1.0, float(retry_after))
                except ValueError:
                    pass
            logger.info("SSB %d på %s — venter %.0f s (forsøk %d/%d).",
                        resp.status_code, url, wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
        return resp

    @staticmethod
    def _validate_codes(table_id: str, var: dict, values: Sequence[str]) -> None:
        valid = set(var["values"])
        unknown = [v for v in values if v not in valid]
        if unknown:
            raise ValueError(
                f"Tabell {table_id}, dimensjon {var['code']!r}: ukjente koder "
                f"{unknown}. Gyldige eksempler: {var['values'][:8]}"
            )

    @staticmethod
    def _cache_key(table_id: str, body: dict) -> str:
        blob = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class SSBQueryError(RuntimeError):
    def __init__(self, table_id: str, body: dict, resp: requests.Response) -> None:
        self.table_id = table_id
        self.body = body
        self.status_code = resp.status_code
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        super().__init__(
            f"SSB-spørring mot tabell {table_id} feilet ({resp.status_code}): {detail}"
        )
