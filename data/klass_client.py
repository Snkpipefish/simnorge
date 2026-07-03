"""Klass-klient (SSB klassifikasjons-API) for kommuneinndeling — Modul 1.

Klassifikasjon 131 = "Standard for kommuneinndeling". Vi bruker den til å
holde styr på at kommunekoder endrer seg mellom årganger (2018- og 2020-
reformene la til nye fylkesprefikser og slo sammen kommuner; 2024 reverserte
mange av sammenslåingene og fylkesinndelingen). Uten dette ville valgdata fra
2017/2021/2023 og strukturdata på ulike årganger blitt koblet på koder som ikke
peker på samme geografi — en stille feiljustering som "ser riktig ut".

Klienten er bevisst lik SSBClient: rene svar, alt rådata cachet til disk
(json) under ``.cache/klass/`` for reproduserbarhet.

Sentrale endepunkter:
- ``/classifications/131``            -> versjoner (årganger)
- ``/classifications/131/codesAt``    -> gyldige koder på en gitt dato
- ``/classifications/131/changes``    -> kodeendringer i et datointervall

NB om ``changes``: ``to``-datoen er EKSKLUSIV for endringer som inntreffer
nøyaktig på den datoen. En reform som trer i kraft 1. januar år R fanges først
når ``to`` ligger senere på året (vi bruker ``{R}-12-31``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("simnorge.klass")

KLASS_BASE = "https://data.ssb.no/api/klass/v1"
MUNICIPALITY_CLASSIFICATION = "131"


class KlassClient:
    """Tynn, cachet klient mot Klass-API-et for kommuneinndeling."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/klass",
        classification_id: str = MUNICIPALITY_CLASSIFICATION,
        timeout: int = 60,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.classification_id = classification_id
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "SimNorge/0.1 (SSB Klass open data)",
        })

    # ------------------------------------------------------------------ #
    # Lavnivå GET med json-diskcache                                     #
    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: dict | None = None, *, force: bool = False) -> dict:
        params = params or {}
        key = hashlib.sha1(
            json.dumps({"p": path, "q": params}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache = self.cache_dir / f"{path.replace('/', '_')}.{key}.json"
        if cache.exists() and not force:
            return json.loads(cache.read_text(encoding="utf-8"))
        url = f"{KLASS_BASE}/{path}"
        logger.info("GET %s params=%s", url, params)
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    # ------------------------------------------------------------------ #
    # Versjoner / årganger                                               #
    # ------------------------------------------------------------------ #
    def versions(self, *, force: bool = False) -> list[dict]:
        data = self._get(f"classifications/{self.classification_id}", force=force)
        return data.get("versions", [])

    def current_division_year(self) -> str:
        """Året for den nyeste versjonen som er gyldig i dag (kalender-nåtid).

        NB: SSBs datatabeller ligger ofte én årgang bak denne. Bruk den som
        referanseår bare hvis du vet at dataene dine også er på den årgangen.
        """
        import datetime
        today = datetime.date.today().isoformat()
        valid = [v for v in self.versions()
                 if v["validFrom"] <= today and (v.get("validTo") is None or v["validTo"] > today)]
        if not valid:
            valid = sorted(self.versions(), key=lambda v: v["validFrom"])[-1:]
        return valid[0]["validFrom"][:4]

    # ------------------------------------------------------------------ #
    # Gyldige koder på en dato                                           #
    # ------------------------------------------------------------------ #
    def codes_at(self, date: str, *, force: bool = False) -> pd.DataFrame:
        """Gyldige kommunekoder + navn på en gitt dato (YYYY-MM-DD).

        Aksepterer også et årstall 'YYYY' (tolkes som midt i året, 1. juli,
        for å unngå reform-grenser).
        """
        date = _as_date(date)
        data = self._get(f"classifications/{self.classification_id}/codesAt",
                         {"date": date}, force=force)
        rows = [{"code": c["code"], "name": c["name"]} for c in data.get("codes", [])]
        return pd.DataFrame(rows).sort_values("code", ignore_index=True)

    def code_set(self, date: str) -> set[str]:
        return set(self.codes_at(date)["code"])

    # ------------------------------------------------------------------ #
    # Korrespondansetabeller (f.eks. sentralitet ↔ kommune)              #
    # ------------------------------------------------------------------ #
    def correspondence(self, table_id: str, *, force: bool = False) -> pd.DataFrame:
        """Hent en korrespondansetabell. Kolonner: source_code, source_name,
        target_code, target_name."""
        data = self._get(f"correspondencetables/{table_id}", force=force)
        rows = [{"source_code": m.get("sourceCode"), "source_name": m.get("sourceName"),
                 "target_code": m.get("targetCode"), "target_name": m.get("targetName")}
                for m in data.get("correspondenceMaps", [])]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Kodeendringer i et intervall                                       #
    # ------------------------------------------------------------------ #
    def code_changes(self, from_date: str, to_date: str, *, force: bool = False) -> pd.DataFrame:
        """Kodeendringer mellom to datoer.

        Returnerer kolonner: old_code, old_name, new_code, new_name, occurred.
        Rene navneendringer (old_code == new_code) er beholdt her; normaliserings-
        laget filtrerer dem bort siden de ikke krever omkoding.
        """
        from_date, to_date = _as_date(from_date), _as_date(to_date)
        data = self._get(f"classifications/{self.classification_id}/changes",
                         {"from": from_date, "to": to_date}, force=force)
        rows = [{
            "old_code": c["oldCode"], "old_name": c["oldName"],
            "new_code": c["newCode"], "new_name": c["newName"],
            "occurred": c["changeOccurred"],
        } for c in data.get("codeChanges", [])]
        df = pd.DataFrame(rows, columns=["old_code", "old_name", "new_code", "new_name", "occurred"])
        return df.sort_values("occurred", ignore_index=True)


def _as_date(date: str) -> str:
    """Tolk '2021' som '2021-07-01'; la fulle datoer stå."""
    date = str(date)
    if len(date) == 4 and date.isdigit():
        return f"{date}-07-01"
    return date
