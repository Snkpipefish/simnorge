"""Normalisering av kommunekoder til ett felles referanseår — Modul 1.

Bygger en korrespondanse fra en kildeårgangs kommunekoder til et referanseårs
koder, ved å løse opp kjeder av kodeendringer fra Klass (transitivt — én kode
kan endres i flere steg, f.eks. Klæbu 1662 → 5030 (2018) → 5001 Trondheim
(2020)).

Eksplisitt informasjonstap (kjernen i hvorfor dette finnes):
- SAMMENSLÅING (mange kilde → én referanse): trygt. Vi kan aggregere fasit
  (summere stemmer/personer) opp til den sammenslåtte kommunen.
- SPLITT (én kilde → flere referanser): IKKE trygt. Vi kan ikke fordele en
  gammel kommunes stemmer presist på de nye delene. Slike tilfeller LOGGES og
  holdes utenfor den rene koblingsnøkkelen — vi gjetter ikke.

Standard referanseår er 2024 — årgangen SSBs gjeldende datatabeller leverer på
(2025-valget og strukturtabellene bruker 2024-koder), og som den verifiserte
fasiten i Modul 1 hviler på. Kan overstyres når SSB ruller fram.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from .klass_client import KlassClient

logger = logging.getLogger("simnorge.kommune")

DEFAULT_REFERENCE_YEAR = "2024"


# --------------------------------------------------------------------------- #
# Korrespondanse-objekt                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Correspondence:
    """Kobling fra kildeårgang til referanseårgang for kommunekoder."""

    source_year: str
    reference_year: str
    # Ren, entydig kobling kilde -> referanse (1:1, navneendring, eller del av
    # en sammenslåing). Trygg å bruke som koblingsnøkkel.
    mapping: dict[str, str] = field(default_factory=dict)
    # Referansekode -> liste av kildekoder som slås sammen i den (|kilder| > 1).
    merges: dict[str, list[str]] = field(default_factory=dict)
    # Kildekode -> liste av referansekoder den splittes til. Tvetydig/tap:
    # holdt UTENFOR ``mapping``. Aldri gjettet fordelt.
    splits: dict[str, list[str]] = field(default_factory=dict)
    # Kildekoder som går uendret gjennom (identitet).
    unchanged: set[str] = field(default_factory=set)

    def normalize(self, code: str) -> str | None:
        """Referansekode for én kildekode, eller None hvis den ikke kan kobles
        entydig (splitt eller ukjent). Splitter logges av kalleren."""
        return self.mapping.get(code)

    def is_split(self, code: str) -> bool:
        return code in self.splits

    def log_summary(self) -> None:
        logger.info(
            "Kommunekorrespondanse %s -> %s: %d entydige (%d uendret, %d sammenslåtte "
            "referansekoder), %d splittede kildekoder (utelatt).",
            self.source_year, self.reference_year, len(self.mapping),
            len(self.unchanged), len(self.merges), len(self.splits),
        )
        for ref, srcs in self.merges.items():
            logger.info("   sammenslåing: %s -> %s", sorted(srcs), ref)
        for src, refs in self.splits.items():
            logger.warning(
                "   SPLITT (kan ikke fordeles presist): %s -> %s — utelatt fra "
                "koblingsnøkkel, logget.", src, sorted(refs))


# --------------------------------------------------------------------------- #
# Bygg korrespondanse                                                         #
# --------------------------------------------------------------------------- #
def build_correspondence(
    client: KlassClient,
    source_year: str | int,
    reference_year: str | int = DEFAULT_REFERENCE_YEAR,
) -> Correspondence:
    """Bygg en :class:`Correspondence` fra ``source_year`` til ``reference_year``.

    Bruker Klass ``changes`` over intervallet [source-01-01, reference-12-31].
    ``to`` settes til 31. desember slik at en reform som trer i kraft 1. januar
    i referanseåret fanges (changes-endepunktet er ekskluderende på ``to``).
    """
    source_year, reference_year = str(source_year), str(reference_year)
    if reference_year < source_year:
        raise ValueError(
            f"reference_year ({reference_year}) må være >= source_year ({source_year}); "
            "vi normaliserer eldre data fram til gjeldende inndeling, ikke bakover."
        )

    source_codes = client.code_set(source_year)
    reference_codes = client.code_set(reference_year)

    # Identitet: samme årgang -> ingen omkoding, ingen changes å hente. (Brukes
    # bl.a. når et valg er kodet på referanseårgangen, f.eks. 2025-valget på
    # 2024-koder.)
    if source_year == reference_year:
        corr = Correspondence(source_year=source_year, reference_year=reference_year)
        corr.mapping = {c: c for c in source_codes}
        corr.unchanged = set(source_codes)
        corr.log_summary()
        return corr

    changes = client.code_changes(f"{source_year}-01-01", f"{reference_year}-12-31")
    # Filtrer bort rene navneendringer (samme kode) — de krever ingen omkoding.
    real = changes[changes["old_code"] != changes["new_code"]]

    # Bygg adjacency old -> [new ...] (kronologisk allerede sortert).
    adj: dict[str, list[str]] = {}
    for old, new in zip(real["old_code"], real["new_code"]):
        adj.setdefault(old, [])
        if new not in adj[old]:
            adj[old].append(new)

    @lru_cache(maxsize=None)
    def terminals(code: str, _path: frozenset = frozenset()) -> frozenset:
        """Endepunkt(er) for en kode: følg kjeden forover til koder uten
        utgående endringer. Syklusvern via ``_path``."""
        if code not in adj or code in _path:
            return frozenset({code})
        out: set[str] = set()
        nxt_path = _path | {code}
        for nxt in adj[code]:
            if nxt == code:
                out.add(code)
            else:
                out |= terminals(nxt, nxt_path)
        return frozenset(out or {code})

    corr = Correspondence(source_year=source_year, reference_year=reference_year)

    # Universet av kildekoder: gyldige i kildeåret + alt som faktisk endres.
    universe = set(source_codes) | set(adj)
    for src in sorted(universe):
        ends = terminals(src)
        if len(ends) == 1:
            ref = next(iter(ends))
            corr.mapping[src] = ref
            if src == ref and src not in adj:
                corr.unchanged.add(src)
        else:
            corr.splits[src] = sorted(ends)

    # Inverter for å finne sammenslåinger (referanse mottar > 1 kilde).
    inv: dict[str, list[str]] = {}
    for src, ref in corr.mapping.items():
        inv.setdefault(ref, []).append(src)
    corr.merges = {ref: sorted(srcs) for ref, srcs in inv.items() if len(srcs) > 1}

    # Sanity: advar hvis en målkode ikke finnes i referanseårgangen.
    bad_targets = {ref for ref in corr.mapping.values() if ref not in reference_codes}
    if bad_targets:
        logger.warning(
            "%d målkoder finnes ikke i referanseår %s: %s — sjekk referanseår.",
            len(bad_targets), reference_year, sorted(bad_targets)[:10])

    corr.log_summary()
    return corr


# --------------------------------------------------------------------------- #
# Normaliser en DataFrame                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class NormalizationReport:
    code_col: str
    rows_in: int
    rows_out: int
    merged_refs: dict[str, list[str]] = field(default_factory=dict)
    split_codes_dropped: dict[str, list[str]] = field(default_factory=dict)
    unmapped_codes_dropped: list[str] = field(default_factory=list)

    def log(self) -> None:
        if self.merged_refs:
            logger.info("Normalisering: aggregerte %d sammenslåtte referansekoder.",
                        len(self.merged_refs))
        for code, refs in self.split_codes_dropped.items():
            logger.warning("Normalisering: droppet splittet kildekode %s -> %s "
                           "(kan ikke fordeles presist).", code, refs)
        if self.unmapped_codes_dropped:
            logger.warning("Normalisering: droppet %d ukoblede kildekoder: %s",
                           len(self.unmapped_codes_dropped),
                           sorted(set(self.unmapped_codes_dropped))[:10])
        logger.info("Normalisering: %d rader inn -> %d rader ut.", self.rows_in, self.rows_out)


def normalize_dataframe(
    df: pd.DataFrame,
    corr: Correspondence,
    *,
    code_col: str,
    group_cols: list[str] | tuple[str, ...] = (),
    additive_cols: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, NormalizationReport]:
    """Re-nøkle ``df`` fra kildekoder til referansekoder.

    - Sammenslåtte koder aggregeres: ``additive_cols`` summeres per
      (``group_cols`` + referansekode). Bruk dette på TELLINGER (stemmer,
      personer) — ikke på prosenter (de må regnes på nytt etterpå).
    - Splittede kildekoder droppes og logges (informasjonstap, ikke gjettet).
    - Ukoblede koder (ikke gyldige i kildeåret / ikke i korrespondansen) droppes
      og logges.

    Returnerer (normalisert DataFrame, :class:`NormalizationReport`).
    """
    group_cols = list(group_cols)
    additive_cols = list(additive_cols)
    work = df.copy()

    split_mask = work[code_col].isin(corr.splits)
    work["_ref"] = work[code_col].map(corr.mapping)
    unmapped_mask = work["_ref"].isna() & ~split_mask

    report = NormalizationReport(
        code_col=code_col,
        rows_in=len(work),
        rows_out=0,
        merged_refs=dict(corr.merges),
        split_codes_dropped={c: corr.splits[c] for c in work.loc[split_mask, code_col].unique()},
        unmapped_codes_dropped=list(work.loc[unmapped_mask, code_col].unique()),
    )

    kept = work[work["_ref"].notna()].copy()
    kept[code_col] = kept["_ref"]
    kept = kept.drop(columns=["_ref"])

    keys = group_cols + [code_col]
    if additive_cols:
        out = (kept.groupby(keys, as_index=False, dropna=False)[additive_cols]
               .sum())
    else:
        out = kept[keys].drop_duplicates(ignore_index=True)

    report.rows_out = len(out)
    report.log()
    return out, report
