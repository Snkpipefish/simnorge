"""Modul 1 — verifisering av Klass-normalisering av kommunekoder.

Kjøres mot ekte Klass-API (krever nett). Kan kjøres direkte:
    .venv/bin/python -m tests.test_klass_normalize

Verifiserer mot faktiske reformer:
1. SAMMENSLÅING (2020): Klæbu (1662, 2017) og Trondheim (1601) → Trondheim
   (5001, 2024). Flerstegs kjede (1662→5030→5001) løses transitivt og
   aggregeres trygt.
2. SPLITT (2024): Ålesund (1507, 2023) → {Ålesund 1508, Haram 1580}. Kan IKKE
   fordeles presist → logget, holdt utenfor koblingsnøkkelen, aldri gjettet.
3. UENDRET: Oslo (0301) går uendret gjennom for alle kildeår.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from data import KlassClient, build_correspondence, normalize_dataframe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _client() -> KlassClient:
    return KlassClient()


def test_merge_2020_multihop() -> None:
    """2017 → 2024: Trondheim+Klæbu sammenslåing, flerstegs kjede."""
    corr = build_correspondence(_client(), source_year="2017", reference_year="2024")
    # Trondheim 1601 (2017) -> 5001 (2024), direkte i 2018.
    assert corr.mapping.get("1601") == "5001", corr.mapping.get("1601")
    # Klæbu 1662 (2017) -> 5030 (2018) -> 5001 (2020): transitivt løst.
    assert corr.mapping.get("1662") == "5001", corr.mapping.get("1662")
    # 5001 registrert som sammenslåing av minst Trondheim + Klæbu.
    assert "5001" in corr.merges
    assert {"1601", "1662"}.issubset(set(corr.merges["5001"]))
    print(f"\nSammenslåing -> 5001 Trondheim: {sorted(corr.merges['5001'])}")


def test_split_2024_is_logged_not_guessed() -> None:
    """2023 → 2024: Ålesund 1507 splittes til Ålesund 1508 + Haram 1580."""
    corr = build_correspondence(_client(), source_year="2023", reference_year="2024")
    assert corr.is_split("1507"), "1507 skulle vært registrert som splitt"
    assert corr.splits["1507"] == ["1508", "1580"], corr.splits["1507"]
    # Splitt skal IKKE gi en entydig kobling (vi gjetter ikke fordelingen).
    assert corr.normalize("1507") is None
    # Oslo upåvirket i samme korrespondanse.
    assert corr.normalize("0301") == "0301"
    print(f"\nSplitt 1507 Ålesund -> {corr.splits['1507']} (logget, utelatt)")


def test_unchanged_oslo_passthrough() -> None:
    client = _client()
    for src in ["2017", "2021", "2023"]:
        corr = build_correspondence(client, source_year=src, reference_year="2024")
        assert corr.normalize("0301") == "0301", f"Oslo endret seg for kildeår {src}"
        assert "0301" in corr.unchanged
    print("\nOslo 0301 uendret for kildeår 2017/2021/2023 ✓")


def test_dataframe_merge_aggregates_and_drops_split() -> None:
    """Re-nøkling av en tellings-DataFrame: sammenslåing summeres, splitt droppes."""
    corr = build_correspondence(_client(), source_year="2017", reference_year="2024")
    # Syntetisk fasit-lignende ramme: stemmetall per (parti, kommune).
    df = pd.DataFrame([
        # Trondheim + Klæbu skal summeres til 5001.
        {"parti": "Ap", "Region": "1601", "stemmer": 40000},
        {"parti": "Ap", "Region": "1662", "stemmer": 2000},
        # Oslo uendret.
        {"parti": "Ap", "Region": "0301", "stemmer": 90000},
    ])
    out, rep = normalize_dataframe(
        df, corr, code_col="Region", group_cols=["parti"], additive_cols=["stemmer"]
    )
    got = {(r.parti, r.Region): r.stemmer for r in out.itertuples()}
    assert got[("Ap", "5001")] == 42000, got        # 40000 + 2000 aggregert
    assert got[("Ap", "0301")] == 90000             # uendret
    assert rep.rows_out == 2                         # to referansekoder

    # Nå en splittet kildekode (2023->2024): skal droppes og rapporteres.
    corr24 = build_correspondence(_client(), source_year="2023", reference_year="2024")
    df2 = pd.DataFrame([
        {"parti": "Ap", "Region": "1507", "stemmer": 12345},   # Ålesund, splitt
        {"parti": "Ap", "Region": "0301", "stemmer": 90000},   # Oslo, beholdes
    ])
    out2, rep2 = normalize_dataframe(
        df2, corr24, code_col="Region", group_cols=["parti"], additive_cols=["stemmer"]
    )
    codes = set(out2["Region"])
    assert "1507" not in codes and "1508" not in codes and "1580" not in codes
    assert "0301" in codes
    assert "1507" in rep2.split_codes_dropped
    print(f"\nSplitt droppet ved re-nøkling: {rep2.split_codes_dropped}")


ALL_TESTS = [
    test_merge_2020_multihop,
    test_split_2024_is_logged_not_guessed,
    test_unchanged_oslo_passthrough,
    test_dataframe_merge_aggregates_and_drops_split,
]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
