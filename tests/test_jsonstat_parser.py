"""Offline enhetstester for json-stat2-parseren og prikkehåndteringen.

Krever ikke nett. Bruker syntetiske json-stat2-datasett for å bevise at:
- flat-indeks dekodes korrekt til dimensjonskoordinater (row-major),
- sparse (dict) value-felt håndteres,
- prikkede celler (null value + statussymbol) markeres med missing=True og
  value=NaN — ALDRI imputert til et tall.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from data.ssb_client import SSBClient  # noqa: E402


def _toy_dataset(values, status=None):
    """2x2 datasett: Region(A,B) x Kjonn(M,K)."""
    js = {
        "id": ["Region", "Kjonn"],
        "size": [2, 2],
        "dimension": {
            "Region": {"category": {"index": {"A": 0, "B": 1},
                                    "label": {"A": "Alfa", "B": "Beta"}}},
            "Kjonn": {"category": {"index": {"M": 0, "K": 1},
                                   "label": {"M": "Menn", "K": "Kvinner"}}},
        },
        "value": values,
    }
    if status is not None:
        js["status"] = status
    return js


def test_rowmajor_decoding() -> None:
    # Row-major: siste dim (Kjonn) varierer raskest.
    # flat 0=(A,M) 1=(A,K) 2=(B,M) 3=(B,K)
    df = SSBClient._jsonstat2_to_tidy(_toy_dataset([10, 11, 20, 21]))
    got = {(r.Region, r.Kjonn): r.value for r in df.itertuples()}
    assert got == {("A", "M"): 10, ("A", "K"): 11, ("B", "M"): 20, ("B", "K"): 21}
    # Etikettkolonner finnes.
    assert set(df.loc[df.Region == "A", "Region_label"]) == {"Alfa"}


def test_suppressed_cell_is_marked_not_imputed() -> None:
    # (B,M) prikket: value=None, status ":" (konfidensielt).
    df = SSBClient._jsonstat2_to_tidy(
        _toy_dataset([10, 11, None, 21], status={"2": ":"})
    )
    bm = df[(df.Region == "B") & (df.Kjonn == "M")].iloc[0]
    assert bool(bm["missing"]) is True
    assert pd.isna(bm["value"])          # NaN, ikke 0 eller annen verdi
    assert bm["status"] == ":"
    # Øvrige celler upåvirket og ikke-missing.
    assert df["missing"].sum() == 1
    assert df.loc[~df["missing"], "value"].notna().all()


def test_sparse_value_dict() -> None:
    # Sparse: bare to celler oppgitt; resten skal bli missing/NaN.
    df = SSBClient._jsonstat2_to_tidy(_toy_dataset({"0": 10, "3": 21}))
    assert df["missing"].sum() == 2
    assert df.loc[~df["missing"], "value"].tolist() == [10, 21]


ALL_TESTS = [
    test_rowmajor_decoding,
    test_suppressed_cell_is_marked_not_imputed,
    test_sparse_value_dict,
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
