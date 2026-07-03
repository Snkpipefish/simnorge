"""Modul 1 — verifisering mot kjent fasit.

Kjøres mot ekte SSB-API (krever nett). Kan også kjøres direkte:
    .venv/bin/python -m tests.test_module1_facit

Verifiserer:
1. FASIT: Oslo (0301), stortingsvalg 2025 (08092) — partiprosent matcher fasit.
2. HOLDNING: tillit (13834) gir demografisk oppdelte prosenttall.
3. STRUKTUR: husholdningsinntekt (06944) gir median per kommune.
4. Prikking: manglende/prikkede celler markeres eksplisitt, aldri imputert.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import SSBClient, election_shares, trust, household_income  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Kjent fasit — Oslo stortingsvalg 2025 (08092, GodkjenteProsent).
OSLO_FACIT = {"Ap": 25.7, "Høyre": 18.5, "FrP": 14.3, "SV": 10.7, "MDG": 10.3}
TOL = 0.15  # prosentpoeng


def _client() -> SSBClient:
    return SSBClient()


def test_election_facit_oslo_2025() -> None:
    client = _client()
    df = election_shares(client, region="0301", year="2025")
    shares = dict(zip(df["parti"], df["prosent"]))
    print("\nOslo stortingsvalg 2025 (08092):")
    for _, row in df.iterrows():
        print(f"  {row['parti']:8s} {row['parti_navn']:28s} {row['prosent']:.1f}%")
    for parti, forventet in OSLO_FACIT.items():
        got = shares.get(parti)
        assert got is not None, f"Mangler {parti} i uttrekket"
        assert abs(got - forventet) <= TOL, (
            f"{parti}: fikk {got}, forventet ≈ {forventet} (±{TOL})"
        )
    # Sanity: prosentene summerer ~100 over alle partier (her bare hovedpartier,
    # så <= 100), og ingen negative.
    assert (df["prosent"] >= 0).all()
    assert df["prosent"].sum() <= 100.5


def test_trust_demographics() -> None:
    client = _client()
    df = trust(client, measure="Gjsnitt", year="2025")
    print("\nTillit til politiet, gj.snittsskår 0-10 (13834):")
    sub = df[df["kjonn"] == "Begge kjønn"]
    for _, row in sub.iterrows():
        print(f"  {row['aldersgruppe']:14s} {row['gjsnitt']}")
    # Demografisk oppdeling finnes: flere aldersgrupper, begge kjønn + M/K.
    assert df["aldersgruppe"].nunique() >= 4
    assert df["kjonn"].nunique() == 3
    # Skår er på 0-10-skala der den finnes.
    present = df.dropna(subset=["gjsnitt"])
    assert (present["gjsnitt"].between(0, 10)).all()
    # Verdiene skal variere mellom aldersgruppene (ikke en konstant).
    assert sub["gjsnitt"].nunique() > 1


def test_household_income_oslo() -> None:
    client = _client()
    df = household_income(client, region="0301")
    row = df.iloc[0]
    median = row["median_kr"]
    print(f"\nMedian husholdningsinntekt {row['region_navn']} "
          f"({row['Tid']}): {median:,.0f} kr".replace(",", " "))
    # Plausibel størrelsesorden for en norsk kommune (>200k, <2M kr).
    assert not math.isnan(median)
    assert 200_000 < median < 2_000_000


def test_suppression_marked_not_imputed() -> None:
    """Bekreft at prikkede celler markeres (missing=True, value=NaN),
    ikke fylt med et tall."""
    client = _client()
    # Tillit brutt fint på alder×kjønn×antall svar kan ha prikkede celler.
    df = trust(client, measure="AntallSvar", year="2025",
               trust_in=["01", "02", "03", "04", "05", "06"])
    missing = df[df["missing"]]
    print(f"\nPrikkede/manglende celler i 13834-uttrekk: {len(missing)}")
    # Uansett om det finnes prikkede celler eller ikke: ingen NaN skal være
    # erstattet av et tall, og hver missing-rad har value=NaN.
    assert df.loc[df["missing"], "antallsvar"].isna().all()
    assert (df["missing"] == df["antallsvar"].isna()).all()


ALL_TESTS = [
    test_election_facit_oslo_2025,
    test_trust_demographics,
    test_household_income_oslo,
    test_suppression_marked_not_imputed,
]


if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tester bestått.")
    sys.exit(1 if failures else 0)
