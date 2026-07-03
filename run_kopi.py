"""Bygg den syntetiske kopien av Norge og skriv den til parquet.

    .venv/bin/python run_kopi.py                    # hele landet (~5,6 mill)
    .venv/bin/python run_kopi.py --kommuner 0301 1103   # test på et utvalg
"""

from __future__ import annotations

import argparse
import logging
import time

from kopi import bygg_kopi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ut", default="kopi_norge.parquet")
    p.add_argument("--befolkningsaar", default="2026")
    p.add_argument("--kommuner", nargs="*", default=None,
                   help="Kommunekoder for et testutvalg (default: alle)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    df = bygg_kopi(befolkningsaar=args.befolkningsaar,
                   kommuner=args.kommuner, seed=args.seed)
    df.to_parquet(args.ut, index=False)
    print(f"\nSkrev {len(df):,} personer til {args.ut} "
          f"({time.time() - t0:.0f} s)")

    print("\n--- Smakebiter ---")
    print("\nPersonlighetstyper (andel av befolkningen):")
    print((df["personlighetstype"].value_counts(normalize=True) * 100)
          .round(1).to_string())
    print("\nParti blant velgere, nasjonalt (%):")
    v = df[~df["parti"].isin(["stemte ikke", "ikke stemmerett"])]
    print((v["parti"].value_counts(normalize=True) * 100).round(1)
          .to_string())
    print("\nMedian bruttoinntekt etter kjønn (17+):")
    print(df[df["alder"] >= 17].groupby("kjonn", observed=True)
          ["brutto_inntekt"].median().round(-3).to_string())


if __name__ == "__main__":
    main()
