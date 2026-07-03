"""Legg tilbøyelighetsskårene på den syntetiske kopien.

    .venv/bin/python run_tilboyeligheter.py        # oppdaterer kopi_norge.parquet
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from data import SSBClient
from kopi.tilboyelighet import TilboyelighetsModell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fil", default="kopi_norge.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    df = pd.read_parquet(args.fil)
    df = TilboyelighetsModell(SSBClient()).tildel(df, seed=args.seed + 50)
    df.to_parquet(args.fil, index=False)
    print(f"\nOppdaterte {args.fil}: +5 tilbøyeligheter ({time.time() - t0:.0f} s)")

    v = df[df["alder"] >= 16]
    kol = ["tilb_stemme", "tilb_flytte", "tilb_frivillig", "tilb_protest",
           "tilb_risiko"]
    print("\nSnitt per hovedgruppe:")
    print(v.groupby("hovedgruppe", observed=True)[kol].mean().round(1).to_string())
    print("\nYtterpunkter per holdningsgruppe (protest og stemme):")
    g = v.groupby("holdningsgruppe", observed=True)[["tilb_protest", "tilb_stemme"]].mean()
    print(g.sort_values("tilb_protest", ascending=False).head(4).round(1).to_string())
    print(g.sort_values("tilb_stemme", ascending=False).head(4).round(1).to_string())


if __name__ == "__main__":
    main()
