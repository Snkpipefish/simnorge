"""Legg holdningsakser + holdningsgruppe på den syntetiske kopien.

    .venv/bin/python run_holdningsgrupper.py            # oppdaterer kopi_norge.parquet
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from data import SSBClient
from kopi.holdning import tildel_holdning, GRUPPE_AKSER
from kopi.kriminalitet import KriminalitetsModell
from kopi.utdanning import UtdanningsModell
from kopi.livssyn import LivssynModell
from kopi.tillit_verdi import TillitVerdiModell
from kopi.personlighet import tildel_personlighet_ess, TYPER
from kopi.ess import les_posisjoner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fil", default="kopi_norge.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    ssb = SSBClient()
    df = pd.read_parquet(args.fil)
    df = KriminalitetsModell(ssb).tildel(df, seed=args.seed + 10)
    df = UtdanningsModell(ssb).tildel(df, seed=args.seed + 20)
    ess = les_posisjoner()
    if ess and "personlighet" in ess:
        koder = tildel_personlighet_ess(df, ess["personlighet"],
                                        seed=args.seed + 25)
        df["personlighetstype"] = pd.Categorical.from_codes(
            koder, categories=TYPER)
    df = LivssynModell(ssb).tildel(df, seed=args.seed + 30)
    df = TillitVerdiModell(ssb).tildel(df, seed=args.seed + 40)
    df = tildel_holdning(df, seed=args.seed)
    df.to_parquet(args.fil, index=False)
    print(f"\nOppdaterte {args.fil}: {len(GRUPPE_AKSER)} akser + utdanning + "
          f"livssyn + kriminalitetsflagg + holdningsgruppe/hovedgruppe "
          f"({time.time() - t0:.0f} s)")

    print("\nHoldningsgrupper nasjonalt (16+, %):")
    v = df[df["alder"] >= 16]
    print((v["holdningsgruppe"].value_counts(normalize=True) * 100)
          .round(1).to_string())

    print("\nKriminalitet (målte rater, trukket per person):")
    print(f"  utsatt for vold/trusler (16+): "
          f"{v['utsatt_vold'].mean() * 100:.1f} %")
    print(f"  utsatt for tyveri/skadeverk (16+): "
          f"{v['utsatt_tyveri'].mean() * 100:.1f} %")
    print(f"  siktet for vold (15+): "
          f"{df[df['alder'] >= 15]['siktet_vold'].mean() * 100:.2f} %")
    print(f"  trygghet: snitt {v['trygghet'].mean():.0f}, "
          f"P10 {v['trygghet'].quantile(0.1):.0f}, "
          f"P90 {v['trygghet'].quantile(0.9):.0f}")

    print("\nGruppe × parti (radprosent, velgere) — sanity:")
    velgere = v[~v["parti"].isin(["stemte ikke", "ikke stemmerett"])]
    tab = pd.crosstab(velgere["holdningsgruppe"], velgere["parti"],
                      normalize="index") * 100
    print(tab.round(0).to_string())

    print("\nPersonlighet × parti (målte tilt) — andel N-typer og E-typer per parti:")
    vel = v[~v["parti"].isin(["stemte ikke", "ikke stemmerett"])].copy()
    pt = vel["personlighetstype"].astype(str)
    vel["N"] = pt.str[1].eq("N")
    vel["E"] = pt.str[0].eq("E")
    print((vel.groupby("parti", observed=True)[["N", "E"]].mean() * 100)
          .round(1).to_string())

    print("\nStørste gruppe per sentralitetsytterpunkt:")
    for navn in ["Oslo - Oslove", "Utsira"]:
        k = v[v["kommune_navn"] == navn]
        print(f"  {navn}: "
              f"{(k['holdningsgruppe'].value_counts(normalize=True) * 100).round(1).head(3).to_dict()}")


if __name__ == "__main__":
    main()
