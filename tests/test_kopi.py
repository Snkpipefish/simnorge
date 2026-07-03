"""Tester for kopi/ — de rene funksjonene, uten nettverk.

Kjør direkte:
    .venv/bin/python -m tests.test_kopi

Byggeren selv verifiseres ved kjøring (run_kopi.py logger avvik mot faktisk
valgresultat); her testes logikken som ikke krever SSB.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from kopi.personlighet import tildel_personlighet, TYPER, _P_T  # noqa: E402
from kopi.valg import ValgModell, PARTIER  # noqa: E402


# --------------------------------------------------------------------------- #
# Personlighet                                                                #
# --------------------------------------------------------------------------- #
def test_personlighet_gyldige_koder_og_deterministisk():
    alder = np.array([3, 17, 30, 50, 80] * 200)
    kjonn = np.array([0, 1] * 500, dtype=np.int8)
    a = tildel_personlighet(alder, kjonn, seed=42)
    b = tildel_personlighet(alder, kjonn, seed=42)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < len(TYPER)


def test_personlighet_kjonnstilt_paa_tf():
    """T-andelen skal følge antakelsen: menn ~0.60, kvinner ~0.38."""
    n = 200_000
    alder = np.full(n, 40)
    for ki, forventet in [(0, _P_T[0]), (1, _P_T[1])]:
        koder = tildel_personlighet(alder, np.full(n, ki, dtype=np.int8), seed=1)
        t_andel = ((koder >> 1) & 1 == 0).mean()      # bit 1: 0=T
        assert abs(t_andel - forventet) < 0.01


def test_personlighet_alle_typer_forekommer():
    n = 100_000
    koder = tildel_personlighet(np.full(n, 40), np.zeros(n, dtype=np.int8), seed=0)
    assert set(np.unique(koder)) == set(range(16))


# --------------------------------------------------------------------------- #
# Raking (uten nettverk: instansier ValgModell uten __init__)                 #
# --------------------------------------------------------------------------- #
def _modell_med_tilt(tilt: np.ndarray) -> ValgModell:
    m = ValgModell.__new__(ValgModell)
    m.tilt = tilt
    return m


def test_rake_reproduserer_fasit_med_skjev_tilt():
    """Uansett tilt skal det vektede aggregatet matche kommunens resultat."""
    rng = np.random.default_rng(0)
    tilt = rng.uniform(0.2, 5.0, size=(2, 3, 5, len(PARTIER)))
    m = _modell_med_tilt(tilt)
    s = np.array([0.30, 0.15, 0.20, 0.03, 0.10, 0.07, 0.04, 0.03, 0.05, 0.03])
    teller = rng.integers(0, 500, size=30)
    q = m._rake(s, teller)
    assert q.shape == (30, len(PARTIER))
    assert np.allclose(q.sum(axis=1), 1.0)
    agg = (teller[:, None] * q).sum(axis=0) / teller.sum()
    assert np.abs(agg - s).max() < 1e-6


def test_rake_bevarer_null_andel():
    """Parti med 0 % i kommunen skal ha 0 sannsynlighet i alle celler."""
    m = _modell_med_tilt(np.ones((2, 3, 5, len(PARTIER))))
    s = np.array([0.5, 0.5] + [0.0] * (len(PARTIER) - 2))
    q = m._rake(s, np.full(30, 100))
    assert np.all(q[:, 2:] == 0)
    assert np.allclose(q.sum(axis=1), 1.0)


def test_rake_tom_kommune_gir_normaliserte_rader():
    m = _modell_med_tilt(np.ones((2, 3, 5, len(PARTIER))))
    s = np.full(len(PARTIER), 1 / len(PARTIER))
    q = m._rake(s, np.zeros(30, dtype=int))
    assert np.allclose(q.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# Holdningsgrupper (offline: sentralitet injiseres)                           #
# --------------------------------------------------------------------------- #
def _syntetisk_kopi(n: int = 20_000, seed: int = 0):
    import pandas as pd
    from kopi.valg import UTFALL
    from kopi.utdanning import NIVAER
    rng = np.random.default_rng(seed)
    kommuner = ["0301", "1151"]                    # sentral, usentral
    alder = rng.integers(0, 95, n).astype(np.int16)
    voksen = alder >= 16

    def akse(mu, sd):
        return np.where(voksen, np.clip(rng.normal(mu, sd, n), 0, 100),
                        np.nan).astype(np.float32)

    return pd.DataFrame({
        "kommune": pd.Categorical(rng.choice(kommuner, n), categories=kommuner),
        "kjonn": pd.Categorical(rng.choice(["mann", "kvinne"], n),
                                categories=["mann", "kvinne"]),
        "alder": alder,
        "brutto_inntekt": np.where(rng.random(n) < 0.8,
                                   rng.lognormal(13, 0.6, n), np.nan),
        "parti": pd.Categorical(rng.choice(UTFALL, n), categories=UTFALL),
        "utdanning": pd.Categorical.from_codes(
            np.where(voksen, rng.integers(0, 4, n), -1), categories=NIVAER),
        "medlem_dnk": rng.random(n) < 0.6,
        "trygghet": akse(70, 15),
        "institusjonstillit": akse(60, 18),
        "verdiliberal": akse(75, 15),
    })


def _kriminalitetsmodell():
    """Offline-modell med injiserte, målt-lignende rater."""
    from kopi.kriminalitet import KriminalitetsModell, _SIKTET_BAND
    m = KriminalitetsModell.__new__(KriminalitetsModell)
    m.p_utsatt_vold = np.array([[0.08, 0.04, 0.03, 0.01],
                                [0.09, 0.05, 0.03, 0.01]])
    m.p_utsatt_tyv = np.full((2, 4), 0.07)
    m.p_uro = np.array([[0.04, 0.04, 0.05, 0.06],
                        [0.10, 0.11, 0.12, 0.14]])           # kvinner mer uro
    m.voldsfaktor = {"0301": 1.8, "1151": 0.6}
    m.tyvfaktor = {"0301": 1.5, "1151": 0.7}
    m.siktede = np.full((2, len(_SIKTET_BAND)), 40.0)
    return m


def test_holdning_barn_uten_gruppe_og_deterministisk():
    from kopi.holdning import tildel_holdning, AKSER
    df = _syntetisk_kopi()
    sent = {"0301": 1, "1151": 6}
    a = tildel_holdning(df, seed=3, sentralitet=sent)
    b = tildel_holdning(df, seed=3, sentralitet=sent)
    assert a["holdningsgruppe"].equals(b["holdningsgruppe"])
    barn = a["alder"] < 16
    assert a.loc[barn, "holdningsgruppe"].isna().all()
    assert a.loc[~barn, "holdningsgruppe"].notna().all()
    for akse in AKSER:
        v = a.loc[~barn, akse]
        assert v.between(0, 100).all()


def test_holdning_parti_og_sentralitet_styrer():
    from kopi.holdning import tildel_holdning
    df = _syntetisk_kopi()
    a = tildel_holdning(df, seed=1, sentralitet={"0301": 1, "1151": 6})
    voksen = a[a["alder"] >= 16]
    # Med MÅLTE posisjoner (ESS) overlapper partiene reelt — vi tester derfor
    # hovedgruppe-tyngdepunkt og modal undergruppe, ikke skarpe familier.
    frp = voksen[voksen["parti"] == "FrP"]
    assert (frp["hovedgruppe"] == "høyresiden").mean() > 0.45
    assert frp["holdningsgruppe"].mode()[0] == "nasjonalkonservativ"
    sp = voksen[voksen["parti"] == "Sp"]
    assert (sp["hovedgruppe"] == "sentrum og tradisjon").mean() > 0.45
    assert sp["holdningsgruppe"].mode()[0] == "distriktsforankret"
    # Sentral kommune skal ligge høyere på sentrum/distrikt-aksen.
    snitt = voksen.groupby("kommune", observed=True)["sentrum_distrikt"].mean()
    assert snitt["0301"] > snitt["1151"] + 5


def test_holdning_eu1994_anker_styrer_sentrum():
    from kopi.holdning import tildel_holdning
    df = _syntetisk_kopi()
    sent = {"0301": 1, "1151": 6}
    # Målt-lignende nei-andeler: ja-kommune vs sterk nei-kommune.
    a = tildel_holdning(df, seed=1, sentralitet=sent,
                        eu1994={"0301": 0.33, "1151": 0.85})
    voksen = a[a["alder"] >= 16]
    snitt = voksen.groupby("kommune", observed=True)["sentrum_distrikt"].mean()
    assert snitt["0301"] > snitt["1151"] + 15
    # Uten anker gjelder sentralitetsantakelsen — mindre spenn, samme retning.
    b = tildel_holdning(df, seed=1, sentralitet=sent)
    snitt_b = (b[b["alder"] >= 16]
               .groupby("kommune", observed=True)["sentrum_distrikt"].mean())
    assert snitt_b["0301"] > snitt_b["1151"] + 5
    # De andre aksene skal ikke påvirkes av EU-ankeret.
    for akse in ["oko_fordeling", "innvandring", "klima"]:
        pd_a = voksen[akse].mean()
        pd_b = b[b["alder"] >= 16][akse].mean()
        assert abs(pd_a - pd_b) < 0.5


def test_kriminalitet_flagg_og_trygghet():
    df = _syntetisk_kopi().drop(columns=["trygghet"])
    a = _kriminalitetsmodell().tildel(df, seed=5)
    b = _kriminalitetsmodell().tildel(df, seed=5)
    assert a["trygghet"].equals(b["trygghet"])                # deterministisk
    barn = a["alder"] < 16
    assert a.loc[barn, "utsatt_vold"].isna().all()
    assert a.loc[barn, "trygghet"].isna().all()
    voksen = a[~barn]
    assert voksen["trygghet"].between(0, 100).all()
    # Målt kjønnsgradient i uro -> kvinner lavere trygghet.
    snitt = voksen.groupby("kjonn", observed=True)["trygghet"].mean()
    assert snitt["mann"] > snitt["kvinne"] + 2
    # Kommunens målte voldsnivå -> lavere trygghet og mer utsatthet i 0301.
    kom = voksen.groupby("kommune", observed=True)
    assert kom["trygghet"].mean()["0301"] < kom["trygghet"].mean()["1151"]
    assert (kom["utsatt_vold"].mean()["0301"]
            > kom["utsatt_vold"].mean()["1151"])


def test_holdning_trygghet_skiller_grupper():
    from kopi.holdning import tildel_holdning
    df = _syntetisk_kopi()
    utrygg = df.copy(); utrygg["trygghet"] = np.where(df["alder"] >= 16, 15.0, np.nan)
    trygg = df.copy(); trygg["trygghet"] = np.where(df["alder"] >= 16, 85.0, np.nan)
    sent = {"0301": 3, "1151": 3}
    gu = tildel_holdning(utrygg, seed=2, sentralitet=sent)["holdningsgruppe"]
    gt = tildel_holdning(trygg, seed=2, sentralitet=sent)["holdningsgruppe"]
    urolige = {"urolig midtstrøms", "trygghetssøkende konservativ",
               "bekymret velferdsvelger"}
    andel_u = gu.isin(urolige).mean()
    andel_t = gt.isin(urolige).mean()
    assert andel_u > 0.25 and andel_t < 0.05


def test_holdning_tillit_og_verdi_skiller_grupper():
    from kopi.holdning import tildel_holdning
    df = _syntetisk_kopi()
    sent = {"0301": 3, "1151": 3}
    lav = df.copy(); lav["institusjonstillit"] = np.where(df["alder"] >= 16, 10.0, np.nan)
    g = tildel_holdning(lav, seed=2, sentralitet=sent)["holdningsgruppe"]
    kritiske = {"politisk fremmedgjort", "systemkritisk protest",
                "systemkritisk venstre"}
    assert g.isin(kritiske).mean() > 0.25
    kons = df.copy(); kons["verdiliberal"] = np.where(df["alder"] >= 16, 8.0, np.nan)
    g2 = tildel_holdning(kons, seed=2, sentralitet=sent)["holdningsgruppe"]
    assert (g2 == "kristenkonservativ").mean() > 0.1


def test_utdanning_rake_treffer_margin():
    from kopi.utdanning import _rake_celler
    rng = np.random.default_rng(0)
    basis = rng.uniform(0.1, 5.0, size=(40, 4))
    teller = rng.integers(1, 300, size=40)
    margin = np.array([0.25, 0.40, 0.22, 0.13])
    q = _rake_celler(basis, teller, margin)
    assert np.allclose(q.sum(axis=1), 1.0)
    agg = (teller[:, None] * q).sum(axis=0) / teller.sum()
    assert np.abs(agg - margin).max() < 1e-6


def test_hovedgruppe_dekker_alle_grupper():
    from kopi.holdning import GRUPPER, HOVEDGRUPPE, HOVEDGRUPPER, tildel_holdning
    assert set(HOVEDGRUPPE) == set(GRUPPER)
    assert set(HOVEDGRUPPE.values()) == set(HOVEDGRUPPER)
    df = _syntetisk_kopi()
    a = tildel_holdning(df, seed=4, sentralitet={"0301": 2, "1151": 5})
    barn = a["alder"] < 16
    assert a.loc[barn, "hovedgruppe"].isna().all()
    assert a.loc[~barn, "hovedgruppe"].notna().all()
    # Konsistens: hovedgruppen skal alltid være gruppens forelder.
    v = a[~barn]
    assert (v["hovedgruppe"].astype(str)
            == v["holdningsgruppe"].astype(str).map(HOVEDGRUPPE)).all()


def test_tilboyeligheter_offline():
    import pandas as pd
    from kopi.tilboyelighet import TilboyelighetsModell
    from kopi.personlighet import TYPER
    m = TilboyelighetsModell.__new__(TilboyelighetsModell)
    m.stemme_alder = np.array([0.55, 0.70, 0.78, 0.85, 0.83, 0.72])
    m._stemme_bandnavn = ["18-19", "20-24", "25-44", "45-66", "67-79", "080+"]
    m.stemme_utd = np.array([0.65, 0.78, 0.90])
    m.stemme_tot = 0.78
    m._flytt_band = ["00-04", "05-09", "10-14", "15-19", "20-24", "25-29",
                     "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
                     "60-64", "65-69", "70-74", "80+"]
    m.flytt = np.zeros((2, 16)); m.flytt[:, 4:6] = 500.0; m.flytt[:, 10] = 50.0
    m.friv = np.full((2, 4), 0.55)
    df = _syntetisk_kopi()
    rng = np.random.default_rng(9)
    df["personlighetstype"] = pd.Categorical.from_codes(
        rng.integers(0, 16, len(df)), categories=TYPER)
    df["siktet_vold"] = pd.array(rng.random(len(df)) < 0.01, dtype="boolean")
    a = m.tildel(df, seed=1, sentralitet={"0301": 1, "1151": 6})
    b = m.tildel(df, seed=1, sentralitet={"0301": 1, "1151": 6})
    assert a["tilb_stemme"].equals(b["tilb_stemme"])
    barn = a["alder"] < 16
    for k in ["tilb_stemme", "tilb_flytte", "tilb_frivillig", "tilb_protest",
              "tilb_risiko"]:
        assert a.loc[barn, k].isna().all()
        assert a.loc[~barn, k].between(0, 100).all()
    v = a[~barn]
    # Unge (20-29) skal ha klart høyere flyttetilbøyelighet enn 50-59.
    unge = v[(v["alder"] >= 20) & (v["alder"] < 30)]["tilb_flytte"].mean()
    eldre = v[(v["alder"] >= 50) & (v["alder"] < 60)]["tilb_flytte"].mean()
    assert unge > eldre + 3
    # Lav tillit -> høyere protest.
    lav = v[v["institusjonstillit"] < 35]["tilb_protest"].mean()
    hoy = v[v["institusjonstillit"] > 75]["tilb_protest"].mean()
    assert lav > hoy + 10


def test_personlighet_ess_tilt():
    import pandas as pd
    from kopi.personlighet import tildel_personlighet_ess, TYPER
    ess_pers = {b: {"inntektsdesil": [0.0] * 10,
                    "utdanning": {str(i): 0.0 for i in range(4)},
                    "parti": {}} for b in "ENFJ"}
    ess_pers["N"]["parti"] = {"MDG": 2.0, "KrF": -2.0}
    df = _syntetisk_kopi()
    a = tildel_personlighet_ess(df, ess_pers, seed=7)
    b = tildel_personlighet_ess(df, ess_pers, seed=7)
    assert np.array_equal(a, b)
    typer = pd.Series([TYPER[k] for k in a], index=df.index)
    voksen = df["alder"] >= 16
    n_mdg = typer[voksen & (df["parti"] == "MDG")].str[1].eq("N").mean()
    n_krf = typer[voksen & (df["parti"] == "KrF")].str[1].eq("N").mean()
    assert n_mdg > n_krf + 0.15          # målt tilt slår gjennom
    # Barn påvirkes ikke av tiltene (ingen parti/inntekt/utdanning).
    n_barn = typer[~voksen].str[1].eq("N").mean()
    assert abs(n_barn - 0.30) < 0.04


def test_helse_offline():
    import pandas as pd
    from kopi.helse import HelseModell, _KPR_BAND, _PSYKISK, _MUSKEL
    m = HelseModell.__new__(HelseModell)
    m.rater = {_PSYKISK: np.array([[80, 150, 160, 150, 140, 150],
                                   [90, 260, 260, 220, 180, 190]], float),
               _MUSKEL: np.full((2, len(_KPR_BAND)), 250.0)}
    m.smr = {_PSYKISK: {"0301": 1.3, "1151": 0.7},
             _MUSKEL: {"0301": 1.0, "1151": 1.0}}
    m.levealder = {("0301", 0, "0"): 80.0, ("0301", 1, "0"): 84.0,
                   ("1151", 0, "0"): 79.0, ("1151", 1, "0"): 83.0,
                   ("03", 0, "0"): 80.0, ("03", 0, "1"): 76.0, ("03", 0, "3"): 83.0,
                   ("0", 0, "0"): 79.5, ("0", 1, "0"): 83.5}
    m.ess_helse = {"basis": [[0.85, 0.8, 0.75, 0.65], [0.8, 0.78, 0.7, 0.55]],
                   "inntektsdesil": [-0.2, -0.1, -0.05, 0, 0, 0.02, 0.04, 0.08, 0.12, 0.12],
                   "utdanning": {"0": -0.1, "1": 0.0, "2": 0.03, "3": 0.12}}
    df = _syntetisk_kopi()
    a = m.tildel(df, seed=2)
    b = m.tildel(df, seed=2)
    assert a["psykisk_kontakt"].equals(b["psykisk_kontakt"])
    # Målt kommunenivå (SMR) skal slå gjennom.
    kom = a[a["alder"] >= 16].groupby("kommune", observed=True)
    assert (kom["psykisk_kontakt"].mean()["0301"]
            > kom["psykisk_kontakt"].mean()["1151"] + 0.02)
    # Fylkesgradienten på levealder: uni over grunnskole for Oslo-menn.
    o = a[(a["kommune"] == "0301") & (a["kjonn"] == "mann")]
    assert (o[o["utdanning"] == "uni_lang"]["forventet_levealder"].mean()
            > o[o["utdanning"] == "grunnskole"]["forventet_levealder"].mean() + 3)
    # Egenvurdert helse: målt inntektsgradient.
    v = a[(a["alder"] >= 25) & a["brutto_inntekt"].notna()].copy()
    import pandas as pd2
    lav = v[v["brutto_inntekt"] < v["brutto_inntekt"].quantile(0.2)]
    hoy = v[v["brutto_inntekt"] > v["brutto_inntekt"].quantile(0.8)]
    assert (hoy["helse_god"].astype("float").mean()
            > lav["helse_god"].astype("float").mean() + 0.1)


if __name__ == "__main__":
    test_personlighet_gyldige_koder_og_deterministisk()
    test_personlighet_kjonnstilt_paa_tf()
    test_personlighet_alle_typer_forekommer()
    test_rake_reproduserer_fasit_med_skjev_tilt()
    test_rake_bevarer_null_andel()
    test_rake_tom_kommune_gir_normaliserte_rader()
    test_holdning_barn_uten_gruppe_og_deterministisk()
    test_holdning_parti_og_sentralitet_styrer()
    test_holdning_eu1994_anker_styrer_sentrum()
    test_kriminalitet_flagg_og_trygghet()
    test_holdning_trygghet_skiller_grupper()
    test_holdning_tillit_og_verdi_skiller_grupper()
    test_utdanning_rake_treffer_margin()
    test_hovedgruppe_dekker_alle_grupper()
    test_tilboyeligheter_offline()
    test_personlighet_ess_tilt()
    test_helse_offline()
    print("\nAlle kopi-tester grønne.")
