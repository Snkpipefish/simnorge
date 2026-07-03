"""Holdningsgrupper — ANTATT fra bosted, inntekt og stemmegivning (eierbeslutning).

Hver person 16+ får skårer 0-100 på fire holdningsakser, og grupperes deretter
i nærmeste av et sett navngitte holdningsprofiler. ALT her er antakelser i
samme regime som personlighetstypene: dokumentert, samlet på ett sted,
justerbart — ikke målt. Signalrekkefølgen er som bestilt:

1. PARTI (sterkest): basisposisjon per akse fra partienes kjente profil.
2. INNTEKT: skyver økonomiaksen mot marked med stigende inntektspersentil.
3. BOSTED: målt ESS-bostedsgradient (domicil, innen parti) per akse for
   øko/innvandring/klima, koblet til kommunens sentralitetsklasse (Klass
   128).
4. REGION: målt NUTS2-residual (innen parti og bosted) per akse — liten,
   men gir aksene ulik geografi (uten den kollapser holdningsgeografien
   til én by/land-dimensjon).
5. EU-1994 (kopi/eu1994.py): sentrum/distrikt-aksen ankres i kommunens
   målte nei-andel fra folkeavstemningen — dens egen geografi, ikke en
   kopi av sentralitetsindeksen. Uten anker: antatt sentralitetsskyv.

Spredning: normalstøy per akse, større for hjemmesittere og 16-17-åringer
(basis 50 — vi vet mindre om dem). Barn under 16 får ingen holdning (NaN).

Aksene (0 -> 100):
- oko_fordeling:    sterk omfordeling -> marked/lav skatt
- innvandring:      restriktiv       -> liberal
- klima:            vekst først      -> vern først
- sentrum_distrikt: distriktsprioritering -> sentraliseringsvennlig
- trygghet:         utrygg/urolig    -> trygg   (bygges i kopi/kriminalitet.py
  av MÅLTE ingredienser: uro-gradient 04621, kommunens voldsnivå 08487, egne
  utsatthetsflagg — grupperingen her bruker den som femte akse)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from .ess import les_posisjoner

logger = logging.getLogger("simnorge.kopi.holdning")

AKSER = ["oko_fordeling", "innvandring", "klima", "sentrum_distrikt"]

# Basisposisjon per parti (ANTAKELSE — partienes grove profil på hver akse).
# Hjemmesittere og 16-17-åringer starter i midten med større spredning.
PARTI_POSISJON: dict[str, tuple[float, float, float, float]] = {
    "Rødt":            (5,  70, 65, 40),
    "SV":              (15, 75, 75, 45),
    "Ap":              (35, 50, 45, 50),
    "Sp":              (45, 35, 35, 15),
    "MDG":             (40, 70, 90, 55),
    "KrF":             (50, 45, 50, 40),
    "Venstre":         (60, 75, 70, 65),
    "Høyre":           (75, 45, 40, 65),
    "FrP":             (80, 15, 20, 55),
    "Andre":           (50, 50, 50, 50),
    "stemte ikke":     (50, 50, 50, 50),
    "ikke stemmerett": (50, 50, 50, 50),
}

# Inntektens skyv på økonomiaksen: ±_INNTEKT_SKYV ved persentil 0/1 (ANTAKELSE).
_INNTEKT_SKYV = 12.0

# Sentralitetens skyv per akse ved mest sentral (klasse 1); lineært til
# motsatt fortegn ved minst sentral (klasse 6). (ANTAKELSE — brukes bare
# som fallback uten ESS-aggregater; med dem måles øko/innvandring/klima
# fra domicil-gradienten, og kun sentrum/distrikt-elementet gjelder.)
_SENTRALITET_SKYV = np.array([0.0, 9.0, 6.0, 15.0])

# Sentrum/distrikt-aksens EU-1994-anker (kopi/eu1994.py): skyv =
# -_EU_SKALA * (kommunens nei-andel - nasjonal nei-andel) * 100.
# Geografien er MÅLT (nei-andel 25-94 %); skalaen og antakelsen om at
# 1994-motsetningen fortsatt rangerer kommunene er ANTAKELSER.
_EU_SKALA = 0.5
_EU_NASJONAL = 0.522

# Kommunenummerets fylkesprefiks (2024) -> NUTS2-region (2021), for det
# målte regiontiltet fra ESS.
_FYLKE_NUTS2 = {
    "03": "NO08", "31": "NO08", "32": "NO08", "33": "NO08",   # Oslo og Viken
    "34": "NO02",                                             # Innlandet
    "39": "NO09", "40": "NO09", "42": "NO09",       # Agder og Sør-Østlandet
    "11": "NO0A", "15": "NO0A", "46": "NO0A",                 # Vestlandet
    "50": "NO06",                                             # Trøndelag
    "18": "NO07", "55": "NO07", "56": "NO07",                 # Nord-Norge
}

# Støy (standardavvik) per person: vanlig / svakt signal (hjemmesittere, 16-17).
_STOY = 10.0
_STOY_SVAK = 15.0

# Aksene grupperingen bruker: de fire som beregnes her + tre fra andre lag
# (trygghet: kopi/kriminalitet.py; tillit/verdi: kopi/tillit_verdi.py).
GRUPPE_AKSER = AKSER + ["trygghet", "institusjonstillit", "verdiliberal"]

# Holdningsgruppene: navn -> prototyp på
# (øko, innvandring, klima, sentrum, trygghet, tillit, verdi).
# (ANTAKELSE — segmentering; nærmeste prototyp vinner. Tillitsaksen skiller
# protest/fremmedgjort fra resten; verdiaksen skiller kristenkonservative og
# etablert høyre fra de liberale.)
GRUPPER: dict[str, tuple[float, ...]] = {
    # venstresiden
    "systemkritisk venstre":         (8, 68, 65, 42, 50, 22, 70),
    "rød aktivist":                  (15, 75, 75, 45, 55, 55, 78),
    "sosialdemokratisk kjerne":      (35, 50, 45, 48, 60, 68, 55),
    "tradisjonell arbeiderklasse":   (30, 40, 35, 40, 55, 55, 35),
    "bekymret velferdsvelger":       (28, 42, 45, 42, 28, 40, 50),
    # høyresiden
    "nasjonalkonservativ":           (72, 15, 22, 52, 50, 45, 38),
    "systemkritisk protest":         (75, 12, 18, 50, 35, 15, 42),
    "trygghetssøkende konservativ":  (60, 20, 30, 45, 22, 35, 42),
    "markedsliberal":                (78, 55, 45, 68, 65, 70, 62),
    "etablert konservativ":          (80, 45, 40, 65, 70, 75, 40),
    # grønne og liberale
    "grønn urban":                   (55, 75, 85, 65, 60, 60, 80),
    "trygg kosmopolitt":             (62, 82, 65, 75, 82, 78, 80),
    "utdannet sentrumsliberal":      (58, 65, 60, 60, 65, 74, 68),
    # sentrum og tradisjon
    "distriktsforankret":            (45, 35, 35, 15, 62, 58, 42),
    "kristenkonservativ":            (52, 42, 48, 30, 55, 62, 12),
    "stille tradisjonalist":         (50, 40, 48, 32, 48, 50, 35),
    # midten
    "trygg midtstrøms":              (50, 50, 50, 50, 72, 65, 55),
    "urolig midtstrøms":             (50, 45, 48, 48, 32, 45, 50),
    "politisk fremmedgjort":         (48, 45, 45, 48, 42, 15, 50),
    "ung uavklart":                  (50, 55, 55, 52, 45, 45, 62),
}

# --------------------------------------------------------------------------- #
# ESS-forankring: når kopi/ess_posisjoner.json finnes (bygget av kopi/ess.py  #
# fra mikrodataene), erstattes antakelsene over med MÅLTE verdier:            #
# - partiposisjon og spredning INNEN parti på øko/innvandring/klima           #
# - målt inntektsgradient på økonomiaksen (i stedet for ±12-antakelsen)       #
# - målt bosteds- (domicil) og regionsgradient (NUTS2) per akse i stedet     #
#   for sentralitetsantakelsen på de tre målte aksene ("geo"-nøkkelen)       #
# - gruppeprototypene ankres i partienes målte posisjon + avvikene under      #
# sentrum_distrikt har ikke noe ESS-mål og forblir antakelse uansett.         #
# --------------------------------------------------------------------------- #
# Gruppe -> (anker i ESS-tabellen, avvik). "sentrum" og "trygghet" er         #
# absolutte (ingen ESS-kilde); øvrige nøkler er avvik fra ankerets målte      #
# posisjon (ANTAKELSE — det som gjør gruppen til noe mer enn partiet).        #
_GRUPPE_SPEC: dict[str, tuple[str, dict[str, float]]] = {
    "systemkritisk venstre":         ("Rødt", {"sentrum": 42, "trygghet": 50, "tillit": -30}),
    "rød aktivist":                  ("SV", {"sentrum": 45, "trygghet": 55}),
    "sosialdemokratisk kjerne":      ("Ap", {"sentrum": 48, "trygghet": 60}),
    "tradisjonell arbeiderklasse":   ("Ap", {"sentrum": 40, "trygghet": 55, "verdi": -25}),
    "bekymret velferdsvelger":       ("Ap", {"sentrum": 42, "trygghet": 28, "tillit": -10}),
    "nasjonalkonservativ":           ("FrP", {"sentrum": 52, "trygghet": 50}),
    "systemkritisk protest":         ("FrP", {"sentrum": 50, "trygghet": 35, "tillit": -30}),
    "trygghetssøkende konservativ":  ("FrP", {"sentrum": 45, "trygghet": 22}),
    "markedsliberal":                ("Høyre", {"sentrum": 68, "trygghet": 65, "oko": 10, "verdi": 5}),
    "etablert konservativ":          ("Høyre", {"sentrum": 65, "trygghet": 78, "tillit": 8, "verdi": -10}),
    "grønn urban":                   ("MDG", {"sentrum": 65, "trygghet": 60}),
    "trygg kosmopolitt":             ("Venstre", {"sentrum": 75, "trygghet": 82, "tillit": 8}),
    "utdannet sentrumsliberal":      ("Venstre", {"sentrum": 60, "trygghet": 65, "oko": -8, "innv": -5}),
    "distriktsforankret":            ("Sp", {"sentrum": 15, "trygghet": 62}),
    "kristenkonservativ":            ("KrF", {"sentrum": 30, "trygghet": 55}),
    "stille tradisjonalist":         ("Sp", {"sentrum": 32, "trygghet": 48, "verdi": -15, "tillit": -8}),
    "trygg midtstrøms":              ("_alle", {"sentrum": 50, "trygghet": 72, "tillit": 5}),
    "urolig midtstrøms":             ("_alle", {"sentrum": 48, "trygghet": 32}),
    "politisk fremmedgjort":         ("stemte ikke", {"sentrum": 48, "trygghet": 42, "tillit": -35}),
    "ung uavklart":                  ("stemte ikke", {"sentrum": 52, "trygghet": 45, "verdi": 5}),
}
_ESS_AKSENAVN = {"oko": "oko_fordeling", "innv": "innvandring",
                 "klima": "klima", "tillit": "institusjonstillit",
                 "verdi": "verdiliberal"}


def bygg_grupper_fra_ess(ess: dict) -> dict[str, tuple[float, ...]]:
    """Prototyper på (øko, innv, klima, sentrum, trygghet, tillit, verdi)
    forankret i partienes MÅLTE posisjoner."""
    ut = {}
    for navn, (anker, avvik) in _GRUPPE_SPEC.items():
        vec = []
        for kort in ["oko", "innv", "klima"]:
            akse = ess["akser"][_ESS_AKSENAVN[kort]]
            basis = akse.get(anker, akse["_alle"])["mean"]
            vec.append(float(np.clip(basis + avvik.get(kort, 0.0), 0, 100)))
        vec.append(float(avvik["sentrum"]))
        vec.append(float(avvik["trygghet"]))
        for kort in ["tillit", "verdi"]:
            akse = ess["akser"][_ESS_AKSENAVN[kort]]
            basis = akse.get(anker, akse["_alle"])["mean"]
            vec.append(float(np.clip(basis + avvik.get(kort, 0.0), 0, 100)))
        ut[navn] = tuple(vec)
    return ut


# Grovnivået: hver av de 20 gruppene hører til én av fem hovedgrupper.
HOVEDGRUPPE: dict[str, str] = {
    "systemkritisk venstre":        "venstresiden",
    "rød aktivist":                 "venstresiden",
    "sosialdemokratisk kjerne":     "venstresiden",
    "tradisjonell arbeiderklasse":  "venstresiden",
    "bekymret velferdsvelger":      "venstresiden",
    "nasjonalkonservativ":          "høyresiden",
    "systemkritisk protest":        "høyresiden",
    "trygghetssøkende konservativ": "høyresiden",
    "markedsliberal":               "høyresiden",
    "etablert konservativ":         "høyresiden",
    "grønn urban":                  "grønne og liberale",
    "trygg kosmopolitt":            "grønne og liberale",
    "utdannet sentrumsliberal":     "grønne og liberale",
    "distriktsforankret":           "sentrum og tradisjon",
    "kristenkonservativ":           "sentrum og tradisjon",
    "stille tradisjonalist":        "sentrum og tradisjon",
    "trygg midtstrøms":             "midten",
    "urolig midtstrøms":            "midten",
    "politisk fremmedgjort":        "midten",
    "ung uavklart":                 "midten",
}
HOVEDGRUPPER = ["venstresiden", "høyresiden", "grønne og liberale",
                "sentrum og tradisjon", "midten"]

_KORRESPONDANSE_URL = ("https://data.ssb.no/api/klass/v1/"
                       "correspondencetables/1417.json")


def hent_sentralitet() -> dict[str, int]:
    """kommune -> sentralitetsklasse 1 (mest sentral) .. 6 (minst).

    Klass 128 (Sentralitet 2020) mot kommuneinndeling 2024. Kommuner som
    mangler (nyere kodeendringer) får klasse 4 hos kalleren — logget der.
    """
    r = requests.get(_KORRESPONDANSE_URL, timeout=30)
    r.raise_for_status()
    ut = {}
    for item in r.json()["correspondenceMaps"]:
        ut[item["targetCode"]] = int(item["sourceCode"])
    logger.info("Klass 128: sentralitetsklasse for %d kommuner.", len(ut))
    return ut


def tildel_holdning(df: pd.DataFrame, *, seed: int = 0,
                    sentralitet: dict[str, int] | None = None,
                    eu1994: dict[str, float] | None = None) -> pd.DataFrame:
    """Legg på aksekolonnene og ``holdningsgruppe``. Deterministisk gitt seed.

    ``eu1994`` (kommune -> nei-andel 1994, fra kopi/eu1994.py) gir sentrum/
    distrikt-aksen målt geografi; uten den gjelder sentralitetsantakelsen.
    Sendes inn eksplisitt (hentes ikke her) så testene forblir offline."""
    sentralitet = sentralitet if sentralitet is not None else hent_sentralitet()
    n = len(df)
    rng = np.random.default_rng(seed)

    ess = les_posisjoner()

    # 1. Parti-basis: MÅLT (ESS) når aggregatene finnes, ellers antakelsene.
    kats = list(df["parti"].cat.categories)
    koder = df["parti"].cat.codes.to_numpy()
    if ess:
        pos = np.zeros((len(kats), 4))
        sd_parti = np.zeros((len(kats), 4))
        for i, p in enumerate(kats):
            nokkel = "_alle" if p == "ikke stemmerett" else p
            for j, akse in enumerate(["oko_fordeling", "innvandring", "klima"]):
                rad = ess["akser"][akse]
                v = rad.get(nokkel, rad["_alle"])
                pos[i, j] = v["mean"]
                sd_parti[i, j] = v["sd"]
            pos[i, 3] = PARTI_POSISJON[p][3]              # sentrum: antakelse
            sd_parti[i, 3] = (_STOY_SVAK if p in ("stemte ikke", "ikke stemmerett")
                              else _STOY)
        sd_person = sd_parti[koder]
        logger.info("Partiposisjoner og spredning på øko/innvandring/klima: "
                    "MÅLT fra ESS (%s).", ess["kilde"])
    else:
        pos = np.array([PARTI_POSISJON[p] for p in kats])
        sd_person = None
        logger.info("ESS-aggregater ikke funnet — antatte partiposisjoner.")
    skar = pos[koder].astype(np.float64)                  # (n, 4)

    # 2. Inntekt skyver økonomiaksen: målt desilgradient (ESS) eller ±12.
    inntekt = df["brutto_inntekt"].to_numpy()
    har = ~np.isnan(inntekt)
    pct = np.full(n, 0.5)
    pct[har] = pd.Series(inntekt[har]).rank(pct=True).to_numpy()
    if ess:
        dev = np.array(ess["oko_inntektsdesil"], dtype=float)
        desil = np.clip((pct * 10).astype(int), 0, 9)
        skar[:, 0] += dev[desil]
    else:
        skar[:, 0] += (pct - 0.5) * 2.0 * _INNTEKT_SKYV

    # 3. Bosted og region. Med ESS-aggregater: målt domicil-gradient (innen
    # parti) per akse for øko/innvandring/klima + målt NUTS2-residual; den
    # antatte sentralitetsvektoren gjelder da bare sentrum/distrikt-aksen.
    # Uten: hele den antatte vektoren, som før.
    klasser = df["kommune"].cat.categories.map(
        lambda k: sentralitet.get(str(k), 0)).to_numpy()
    mangler = [str(k) for k, s in zip(df["kommune"].cat.categories, klasser)
               if s == 0]
    if mangler:
        logger.warning("Sentralitet mangler for %d kommuner (får midtklasse 4, "
                       "logget): %s", len(mangler), mangler)
    klasser = np.where(klasser == 0, 4, klasser)
    kl_person = klasser[df["kommune"].cat.codes].astype(np.float64)
    faktor = (3.5 - kl_person) / 2.5                          # +1 .. -1
    if ess and "geo" in ess:
        # Sentralitetsklasse 1..6 -> domicil-posisjon 1..5 (lineær mapping —
        # ANTAKELSE; selve gradienten per posisjon er målt).
        pos = 1.0 + (kl_person - 1.0) * (4.0 / 5.0)
        i0 = np.clip(np.floor(pos).astype(int), 1, 4)
        frac = pos - i0
        nuts = df["kommune"].cat.categories.map(
            lambda k: _FYLKE_NUTS2.get(str(k)[:2], "")).to_numpy()
        kom_koder = df["kommune"].cat.codes.to_numpy()
        for j, akse in enumerate(["oko_fordeling", "innvandring", "klima"]):
            grad = np.asarray(ess["geo"]["bosted"][akse], dtype=np.float64)
            skar[:, j] += grad[i0 - 1] * (1.0 - frac) + grad[i0] * frac
            reg = ess["geo"]["region"][akse]
            per_kommune = np.array([reg.get(r, 0.0) for r in nuts])
            skar[:, j] += per_kommune[kom_koder]
        logger.info("Bosteds- og regiongradienter på øko/innvandring/klima: "
                    "MÅLT fra ESS (domicil + NUTS2, innen parti).")
    else:
        skar[:, :3] += faktor[:, None] * _SENTRALITET_SKYV[None, :3]

    # Sentrum/distrikt: EU-1994-anker (målt geografi) når det er sendt inn,
    # ellers sentralitetsantakelsen.
    if eu1994:
        nei = df["kommune"].cat.categories.map(
            lambda k: eu1994.get(str(k), _EU_NASJONAL)).to_numpy(dtype=float)
        uten = int((~df["kommune"].cat.categories.isin(eu1994)).sum())
        if uten:
            logger.warning("EU-1994 mangler for %d kommuner — de får "
                           "nasjonal nei-andel (nøytralt skyv).", uten)
        skar[:, 3] += (-_EU_SKALA * (nei - _EU_NASJONAL) * 100.0)[
            df["kommune"].cat.codes]
        logger.info("Sentrum/distrikt-aksen ankret i EU-1994 (målt nei-andel "
                    "per kommune, antatt skala %.2f).", _EU_SKALA)
    else:
        skar[:, 3] += faktor * _SENTRALITET_SKYV[3]

    # 4. Spredning: målt innen-parti-sd (ESS) på de målte aksene; ellers
    # antatt flat støy (svakere signal for dem uten avgitt stemme).
    svak = df["parti"].isin(["stemte ikke", "ikke stemmerett"]).to_numpy()
    if sd_person is not None:
        skar += rng.normal(0.0, 1.0, size=(n, 4)) * sd_person
    else:
        sd = np.where(svak, _STOY_SVAK, _STOY)
        skar += rng.normal(0.0, 1.0, size=(n, 4)) * sd[:, None]
    skar = np.clip(skar, 0.0, 100.0)

    # 5. Nærmeste gruppe-prototyp på sju akser (de fire over + trygghet,
    # institusjonstillit og verdiliberal fra egne lag). Under 16: ingen.
    ekstra = ["trygghet", "institusjonstillit", "verdiliberal"]
    for kol in ekstra:
        if kol not in df.columns:
            raise ValueError(f"Kolonnen '{kol}' mangler — kjør kriminalitets- "
                             "og tillit/verdi-laget før tildel_holdning().")
    fler = [np.nan_to_num(df[k].to_numpy(dtype=np.float64), nan=50.0)
            for k in ekstra]
    syv = np.column_stack([skar] + fler)
    grupper = bygg_grupper_fra_ess(ess) if ess else GRUPPER
    proto = np.array(list(grupper.values()))                  # (G, 7)
    avstand = ((syv[:, None, :] - proto[None, :, :]) ** 2).sum(axis=2)
    gruppe = avstand.argmin(axis=1).astype(np.int8)

    barn = (df["alder"] < 16).to_numpy()
    skar[barn] = np.nan
    gruppe_kode = np.where(barn, -1, gruppe).astype(np.int8)

    ut = df.copy()
    for i, akse in enumerate(AKSER):
        ut[akse] = np.round(skar[:, i], 1).astype(np.float32)
    ut["holdningsgruppe"] = pd.Categorical.from_codes(
        gruppe_kode, categories=list(grupper))
    hoved_i = {navn: HOVEDGRUPPER.index(HOVEDGRUPPE[navn])
               for navn in grupper}
    hoved_kode = np.where(gruppe_kode < 0, -1, np.array(
        [hoved_i[navn] for navn in grupper])[gruppe_kode]).astype(np.int8)
    ut["hovedgruppe"] = pd.Categorical.from_codes(
        hoved_kode, categories=HOVEDGRUPPER)
    logger.info("Holdning tildelt: %d personer 16+ i %d grupper, %d barn "
                "uten (ren antakelse fra parti/inntekt/sentralitet).",
                int((~barn).sum()), len(GRUPPER), int(barn.sum()))
    return ut
