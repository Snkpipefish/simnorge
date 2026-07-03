"""Prompt-bygging for persona-polling — Modul 5.

Prompten er bygget mot de to fellene brukeren pekte på:

FELLE 1 — modellen overstyrer prioren. Vi brukte tre moduler på å forankre
holdningsprioren i SSB-data. Personaens KONKRETE holdningsrealisering legges
derfor inn som eksplisitt, ikke-forhandlingsbar kontekst modellen skal resonnere
FRA — ikke fra sine egne forestillinger om «folk i Oslo».

FELLE 2 — modellen demper ytterpunkter mot midten. Da mister aggregatet halene
som finnes i befolkningen. Prompten instruerer eksplisitt om å bære posisjonen
fullt ut, også når den er ytterliggående, og å IKKE moderere mot et forsiktig
midtpunkt. Det er hele grunnen til at vi trakk fra en fordeling, ikke et snitt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuestionSpec:
    """Et spørsmål + skala + hvilken holdningsakse personaens disposisjon hentes
    fra (for å feste den konkrete realiseringen i prompten)."""

    question: str
    scale_min: int
    scale_max: int
    scale_low_label: str             # hva lav ende betyr (f.eks. «stoler ikke i det hele tatt»)
    scale_high_label: str            # hva høy ende betyr
    focus_item: str                  # holdningsmål-kolonne disposisjonen kommer fra
    level_phrasing: dict[str, str]   # nivå -> norsk frase, f.eks. {"lav": "LAV tillit ..."}


AGE_TEXT = {"16-24": "16–24 år", "25-44": "25–44 år", "45-66": "45–66 år", "67+": "67 år eller eldre"}
EDU_TEXT = {
    "grunnskole": "grunnskole", "videregaaende": "videregående skole",
    "fagskole": "fagskole", "uh_kort": "kort universitets-/høgskoleutdanning",
    "uh_lang": "lang universitets-/høgskoleutdanning", "uoppgitt": "uoppgitt utdanning",
}
ECON_TEXT = {"lavinntekt": "lav husholdningsinntekt", "ovrig": "vanlig/høyere husholdningsinntekt",
             "ukjent": "uoppgitt inntektsnivå"}


def build_prompt(geo_line: str | None, demographics: dict, spec: QuestionSpec,
                 disposition_level: str) -> str:
    """``geo_line`` er geo-kontekst UTEN kommunenavn (engine.geo), eller None
    for nasjonale poller. Navn holdes utenfor: Modul 6-backtesten målte at
    navn ga svakere geografi-resonnement (memoreringsrisiko) og dyrere
    skalering enn sentralitet + inntektsnivå."""
    kjonn = "kvinne" if demographics.get("kjonn") == "kvinne" else "mann"
    alder = AGE_TEXT.get(demographics.get("aldersgruppe", ""), demographics.get("aldersgruppe", ""))
    utd = EDU_TEXT.get(demographics.get("utdanning", ""), demographics.get("utdanning", ""))
    okon = ECON_TEXT.get(demographics.get("okonomisk_status", ""), "")
    disposition = spec.level_phrasing[disposition_level]
    bosted = f"\n{geo_line}" if geo_line else ""

    return f"""Du svarer i rollen som én innbygger i Norge.

Demografi (fra registerdata): {kjonn}, {alder}, {utd}, {okon}.{bosted}

Din etablerte holdning: {disposition}
Dette er forankret i offentlig statistikk for personer som deg — det er DITT
utgangspunkt, ikke noe du skal overstyre med generelle antakelser. Resonner FRA
denne holdningen.

Spørsmål: {spec.question}
Skala {spec.scale_min}–{spec.scale_max}, der {spec.scale_min} = «{spec.scale_low_label}»
og {spec.scale_max} = «{spec.scale_high_label}».

VIKTIG: Bær holdningen din fullt ut. Hvis den er sterk eller ytterliggående,
svar deretter — IKKE demp mot et forsiktig midtpunkt. Ekte mennesker har klare
standpunkter; et trygt midtsvar forfalsker fordelingen i befolkningen.

Svar KUN med JSON, uten noe annet tekst:
{{"posisjon": <heltall {spec.scale_min}-{spec.scale_max}>, "begrunnelse": "<maks 15 ord>", "sikkerhet": <0.0-1.0>}}"""


# --------------------------------------------------------------------------- #
# Ferdige spørsmål for testene                                                #
# --------------------------------------------------------------------------- #
# Konsistens-spørsmål: tillit til politiet (matcher SSB 13834 tillit_politiet).
TRUST_POLICE = QuestionSpec(
    question="Hvor mye stoler du på politiet?",
    scale_min=0, scale_max=10,
    scale_low_label="stoler ikke i det hele tatt",
    scale_high_label="stoler fullstendig",
    focus_item="tillit_politiet",
    level_phrasing={
        "lav": "LAV tillit til politiet.",
        "middels": "MIDDELS tillit til politiet.",
        "hoy": "HØY tillit til politiet.",
    },
)

# Konformitets-spørsmål: tillit til det politiske systemet — mer DELT i
# befolkningen, så vi kan se om halene overlever (matcher SSB 13834 nr. 04).
TRUST_POLITICAL_SYSTEM = QuestionSpec(
    question="Hvor mye stoler du på det politiske systemet i Norge?",
    scale_min=0, scale_max=10,
    scale_low_label="stoler ikke i det hele tatt",
    scale_high_label="stoler fullstendig",
    focus_item="tillit_politisk_system",
    level_phrasing={
        "lav": "LAV tillit til det politiske systemet.",
        "middels": "MIDDELS tillit til det politiske systemet.",
        "hoy": "HØY tillit til det politiske systemet.",
    },
)
