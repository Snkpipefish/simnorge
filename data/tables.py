"""Register over SSB-tabeller SimNorge bruker, gruppert i de tre lagene
fra PLAN.md.

- STRUKTUR: finnes per kommune -> bygger personaene (hvem bor hvor).
- HOLDNING: kun nasjonalt, brutt på demografi -> gir personaene meninger.
- FASIT:   per kommune -> eneste geografiske holdningsmål (kalibrering).

Den avgjørende innsikten: kommunale holdninger EKSTRAPOLERES ved å koble
strukturlaget (per kommune) til holdningslaget (nasjonalt × demografi). De
måles ikke direkte. Skillet under gjør det eksplisitt i koden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Layer(str, Enum):
    STRUKTUR = "struktur"   # per kommune
    HOLDNING = "holdning"   # nasjonalt × demografi
    FASIT = "fasit"         # per kommune (kalibrering)


@dataclass(frozen=True)
class Table:
    id: str
    layer: Layer
    title: str
    note: str = ""


TABLES: dict[str, Table] = {t.id: t for t in [
    # --- Strukturlag (per kommune) ---------------------------------------
    Table("07459", Layer.STRUKTUR, "Befolkning etter region, kjønn, alder"),
    Table("09429", Layer.STRUKTUR, "Personer 16+ etter utdanningsnivå, kjønn"),
    Table("06944", Layer.STRUKTUR, "Husholdningsinntekt (median) etter region, husholdningstype"),
    Table("12944", Layer.STRUKTUR, "Personer i husholdninger med vedvarende lavinntekt"),
    Table("03797", Layer.STRUKTUR, "Formuesposter fra selvangivelsen"),

    # --- Holdningslag (nasjonalt × demografi) ----------------------------
    Table("13834", Layer.HOLDNING, "Tillit (myndigheter, media, personer)", "brutt på alder, kjønn"),
    Table("13839", Layer.HOLDNING, "Tillit", "brutt på økonomisk status"),
    Table("13790", Layer.HOLDNING, "Livskvalitet: mening og mestring", "brutt på kjønn, alder"),
    Table("13792", Layer.HOLDNING, "Livskvalitet", "brutt på inntektsgruppe"),
    Table("13793", Layer.HOLDNING, "Livskvalitet", "brutt på økonomisk status"),
    Table("13831", Layer.HOLDNING, "Frivillig innsats siste 12 mnd", "brutt på økonomisk status"),

    # --- Fasit-/kalibreringslag (per kommune) ----------------------------
    Table("08092", Layer.FASIT, "Stortingsvalg: godkjente stemmer etter parti", "siste år 2025"),
    Table("01180", Layer.FASIT, "Kommunestyrevalg: godkjente stemmer etter parti", "siste år 2023"),
    Table("13360", Layer.FASIT, "Valgdeltakelse % etter region, kjønn, utdanning", "siste år 2025"),
]}


def by_layer(layer: Layer) -> list[Table]:
    return [t for t in TABLES.values() if t.layer == layer]


# Partikoder for valgtabellene 08092 / 01180.
PARTY_CODES: dict[str, str] = {
    "01": "Ap", "02": "FrP", "03": "Høyre", "04": "KrF",
    "05": "Sp", "06": "SV", "07": "Venstre", "08": "MDG", "55": "Rødt",
}

# De ni hovedpartiene vi backtester mot (rekkefølge fra PLAN.md-eksempelet).
MAIN_PARTY_CODES: list[str] = ["01", "02", "03", "04", "08", "55", "05", "06", "07"]
