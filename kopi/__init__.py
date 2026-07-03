"""SimNorge KOPI — en syntetisk kopi av Norge på individnivå.

Én rad per bosatt innbygger (~5,6 mill), med kommune, kjønn, alder, inntekt,
personlighetstype (16 typer, antakelsesbasert) og stemmegivning (forankret i
faktiske valgresultater per kommune). Ingen LLM, ingen personas — bare et
stort, grupperbart datasett.

Gjenbruker KUN datalaget fra det gamle prosjektet (SSB-klient + Klass).
"""

from .bygg import bygg_kopi
from .personlighet import tildel_personlighet, TYPER
from .inntekt import InntektsModell
from .valg import ValgModell

__all__ = ["bygg_kopi", "tildel_personlighet", "TYPER",
           "InntektsModell", "ValgModell"]
