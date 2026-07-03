"""Kategoriaksene for den syntetiske befolkningen — Modul 2.

Befolkningen bygges for personer 16 år og over (den holdnings- og
valgrelevante delen — barn poller vi ikke). Aldersbåndene er bevisst valgt
likt SSBs holdningstabell 13834 (16-24/25-44/45-66/67+), slik at Modul 3
(holdningsberikelse) kan slå opp rett inn uten ny binning.
"""

from __future__ import annotations

# Akse 0 — kjønn (rekkefølge = indeks i kontingenstabellen)
SEX_CODES = {"1": "mann", "2": "kvinne"}
SEX_LABELS = ["mann", "kvinne"]

# Akse 1 — aldersgruppe (16+), lik 13834
AGE_LABELS = ["16-24", "25-44", "45-66", "67+"]

# Akse 2 — utdanningsnivå (09429 Nivaa-koder -> etikett)
EDU_CODES = {
    "01": "grunnskole",
    "02a": "videregaaende",
    "11": "fagskole",
    "03a": "uh_kort",
    "04a": "uh_lang",
    "09a": "uoppgitt",
}
EDU_LABELS = ["grunnskole", "videregaaende", "fagskole", "uh_kort", "uh_lang", "uoppgitt"]

# Akse 3 — økonomisk status (avledet fra 12944 lavinntektsandel)
ECON_LABELS = ["lavinntekt", "ovrig"]

# Akse 5 (post-IPF) — arbeidsmarkedsstatus fra 13563 (prioritert partisjon:
# hver bosatt telles i nøyaktig én status). Kategoriene speiler SSBs
# holdningstabellers OkonomiskStatus-dimensjon (13839/13793/13831), slik at
# Modul 3 kan koble holdning på status DIREKTE — det er hele poenget.
STATUS_LABELS = ["yrkesaktiv", "arbeidsledig", "student", "ufor", "pensjonist", "annet"]
# Vår etikett -> OkonomiskStatus-kode i 13839/13793/13831.
STATUS_TO_OKSTATUS = {
    "yrkesaktiv": "11", "arbeidsledig": "20", "student": "31",
    "ufor": "52", "pensjonist": "42", "annet": "98b",
}

# Tabellens form (kjønn × alder × utdanning × økonomisk status)
SHAPE = (len(SEX_LABELS), len(AGE_LABELS), len(EDU_LABELS), len(ECON_LABELS))


def age_band(age: int) -> str | None:
    """1-årig alder -> aldersbånd (16+). None for under 16."""
    if age < 16:
        return None
    if age < 25:
        return "16-24"
    if age < 45:
        return "25-44"
    if age < 67:
        return "45-66"
    return "67+"
