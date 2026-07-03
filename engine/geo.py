"""Geografisk kontekst UTEN kommunenavn — delt av spørremotor og backtest.

Modul 6-backtesten målte at sentralitet + kommunens inntektsnivå (uten navn)
løfter mest (MAE 3.79 mot 4.11 med navn, 4.72 uten geografi), og skalerer
billig: unike prompt-celler følger kontekst-nivåer, ikke antall kommuner.
Derfor er dette den kanoniske geo-konteksten for ALLE persona-prompter.
Kommunenavn holdes bevisst utenfor prompten — navn åpner for memorering og
ga empirisk svakere geografi-resonnement.
"""

from __future__ import annotations

from data import SSBClient, KlassClient

SENTRALITET_TABLE_2024 = "1417"          # Sentralitet 2020 - Kommuneinndeling 2024
SENT_MARKER = {
    "01": "et sentralt strøk (storbyområde)", "02": "et sentralt strøk (by)",
    "03": "et mellomsentralt strøk", "04": "et mellomsentralt strøk",
    "05": "et lite sentralt distrikt", "06": "et grisgrendt distrikt",
}


def load_centrality(klass: KlassClient) -> dict[str, str]:
    """kommune-kode -> sentralitetskode '01'..'06' (2024-vintage)."""
    df = klass.correspondence(SENTRALITET_TABLE_2024)
    return {r["target_code"]: r["source_code"] for _, r in df.iterrows()}


def national_median_income(ssb: SSBClient) -> float:
    df = ssb.fetch("06944", {"Region": ["0"], "HusholdType": ["0000"],
                             "ContentsCode": ["SamletInntekt"],
                             "Tid": [ssb.variable_codes("06944")["Tid"][-1]]})
    return float(df.iloc[0]["value"])


def income_tier(median: float | None, national: float) -> str:
    if median is None or national <= 0:
        return "ukjent inntektsnivå"
    r = median / national
    if r >= 1.12:
        return "godt over landssnittet i inntekt"
    if r >= 1.03:
        return "over landssnittet i inntekt"
    if r >= 0.95:
        return "rundt landssnittet i inntekt"
    if r >= 0.88:
        return "under landssnittet i inntekt"
    return "godt under landssnittet i inntekt"


def lowinc_tier(share: float | None) -> str:
    if share is None:
        return "ukjent lavinntektsandel"
    if share >= 0.13:
        return "høy andel lavinntekt"
    if share >= 0.09:
        return "middels andel lavinntekt"
    return "lav andel lavinntekt"


def geo_line(sentralitet_kode: str | None, median_income: float | None,
             lavinntekt_andel: float | None, national_median: float) -> str:
    """Én linje geo-kontekst uten navn, lik geo_no_name-varianten i backtesten."""
    sted = SENT_MARKER.get(sentralitet_kode or "", "et sted i Norge")
    return (f"Bosted: {sted}. Kommunens økonomi: "
            f"{income_tier(median_income, national_median)}, "
            f"{lowinc_tier(lavinntekt_andel)}.")


class GeoContext:
    """Bygg geo-linjer per kommune for spørremotoren.

    ``line_for("0")`` (hele Norge) gir None — nasjonale poller skal ikke ha et
    fiktivt bosted."""

    def __init__(self, ssb: SSBClient, klass: KlassClient) -> None:
        self._sent = load_centrality(klass)
        self._national = national_median_income(ssb)

    def line_for(self, kommune: str, median_income: float | None,
                 lavinntekt_andel: float | None) -> str | None:
        if kommune == "0":
            return None
        return geo_line(self._sent.get(kommune), median_income,
                        lavinntekt_andel, self._national)
