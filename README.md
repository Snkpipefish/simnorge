# SimNorge — syntetisk meningsmålingsmaskin for Norge

En digital tvilling av Norges befolkning (16+), bygget kommune for kommune fra
åpen SSB-data, med en interaktiv spørretab: still et spørsmål, få et
representativt vektet svar for hele landet eller én kommune. Inspirert av
[Sim Francisco](https://github.com/tejasprabhune/simfrancisco).
`PLAN.md` er kilde til sannhet for design og metodikk.

## Kom i gang

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

LLM-kall går via `claude`-CLI (må være installert og innlogget). Alle SSB-kall
og LLM-svar diskcaches i `.cache/` — gjentatte kjøringer er billige/gratis.

## Kjørerekkefølge

Modulene bygger på hverandre (Modul 1 → 8); alt hentes/bygges på forespørsel,
så du kan starte hvor som helst:

| Hva | Kommando |
|---|---|
| Spørretaben (hovedfunksjonen) | `.venv/bin/python -m app.server 8765` → http://localhost:8765 |
| Valg-backtest, LLM-motor vs baseliner | `.venv/bin/python run_engine_backtest.py [bygg] [mål]` (standard 2021 2025) |
| Økologisk backtest (null LLM, alle kommuner) | `.venv/bin/python run_ecological_backtest.py [bygg] [mål]` |
| Nasjonalt holdningskart | `.venv/bin/python run_national_map.py` |
| Tester (én modul) | `.venv/bin/python -m tests.test_module1_facit` osv. |

Testene i `tests/` er nummerert per modul; de fleste går mot live SSB (cachet).
Første LLM-kjøring for en ny (spørsmål × geografi) tar ~1 min; deretter cache.

## Arkitektur (moduler)

1. **`data/`** — SSB-klient (json-stat2, diskcache), Klass-normalisering av
   kommunekoder på tvers av årganger (sammenslåing aggregeres, splitt droppes
   og logges — aldri gjettes), tabellregister i tre lag: struktur (per
   kommune), holdning (nasjonalt × demografi), fasit (per kommune).
2. **`population/`** — syntetisk befolkning per kommune via IPF på
   kjønn × alder × utdanning × økonomisk status (+ arbeidsmarkedsstatus),
   marginaler verifisert mot SSB.
3. **`enrich/`** — holdningsberikelse: hver person trekker holdningsnivåer fra
   sin demografiske gruppes *fordeling* (nasjonale tabeller), med spredning.
4. **`personas/`** — vektede personas som bærer fordelinger (ikke punkt);
   kommune er hard nøkkel; aggregat + spredning bevart.
5. **`engine/`** — LLM-spørremotor (claude-CLI): ekspanderer personas til
   pollings-instanser, spør per celle, parser trygt, aggregerer vektet.
6. **`aggregate/` + `calibration/`** — valg-backtest med tre motorvarianter
   (stedløs / geo uten navn / med kommunenavn) mot persistence- og
   nasjonalsnitt-baseliner; skiller resonnement fra memorering.
7. **`calibration/ecological.py`** — økologisk inferens: celle-stereotyper
   lært fra alle ~355 kommuner (null LLM). `correction.py`: affin
   kalibreringskorreksjon av LLM-skjevhet lært på backtest-residualer.
   `turnout.py`: valgdeltakelsesvekting per kjønn × utdanning (13360).
8. **`app/`** — spørretaben med ærlighetsbannere per spørsmålsdomene.

## Ærlige begrensninger (les før du stoler på tall)

- **Ekstrapolering, ikke måling.** Kommunale holdninger avledes fra nasjonale
  demografi-holdningskoblinger + kommunens sammensetning. SSB måler ikke
  holdninger per kommune; det gjør heller ikke vi.
- **Tre stablede uavhengighetsantakelser** multipliseres: (1) holdningsakser
  antas betinget uavhengige gitt demografi, (2) demografiakser kobles via IPF
  med antatt nasjonal interaksjonsstruktur, (3) LLM-ens celle-svar antas
  gyldige på tvers av geografi. Feilene komponerer — se `enrich/attitudes.py`.
- **Demografi forklarer ~null geografisk variasjon i livskvalitet**
  (validert mot 13798/13799): politikk-domenet er geografisk validert via
  valg-backtesten, livskvalitetsdomenet er i praksis flatt geografisk.
  Spørretaben viser dette skillet som ærlighetsbannere.
- **Valg-backtest 2021→2025** (8 prøvekommuner, 9-parti-basis): beste
  motorvariant (geo uten navn) MAE ≈ 3,7 pp mot persistence 3,9 pp — og
  navn-varianten er *svakere* enn geo-uten-navn, så gevinsten er resonnement,
  ikke memorering. Med kommune-CV-lært affin korreksjon (`correction.py`)
  faller MAE til ≈ 2,0 pp; korreksjonen viser at LLM-en systematisk demper
  variasjon mot midten (b > 1 for de fleste partier).
- **Korreksjonen gjelder KUN valg.** Den er lært på valgfasit og skal aldri
  brukes som skjult presisjon på frie spørsmål i spørretaben.
- **Deltakelsesvekting (13360 + 13085)** dekker både utdannings- og alders-
  gradienten (kjønn × aldersbånd × utdanning, multiplikativt kombinert —
  dokumentert uavhengighetsantakelse; 16–17-åringer teller med deltakelse 0).
  Målingen viste likevel at den gjør LLM-motoren marginalt svakere
  (3,71 → 3,81 pp; LLM-svarene er allerede implisitt deltakelsesvektede, så
  eksplisitt vekting dobbelteller). Av som standard i motoren
  (`turnout_weighting=False`); i den økologiske modellen brukes den fordi
  stereotypene da beskriver velgere, ikke bosatte.
- **LLM-ens selvrapporterte `sikkerhet` rapporteres, men vektes aldri inn**:
  selvrapportert sikkerhet er ukalibrert, og nedvekting av usikre celler ville
  stille skjevfordele representativiteten. Spørretaben viser vektet snitt og
  andel lav-sikkerhet i dekningslinjen.
- **Små kommuner** gir < 24 demografiske celler → bredere usikkerhet i
  spørretaben.
