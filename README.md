# SimNorge — syntetisk meningsmålingsmaskin for Norge

En digital tvilling av Norges befolkning (16+), bygget kommune for kommune fra
åpen SSB-data, med en interaktiv spørretab: still et spørsmål, få et
representativt vektet svar for hele landet eller én kommune. Inspirert av
[Sim Francisco](https://github.com/tejasprabhune/simfrancisco).
`PLAN.md` er kilde til sannhet for design og metodikk.

> **Status 2026-07-03:** LLM-pipelinen (Modul 3–8) er avsluttet — se
> `RAPPORT.md` for post mortem med tall. Aktivt spor er nå **kopien på
> individnivå** under.

## KOPI — Norge på individnivå (aktivt spor)

`kopi/` bygger en syntetisk kopi av hele Norges befolkning: én rad per bosatt
innbygger (~5,63 mill) med kommune, kjønn, alder, bruttoinntekt,
personlighetstype (16 typer) og stemmegivning ved stortingsvalget 2025.

```bash
.venv/bin/python run_kopi.py                 # hele landet -> kopi_norge.parquet
.venv/bin/python run_holdningsgrupper.py     # + holdningsakser og holdningsgruppe
.venv/bin/python -m tests.test_kopi          # tester (offline)
```

Ærlighetsregime per kolonne: befolkning (07459) og valgresultat per kommune
(08092) er **ekte tall** — kommuneaggregatet av simulerte stemmer matcher det
faktiske resultatet (verifiseres ved hver bygging). Deltakelse (13085) og
demografisk stemmetilt (13698) er **målte gradienter**. Inntekt trekkes fra
SSBs **faktiske fordeling** (06655) med en antatt kommuneskala (06944).
Personlighetstypene har **antatte basissatser** (kjønn/alder, dokumentert i
`kopi/personlighet.py`). Med ESS-mikrodata (se under) tiltes bokstav-
sannsynlighetene i tillegg av **målte** gruppeforskjeller i inntekt,
utdanning og parti (Schwartz-verdier + sosial aktivitet som proxy —
koblingen komposit→bokstav er antakelse, gradientene er målte). Uten ESS
er typene uavhengige av inntekt/parti — vi fabrikkerer ikke korrelasjoner
uten kilde.

Kriminalitetslaget (`kopi/kriminalitet.py`) trekker per person tre flagg fra
**målte rater**: utsatt for vold/trusler og tyveri/skadeverk (04621, kjønn ×
alder, skalert med kommunens faktiske anmeldelsesnivå fra 08487) og siktet
for vold (11453). Av dette bygges aksen **trygghet** (målte ingredienser,
antatte vekter).

Utdanningslaget (`kopi/utdanning.py`) gir hver person 16+ ett av fire
utdanningsnivåer fra tre **målte** kilder: kommunens faktiske fordeling per
kjønn (09429, hard marginal som holdes eksakt via raking), den nasjonale
aldersgradienten (08921) og velgerundersøkelsens utdanning–parti-sammenheng
(13555). Livssynslaget (`kopi/livssyn.py`) trekker Dnk-medlemskap fra
kommunens **faktiske** medlemsrate (KOSTRA 12025, antatt aldersgradient).

Tillit/verdi-laget (`kopi/tillit_verdi.py`) gir aksen **institusjonstillit**
fra velgerundersøkelsens målte tillit til Stortinget etter alder ×
stemt/ikke stemt × utdanning (13908), og aksen **verdiliberal** (antakelse:
parti + kirkemedlemskap + alder + utdanning).

Tilbøyelighetslaget (`kopi/tilboyelighet.py`, kjøres med
`run_tilboyeligheter.py`) gir fem skårer 0–100 per person 16+: stemme
(målt: 10440 deltakelse etter alder × utdanning + vane/tillit), flytte
(målt: 05540 per kjønn × alder), frivillig innsats (målt basis 13826 +
antatte tillegg), protest og risiko (rene antakelser fra egne kolonner).

Holdningslaget (`kopi/holdning.py`) gir hver person 16+ skårer på sju akser
(økonomisk fordeling, innvandring, klima, sentrum–distrikt, trygghet,
institusjonstillit, verdiliberal) og en av **20 holdningsgrupper** (rullet
opp i 5 hovedgrupper). Gruppeprototypene står samlet i den fila og kan
justeres på ett sted.

**ESS-forankring** (`kopi/ess.py`): med European Social Survey-mikrodata
(runde 10+11, norsk utvalg, lastes ned gratis fra ess.sikt.no til
`data/ess/`) erstattes de *antatte* partiposisjonene med **målte**: posisjon
og spredning innen hvert parti på økonomi/innvandring/klima, målt
inntektsgradient, målte tillits- og verdiposisjoner (inkl. hjemmesittere),
og gruppeprototyper ankret i partienes målte posisjon. Kjør
`.venv/bin/python -m kopi.ess` for å (re)bygge aggregatene
(`kopi/ess_posisjoner.json`); uten dem gjelder de dokumenterte antakelsene.
Kun sentrum–distrikt-aksen forblir antakelse (ESS mangler mål for den).

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
  ikke memorering. Affin korreksjon (`correction.py`) løfter motoren til
  ≈ 2,0 pp og avdekker at LLM-en demper variasjon mot midten (b > 1).
  **MEN — den rettferdige sammenlikningen:** samme korreksjon på persistence
  gir **1,45 pp**. «Forrige valg + partivis sving» (pollofpolls-metoden, som
  allerede finnes) slår altså hele LLM-motoren når begge korrigeres. For
  valgprediksjon tilfører LLM-en per i dag negativ verdi mot enkleste
  eksisterende metode; modellens eventuelle verdi ligger i frie spørsmål —
  som er uvalidert.
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
