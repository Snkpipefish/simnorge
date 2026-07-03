# SimNorge — syntetisk meningsmålingsmaskin for Norge

## Hva dette er
En digital tvilling av Norges befolkning, bygget kommune for kommune fra åpen
SSB-data. Hovedfunksjonen er en **interaktiv spørretab**: du stiller et
spørsmål ("Hva synes folk om bompenger?", "Bør Norge bygge mer vindkraft?",
"Hvor bekymret er folk for økonomien?") og får et representativt, vektet svar
— for hele Norge eller for én bestemt kommune, med oppdeling per region.

Valgsimulering er IKKE hovedmålet. Valg er én av flere kalibreringskilder, og
det eneste holdningsmålet vi har på faktisk kommunenivå — derfor brukes det som
lakmustest på om modellen ekstrapolerer geografisk på en troverdig måte.

Inspirert av "Sim Francisco" (Anthropic Claude Opus 4.8 hackathon, juni 2026):
en census-seedet befolkning der man kan polle den syntetiske byen på sekunder.

## Den avgjørende datainnsikten (les nøye — den former alt)
SSB-data deler seg i to typer:

1. **Strukturdata FINNES per kommune**: befolkning, alder, kjønn, utdanning,
   husholdningsinntekt (helt ned til grunnkrets), valgresultat. Dette gir oss
   *hvem som bor hvor*.

2. **Holdningsdata finnes stort sett BARE nasjonalt, brutt på demografi**:
   tillit (13834), livskvalitet/mening/mestring (13790/13792/13793), frivillig
   innsats (13831). Brutt på alder, kjønn, inntekt, økonomisk status — men
   IKKE på kommune. SSB måler ikke holdninger representativt per kommune.
   (Unntak funnet underveis: mening/mestring finnes OGSÅ per fylke (13798) og
   sentralitet (13799) — for grovt til å drive modellen, men gull som
   VALIDERING av den geografiske ekstrapoleringen. Se Modul 7.)

**Maskinens kjernemekanikk** kobler disse: kommunen gir den demografiske miksen;
de nasjonale holdningstabellene (brutt på demografi) gir hva folk med disse
egenskapene typisk mener. En 55-årig mann med fagbrev i Vinje arver
holdningsprofilen for sin demografiske gruppe nasjonalt, men plasseres i Vinjes
konkrete miks. Aggregert opp får hver kommune sin egen holdningssignatur — ikke
fordi SSB målte kommunen direkte, men fordi sammensetningen er forskjellig.

**Vær ærlig om dette i systemet**: modellen EKSTRAPOLERER kommunale holdninger
fra nasjonale demografi-holdningskoblinger. Den måler dem ikke direkte. LLM-en
fyller gapet mellom de grove SSB-aksene og et fritt spørsmål, forankret i den
demografiske profilen. Valg-backtesten er det vi bruker for å sjekke at
ekstrapoleringen faktisk fungerer geografisk.

## Ærlige begrensninger (skjul dem aldri i UI-et)
- Kommunale holdninger er ekstrapolert, ikke målt. Vis usikkerhet.
- LLM-persona-svar kan ha systematisk skjevhet (f.eks. dempe ytterpunkter).
  Kalibreringssteget finnes nettopp for å fange og korrigere dette.
- SSB prikker (anonymiserer) celler < 3 personer → manglende verdier. Håndter
  eksplisitt, aldri stille imputering.
- For frie spørsmål uten noe fasit-signal kan vi IKKE måle nøyaktighet direkte
  — vi kan bare vise at modellen er konsistent med de holdningsaksene vi HAR,
  og at den treffer på det ene området vi kan måle (valg). Kommuniser dette.
- Flerpartisystem + ingen likvide prediksjonsmarkeder gjør valgkalibrering
  vanskeligere enn Sim Franciscos binære case. Til gjengjeld: flere valg å
  backteste mot per kommune.

---

## Datakilder (SSB API v0 — alle bekreftet fungerende)

Base: `https://data.ssb.no/api/v0/no/table/{id}`
Metadata: GET. Datauttrekk: POST med json-stat2. Åpent, CC BY 4.0, ingen nøkkel.

### Strukturlag (per kommune — bygger personaene og stereotypene)
| Tabell | Innhold | Regioner |
|--------|---------|----------|
| 07459 | Befolkning etter region, kjønn, alder (106 grupper) | 994 |
| 09429 | Personer 16+ etter utdanningsnivå, kjønn | 979 |
| 06944 | Husholdningsinntekt (median) etter region, husholdningstype | 3283 |
| 12944 | Personer i husholdninger med vedvarende lavinntekt | kommune |
| 03797 | Formuesposter fra selvangivelsen | kommune |

NB om skattedata: Skatteetatens API-er (inntekt, beregnet skatt, formue) og den
offentlige skattelista gir IKKE en farbar vei — de krever Maskinporten/
virksomhetstilgang og returnerer navngitte enkeltpersoner med fullt GDPR-ansvar.
Vi trenger dem ikke: vi bygger personas fra FORDELINGER, ikke ekte individer, og
inntekts-/formuesfordelingen finnes åpent per kommune i tabellene over. Dette gir
oss stereotyper ("velstående barnefamilie i Bærum" vs. "aleneforelder med
lavinntekt i Drammen") uten å røre en eneste ekte skatteoppføring.

### Holdningslag (nasjonalt, brutt på demografi — gir personaene meninger)
| Tabell | Innhold | Brutt på |
|--------|---------|----------|
| 13834 | Tillit (myndigheter, media, folk) | alder, kjønn |
| 13839 | Tillit | økonomisk status |
| 13790 | Livskvalitet: mening og mestring | kjønn, alder |
| 13792 | Livskvalitet | inntektsgruppe |
| 13793 | Livskvalitet | økonomisk status |
| 13831 | Frivillig innsats siste 12 mnd | økonomisk status |

### Kalibrerings-/fasitlag (per kommune — eneste geografiske holdningsmål)
| Tabell | Innhold | Siste år |
|--------|---------|----------|
| 08092 | Stortingsvalg: godkjente stemmer etter parti | 2025 |
| 01180 | Kommunestyrevalg: godkjente stemmer etter parti | 2023 |
| 13360 | Valgdeltakelse % etter region, kjønn, utdanning | 2025 |
| 13798 | Mening/mestring etter FYLKE (myk geo-fasit for holdninger) | 2025 |
| 13799 | Mening/mestring etter SENTRALITET (myk geo-fasit) | 2025 |

Partikoder (08092/01180): 01=Ap 02=FrP 03=Høyre 04=KrF 05=Sp 06=SV 07=Venstre
08=MDG 55=Rødt. Kommunekoder 4-sifret; bruk Klass-API (klass.ssb.no) for å
mappe på tvers av sammenslåinger (2020-reform, 2024-reverseringer).

### Mulige eksterne supplement (senere, valgfritt)
- pollofpolls.no — partibarometre for løpende kalibrering.
- Norsk medborgerpanel / ESS (European Social Survey) — rikere holdningsakser
  enn SSB, hvis vi vil utvide verdiprofilene. Sjekk lisens før bruk.

---

## Arkitektur (modulær — én ansvarsoppgave per modul)

### Modul 1 — Datalag (`data/`)
SSB-klient med diskcache (parquet). Henter struktur-, holdnings- og fasitlag.
Klass-mapping for kommunekoder. Logg prikkede/manglende celler. Rene dataframes.

### Modul 2 — Populasjonsbygger (`population/`)
Per kommune: sample N syntetiske personer som matcher fellesfordeling av
(alder × kjønn × utdanning × inntekt) via iterativ proporsjonal tilpasning (IPF)
fra marginalene. Hver person: {kommune, alder, kjønn, utdanning, inntektsgruppe,
økonomisk status, urban/rural}.

### Modul 3 — Holdningsberikelse (`enrich/`)
KJERNEN i forskjellen fra ren valgmodell. For hver syntetisk person: slå opp i
holdningslaget basert på personens demografiske akser (alder/kjønn/inntekt/
økonomisk status) → fest en holdningsprofil (tillitsnivå, livskvalitetsskår,
frivillighetstilbøyelighet). Dette blir konteksten LLM-en får når den svarer.

### Modul 4 — Persona-klynging (`personas/`)
Klyng personene til ~300 representative personas (à la Sim Francisco) for å
kutte inferenskostnad 10–100x. Hver persona bærer vekt = antall innbyggere den
representerer. Bevar geografisk + demografisk + holdningsmessig spredning.

### Modul 5 — Spørremotor (`engine/`)
Hjertet i spørretaben. Input: et fritt spørsmål + valgt geografi (hele Norge
eller kommune X). For hver relevant persona: kall Claude med
persona-beskrivelse (demografi + holdningsprofil) + spørsmålet → strukturert
JSON-svar (f.eks. posisjon på skala + kort begrunnelse + sikkerhet).
Vekt svaret med personavekt OG deltakelses-/engasjementssannsynlighet der
relevant. Batch + cache. Valgfritt: verifier/adversarial-agent som Sim Francisco.

### Modul 6 — Aggregering (`aggregate/`)
Vektet opprulling: persona → kommune → fylke → nasjonalt. Returner fordeling
med usikkerhetsbånd, ikke bare ett tall. Per-kommune-utslag for kart.

### Modul 7 — Kalibrering & backtest (`calibration/`)
- Valg-backtest (eneste harde fasit): bygg på 2021-demografi → simuler
  partipreferanse → mål mot faktisk 2021-resultat per kommune (08092).
  Metrikk: MAE per parti, kommune-korrelasjon. Kalibrer på ett valg, valider
  out-of-sample på et annet. ALDRI kalibrer + valider på samme valg.
- Holdnings-konsistens (myk sjekk): når modellen polles på et tema som
  korresponderer med en SSB-holdningsakse (f.eks. generell tillit), skal den
  nasjonale aggregaten reprodusere SSB-fordelingen den ble seedet fra.
- For frie spørsmål uten fasit: rapporter konsistens + usikkerhet, ikke falsk
  presisjon.

### Modul 8 — Spørretab & visning (`app/`)
Interaktivt UI: tekstboks for spørsmål + geografivelger (hele Norge / kommune).
Resultat: aggregert svar + kommune-choropleth (kart) + demografisk oppdeling.
Vis ALLTID usikkerhet og at kommunale holdninger er ekstrapolert. Kan kobles
til GitHub Pages senere.

---

## Rekkefølge (bygg slik, verifiser hvert steg mot fasit)
1. Modul 1 — data inn + validert. Verifiser mot kjent fasit (Oslo Ap 2025 ≈
   25.7% fra 08092). Hent ut minst én holdningstabell og én strukturtabell.
2. Modul 7 sin valg-backtest-løkke TIDLIG (skjelett), så vi måler fra dag én.
3. Modul 2 → 3 — populasjon + holdningsberikelse. Dette er det nye kjernesteget.
4. Modul 4 — klynging.
5. Modul 5 — spørremotor. Mest iterasjon her. Test først på et spørsmål som
   HAR en SSB-akse (tillit), så vi kan sjekke konsistens.
6. Modul 6 — aggregering. Lukk valg-backtest-loopen, mål, juster.
7. Modul 8 — spørretab.

## Designprinsipper
- Reproduserbarhet: fast seed, cache alt rådata til disk.
- Mål alt vi KAN måle (valg + holdningskonsistens). For resten: vis usikkerhet,
  aldri falsk presisjon.
- Håndter manglende/prikket data eksplisitt.
- Løst koblede moduler — rene dataframes/JSON mellom lag.
- Start lite: 5–10 kommuner end-to-end før hele landet.
- Vær radikalt ærlig i UI-et om hva som er målt vs. ekstrapolert.

## Definisjon av "interessant nok"
(a) Valg-backtest: reproduserer 2021 per kommune, out-of-sample, MAE under en
satt baseline på de store partiene. (b) Holdningskonsistens: nasjonal aggregat
på et tillitsspørsmål reproduserer SSBs tillitsfordeling. (c) Spørretaben gir
plausible, geografisk differensierte svar på frie spørsmål, med ærlig usikkerhet.

## Stack
Python 3.11+, requests, pandas, pyarrow, scikit-learn (IPF/klynging),
anthropic SDK. Visning: HTML/Plotly choropleth (kommunekart).
