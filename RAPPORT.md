# SimNorge — sluttrapport (post mortem)

**Dato:** 2026-07-03. **Status:** avsluttet av eier; prosjektet nådde ikke målet.
**Formål med rapporten:** ærlig oppsummering av hva som ble bygget, hva som ble
målt, hvorfor det ikke holdt, og hva som er verdt å ta med inn i en ny start.

---

## 1. Tesen som ble testet

En norsk versjon av Sim Francisco (github.com/tejasprabhune/simfrancisco):
en syntetisk befolkning bygget av åpen SSB-data som kan «polles» med fritt
formulerte spørsmål, per kommune. Antakelsen var at Norges åpne data
(holdningstabeller brutt på demografi + valgfasit for alle ~357 kommuner)
ville gjøre den norske varianten *bedre* enn originalen.

## 2. Hva som ble bygget (moduler, alle fungerende)

| Modul | Innhold | Tilstand |
|---|---|---|
| 1 `data/` | SSB-klient (json-stat2, diskcache), Klass-normalisering av kommunekoder på tvers av årganger | Verifisert mot fasit (Oslo 2025 eksakt) |
| 2 `population/` | Syntetisk 16+-befolkning per kommune via IPF (kjønn×alder×utdanning×økonomi + arbeidsmarkedsstatus) | Marginaler verifisert ±3 personer |
| 3 `enrich/` | Holdningsberikelse fra nasjonale tabeller (tillit, livskvalitet, frivillighet), trukket med spredning | Nasjonal konsistens ±2 pp |
| 4 `personas/` | Vektede personas som bærer fordelinger | Aggregat + spredning bevart |
| 5 `engine/` | LLM-polling via claude-CLI (claude-sonnet-4-6), diskcache, trygg parsing | SSB-konsistenssjekk bestått |
| 6/7 `calibration/` | Valg-backtest (3 motorvarianter mot baseliner), økologisk inferens (ikke-LLM), affin korreksjon, deltakelsesvekting | Alt målt — se §3 |
| 8 `app/` | Spørretab med ærlighetsbannere, LLM-sikkerhet rapportert | Live-verifisert |

Tester: 15 testfiler, alle grønne. Git-historikk: 8 commits.

## 3. Måleresultatene (grunnen til konklusjonen)

Valg-backtest 2021→2025, 8 prøvekommuner, 9-parti-basis, MAE i prosentpoeng.
All skåring out-of-sample (byggeår ≠ målår; korreksjon kryssvalidert per kommune).

| Metode | Rå MAE | Med samme affine korreksjon |
|---|---|---|
| **Persistence (forrige valg)** | 3,89 | **1,45** |
| LLM-motor, geo uten navn (beste variant) | 3,71 | 2,01 |
| LLM-motor, demografi alene | 4,87 | — |
| LLM-motor, med kommunenavn | 4,23 | — |
| Nasjonalt snitt (ingen geografi) | 4,69 | — |
| Økologisk modell (ikke-LLM, alle 355 kommuner) | 5,01 | — |

**Hovedfunnet:** «forrige valg + partivis svingjustering» (i praksis
pollofpolls-metoden, som allerede finnes offentlig) slår hele LLM-pipelinen
når begge korrigeres likt: 1,45 mot 2,01 pp. For valgprediksjon — det eneste
domenet med kommunefasit — tilfører LLM-motoren negativ verdi mot enkleste
eksisterende metode.

Øvrige målte funn:
- **LLM-en demper variasjon mot midten** (korreksjonens b > 1 for Høyre 1,76,
  Sp 1,62, MDG 1,66, FrP 1,27, SV 1,21). Prompt-instrukser hindret det ikke.
- **Deltakelsesvekting (13360+13085, inkl. aldersgradient) hjelper ikke:**
  marginalt negativ for LLM-motoren (3,71→3,81 — LLM-anslag er allerede
  implisitt deltakelsesvektede; eksplisitt vekting dobbelteller), eksakt
  nøytral for den økologiske modellen (fitten absorberer den).
- **Geografisk holdningsvariasjon utenfor politikk er ~flat:** mot SSBs
  fylkes-/sentralitetstall for livskvalitet (13798/13799) forklarer demografi
  omtrent null av geo-variasjonen. Kommunesvar i livskvalitetsdomenet er
  dermed i praksis nasjonaltabellen pluss støy.
- **Memorering ble utelukket** som kilde til motorens (beskjedne) løft:
  varianten MED kommunenavn var svakere enn varianten uten.

## 4. Hvorfor prosjektet ikke holdt

1. **Verdiforslaget var todelt, og begge deler falt.** (a) Valgprediksjon:
   slått av triviell eksisterende metode (tabellen over). (b) Frie spørsmål
   per kommune: det genuint nye — men uten noen fasit å validere mot, så det
   forble en påstand. Et produkt hvis validerte del er dårligere enn det som
   finnes, og hvis nye del er ubevist, har ikke demonstrert eksistensgrunnlag.
2. **Tre stablede uavhengighetsantakelser** (holdningsakser uavhengige gitt
   demografi; IPF-interaksjonsstruktur; LLM-svar gyldige på tvers av
   geografi) multipliserer feil, og Norge mangler åpne mikrodata (PUMS-
   ekvivalent) som kunne erstattet dem. Originalen hadde ekte individdata;
   det var en større fordel enn Norges holdningstabeller.
3. **Rekkefølgefeil i valideringen:** den *rettferdige* baselinen (korrigert
   persistence) ble målt sist, etter at mesteparten av LLM-ressursene var
   brukt. Hadde 1,45-tallet eksistert dag én, hadde LLM-motoren aldri blitt
   bygget i denne formen.

## 5. Ressursbruk (LLM)

Unike betalte celler i cache: ~5 700 valg-celler (456 + 2 179 + 3 091 for de
tre variantene) + app-/konsistenskjøringer. Alt cachet i `.cache/` (53 MB) —
gjenbrukbart, men bundet til modellstreng `claude-sonnet-4-6` og promptene
slik de står.

## 6. Hva som er verdt å ta med videre

**Gjenbrukbar kode (LLM-fri, verifisert):**
- `data/` — SSB-klient med diskcache og prikking-håndtering, og særlig
  Klass-normaliseringen (kommunesammenslåing/splitt på tvers av årganger —
  kjedelig å skrive på nytt, lett å gjøre feil). NB: dimensjoner må filtreres
  eksplisitt i fetch, ellers elimineres de.
- `population/` — IPF-populasjonsbygger verifisert mot marginaler.
- `calibration/` — backtest-harnessen med out-of-sample-vakt, korreksjons-
  modulen med kommune-CV, og den økologiske modellen. Dette apparatet er det
  som *avslørte* at prosjektet ikke holdt — det er verdifullt uansett retning.

**Lærdommer til en ny start:**
1. **Fastsett fasiten og den sterkeste eksisterende metoden FØR noe bygges.**
   Kravet er «slå korrigert persistence / pollofpolls», ikke «slå naiv
   baseline». Mål det billigste først; LLM-kall sist.
2. **Korriger baseliner og modell likt** — en korrigert modell mot ukorrigert
   baseline er en meningsløs (og forførende) sammenlikning.
3. **Frie spørsmål trenger egen fasit** (publiserte meningsmålinger på
   konkrete tema, folkeavstemninger, Medborgerpanel-rapporter). Uten den kan
   spørretab-delen aldri bevise noe.
4. **Mikrodata > marginaler.** Uten felles fordelinger (PUMS/ESS) står alt på
   uavhengighetsantakelser som ikke kan testes innenfra.
5. **Memoreringsvern fra originalen:** backtest med modell med kunnskaps-
   grense FØR målvalget, ikke bare navn-fjerning.
6. **Ressursdisiplin:** ingen betalte kall uten eksplisitt beslutning; cache
   er bundet til modellstreng — modellbytte = full re-polling.

## 7. Konklusjon

Tesen «mer åpen data i Norge → bedre Sim Francisco» ble testet redelig og
falt: Norges åpne *aggregater* kompenserer ikke for manglende *mikrodata*, og
i det eneste domenet med fasit gjør en triviell eksisterende metode jobben
bedre enn hele maskineriet. Det som står igjen med verdi er datalaget,
populasjonsbyggeren og målingsapparatet — og målingene selv, som er grunnen
til at denne rapporten kan skrives med tall i stedet for meninger.
