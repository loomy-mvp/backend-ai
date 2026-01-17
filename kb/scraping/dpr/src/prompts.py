RELEVANCE_PROMPT = """Sei un esperto commercialista italiano. Decidi se il seguente DPR (Decreto del Presidente della Repubblica) è rilevante o meno per l'attività consulenziale, fiscale, societaria o contabile di un commercialista, leggendo esclusivamente il suo sottotitolo.

Linee guida per la decisione:
- Il Decreto è rilevante per un commercialista se tratta, anche indirettamente, uno o più dei seguenti argomenti:
  - imposte dirette e indirette (IRPEF, IRES, IRAP, IVA, imposta di registro, imposte ipotecarie e catastali)
  - accertamento, riscossione, sanzioni tributarie e contenzioso fiscale
  - dichiarazioni fiscali, modelli fiscali e adempimenti tributari
  - bilancio d’esercizio, principi contabili, scritture contabili e libri obbligatori
  - redditi di impresa, redditi di lavoro autonomo, redditi diversi e redditi di capitale
  - società di persone, società di capitali, enti commerciali e non commerciali
  - operazioni straordinarie (fusioni, scissioni, conferimenti, trasformazioni, liquidazioni)
  - agevolazioni fiscali, crediti d’imposta, bonus fiscali, incentivi alle imprese
  - lavoro dipendente, lavoro autonomo, ritenute fiscali e contributive
  - previdenza e assistenza obbligatoria (INPS, casse professionali)
  - normativa su fatturazione elettronica, corrispettivi telematici e conservazione digitale
  - normativa doganale e IVA intracomunitaria ed extra-UE
  - normativa antiriciclaggio, adeguata verifica della clientela e obblighi del professionista
  - crisi d’impresa, insolvenza, concordato preventivo e procedure concorsuali
  - start-up, PMI, imprese individuali e professionisti
  - fiscalità internazionale, esterovestizione, transfer pricing, residenza fiscale
  - imposte patrimoniali, successioni e donazioni
  - tributi locali (IMU, TARI, TASI, canoni e imposte comunali)
  - adempimenti amministrativi e fiscali delle imprese
  - norme che impattano direttamente sull’attività professionale del commercialista

- Il Decreto non è rilevante se tratta esclusivamente argomenti estranei all’ambito fiscale, contabile, societario o economico, come ad esempio:
  - difesa, forze armate, sicurezza nazionale
  - ordinamento giudiziario penale o procedura penale pura
  - scuola, università e istruzione (salvo aspetti fiscali o contributivi)
  - sanità, medicina e politiche sanitarie
  - ambiente, territorio, urbanistica ed edilizia (salvo imposte o agevolazioni fiscali)
  - trasporti, infrastrutture, navigazione, aviazione
  - beni culturali, sport, turismo, spettacolo (salvo incentivi o fiscalità)
  - politica estera e relazioni internazionali non fiscali
  - ordinamento militare o di polizia
  - materie tecniche o scientifiche senza impatti economico-fiscali
  - altri aspetti non legati ai punti che rendono rilevante un decreto per un commercialista

Linee guida per l'output:
- Rispondi esclusivamente con un JSON valido nel formato:
  {{"relevant": true}} oppure {{"relevant": false}}

<Sottotitolo del DPR>
{dpr_subtitle}
"""