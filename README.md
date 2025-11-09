# Toolkit di Analisi Stilometrica e Complessità Testuale

Questo è un toolkit Python da riga di comando per eseguire un'analisi stilometrica e di complessità su un corpus di testi in italiano.

Lo script carica un file (CSV o Parquet), applica un'avanzata elaborazione NLP utilizzando `spaCy` (una sola volta, in modo efficiente) e calcola un'ampia gamma di feature. Il risultato è un singolo file (CSV o Parquet) che unisce i dati originali con tutte le nuove feature calcolate.



## Funzionalità

Questo script è modulare. Puoi scegliere quali analisi eseguire utilizzando i flag da riga di comando.

### Moduli di Analisi Disponibili

* **`--run_complexity`**: Calcola le metriche di complessità sintattica e lessicale.
    * `cttr`: Corrected Type-Token Ratio
    * `avg_sent_len`: Lunghezza media della frase
    * `avg_depth`: Profondità media dell'albero sintattico
    * `avg_width`: Larghezza media dell'albero sintattico
    * `avg_subordinates`: Media di clausole subordinate per frase
    * `avg_passives`: Media di verbi passivi per frase
* **`--run_pos`**: Calcola le frequenze normalizzate (L2 opzionale) dei Part-of-Speech (es. `NOUN`, `VERB`, `ADJ`).
* **`--run_hapax`**: Calcola le frequenze (L2 opzionale) degli Hapax Legomena (lemmi che appaiono una sola volta).
* **`--run_pos_trigrams`**: Calcola le frequenze normalizzate (L2 opzionale) dei trigrammi di POS (es. `NOUN_ADP_DET`).
* **`--run_char_trigrams`**: Calcola i conteggi (L2 opzionale) dei trigrammi di caratteri (solo lettere).
* **`--run_function_words`**: Calcola le frequenze normalizzate (L2 opzionale) delle parole funzione (es. `DET`, `ADP`, `CCONJ`).
* **`--run_punctuation`**: Calcola le frequenze normalizzate (L2 opzionale) dei segni di punteggiatura.
* **`--run_vocab_tfidf`**: Calcola i punteggi TF-IDF per un vocabolario filtrato e ridotto ai `top_n` lemmi più frequenti.

---

## Setup e Installazione

### 1. Clona il Repository
```bash
git clone [https://github.com/tuo-username/tuo-repository.git](https://github.com/tuo-username/tuo-repository.git)
cd tuo-repository
```

### 2. Crea un ambiente virtuale
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

### 3. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 4. Scarica il Modello spaCy
```bash
python -m spacy download it_core_news_lg
```

##Esecuzione
Esegui lo script stylometry_analyzer.py dal tuo terminale, specificando i file di input/output e i moduli di analisi che desideri attivare.
```bash
python stylometry_analyzer.py \
    --input_file "./dati/miei_dati.csv" \
    --output_file "./risultati/analisi_completa.csv" \
    --text_column "testo_pulito" \
    --id_column "document_id" \
    --run_complexity \
    --run_pos \
    --run_hapax \
    --normalize_features
```

##Output
Lo script produrrà un singolo file (es. analisi_completa.csv) che contiene tutte le colonne del tuo file di input originale più tutte le nuove colonne di feature calcolate dai moduli che hai attivato.
