# -*- coding: utf-8 -*-
"""
Stylometric Analysis and Textual Complexity Toolkit.

This script loads a file (CSV or Parquet), applies advanced NLP analysis
using spaCy, and calculates a wide range of stylometric features.

Execution:

1. Install dependencies:
   pip install -r requirements.txt

2. Download the spaCy model:
   python -m spacy download it_core_news_lg

3. Run the analysis:
   python stylometry_analyzer.py \
       --input_file "path/to/your/data.csv" \
       --output_file "path/to/your/results.csv" \
       --text_column "testo" \
       --id_column "nome" \
       --run_complexity --run_pos --run_hapax
"""

import spacy
import pandas as pd
import numpy as np
import re
import argparse
import logging
from math import sqrt
from collections import Counter, defaultdict
from tqdm import tqdm
from functools import reduce
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def text_preprocessing(test):
    """Applies extensive regex cleaning to normalize legal acronyms and text."""
    testo = re.sub("''", "'", testo)
    testo = re.sub(r'www\.', r'www', testo, flags=re.IGNORECASE)
    testo = re.sub(r'\.com', r'com', testo, flags=re.IGNORECASE)
    testo = re.sub(r'\.it', r'it', testo, flags=re.IGNORECASE)
    testo = re.sub(r'nd\.r\.', r'ndr', testo, flags=re.IGNORECASE)
    testo = re.sub(r'n\.d\.r\.', r'ndr', testo, flags=re.IGNORECASE)
    testo = re.sub(r's\. ?r\. ?l\.', r'srl', testo, flags=re.IGNORECASE)
    testo = re.sub(r's\. ?m ?.i\.', r'smi', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?c\.', r'cc', testo, flags=re.IGNORECASE)
    testo = re.sub(r's\. ?p\. ?a\.', r'spa', testo, flags=re.IGNORECASE)
    testo = re.sub(r'prot\.', r'prot', testo, flags=re.IGNORECASE)
    testo = re.sub(r'lett\.', r'lett', testo, flags=re.IGNORECASE)
    testo = re.sub(r'art\.', r'art', testo, flags=re.IGNORECASE)
    testo = re.sub(r'par\.', r'par', testo, flags=re.IGNORECASE)
    testo = re.sub(r'sig\.', r'sig', testo, flags=re.IGNORECASE)
    testo = re.sub(r'ss\.', r'ss', testo, flags=re.IGNORECASE)
    testo = re.sub(r'd\. ?l\.', r'dl', testo, flags=re.IGNORECASE)
    testo = re.sub(r'd\. ?lgs\.', r'dlgs', testo, flags=re.IGNORECASE)
    testo = re.sub(r'v\.', r'v', testo, flags=re.IGNORECASE)
    testo = re.sub(r'cfr\.', r'cfr', testo, flags=re.IGNORECASE)
    testo = re.sub(r'vd\.', r'vd', testo, flags=re.IGNORECASE)
    testo = re.sub(r'd\. ?p\. ?r.', r'dpr', testo, flags=re.IGNORECASE)
    testo = re.sub(r'ter\.', r'ter', testo, flags=re.IGNORECASE)
    testo = re.sub(r'p\. ?a\.', r'pa', testo, flags=re.IGNORECASE)
    testo = re.sub(r'g\. ?u\.', r'gu', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?p\.', r'cp', testo, flags=re.IGNORECASE)
    testo = re.sub(r'cpc\.', r'cpc', testo, flags=re.IGNORECASE)
    testo = re.sub(r'cpp\.', r'cpp', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?d\. ?s\.', r'cds', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?d\. ?c\.', r'cdc', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?g\. ?a\.', r'cga', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?n\. ?f\.', r'cnf', testo, flags=re.IGNORECASE)
    testo = re.sub(r'co\.', r'co', testo, flags=re.IGNORECASE)
    testo = re.sub(r'lett\.', r'lett', testo, flags=re.IGNORECASE)
    testo = re.sub(r'l\.', r'l', testo, flags=re.IGNORECASE)
    testo = re.sub(r'cir\.', r'cir', testo, flags=re.IGNORECASE)
    testo = re.sub(r'circ\.', r'circ', testo, flags=re.IGNORECASE)
    testo = re.sub(r'del\.', r'del', testo, flags=re.IGNORECASE)
    testo = re.sub(r'o\. ?m\.', r'om', testo, flags=re.IGNORECASE)
    testo = re.sub(r'reg\.', r'reg', testo, flags=re.IGNORECASE)
    testo = re.sub(r'all\.', r'all', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\. ?d\.', r'cd', testo, flags=re.IGNORECASE)
    testo = re.sub(r't\. ?u\. ?a.', r'tua', testo, flags=re.IGNORECASE)
    testo = re.sub(r'cd\.', r'cd', testo, flags=re.IGNORECASE)
    testo = re.sub(r'sez\.', r'sez', testo, flags=re.IGNORECASE)
    testo = re.sub(r'cass\.', r'cass', testo, flags=re.IGNORECASE)
    testo = re.sub(r'civ\.', r'civ', testo, flags=re.IGNORECASE)
    testo = re.sub(r'd\. ?m\.', r'dm', testo, flags=re.IGNORECASE)
    testo = re.sub(r'd ?m\.', r'dm', testo, flags=re.IGNORECASE)
    testo = re.sub(r'n\.', r'n', testo, flags=re.IGNORECASE)
    testo = re.sub(r'c\.', r'c', testo, flags=re.IGNORECASE)
    testo = re.sub(r'\u00a0', r' ', testo)
    testo = re.sub(r'\ufeff', r' ', testo)
    testo = re.sub(r'\u200b', r' ', testo)
    testo = re.sub(r'\x0C', r' ', testo)
    testo = re.sub(r'\xa0', r' ', testo)
    testo = re.sub(r'\s+', ' ', testo)
    return testo.strip()

def normalizza_l2(df, id_column):
    """Applies L2 normalization to all columns except the ID column."""
    if id_column not in df.columns:
        logging.warning(f"ID column '{id_column}' not found for L2 normalization.")
        return df

    ids = df[[id_column]].copy()
    dati = df.drop(columns=[id_column])
    dati_normalizzati = normalize(dati, norm='l2', axis=1)
    df_normalizzato = pd.DataFrame(dati_normalizzati, columns=dati.columns)

    df_normalizzato.reset_index(drop=True, inplace=True)
    ids.reset_index(drop=True, inplace=True)

    df_normalizzato.insert(0, id_column, ids)
    return df_normalizzato

def get_valid_sentences(doc):
    """Returns a list of sentences with more than 3 valid tokens."""
    valid_sentences = []
    if not hasattr(doc, "sents"):
        return []
    for sent in doc.sents:
        valid_tokens = [token for token in sent if not token.is_punct and not token.is_space]
        if len(valid_tokens) > 3:
            valid_sentences.append(sent)
    return valid_sentences

def corrected_type_token_ratio(doc):
    """Calculates CTTR (types / √(2 * tokens)) excluding punctuation and stop words."""
    if doc is None or len(doc) == 0:
        return 0.0
    tokens = [token for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    if not tokens:
        return 0.0
    types = set(token.lemma_.lower() for token in tokens)
    return round(len(types) / sqrt(2 * len(tokens)), 2)

def avg_sentence_length(doc):
    """Calculates the average sentence length (only sentences > 3 words)."""
    valid_sentences = get_valid_sentences(doc)
    if not valid_sentences:
        return 0.0
    words_count = sum(len([token for token in sent if not token.is_punct and not token.is_space]) for sent in valid_sentences)
    return round(words_count / len(valid_sentences), 2)

def compute_depth(node):
    """Recursively calculates the depth of the syntactic tree."""
    if node is None or not hasattr(node, "children"):
        return 0
    children = list(node.children)
    if not children:
        return 0
    return 1 + max(compute_depth(child) for child in children)

def count_nodes_per_level(node, level=0, levels=None):
    """Counts nodes per level in the syntactic tree."""
    if levels is None:
        levels = defaultdict(int)
    if node is None or not hasattr(node, "children"):
        return levels
    levels[level] += 1
    for child in node.children:
        count_nodes_per_level(child, level + 1, levels)
    return levels

def calculate_avg_depth_width(doc):
    """Calculates average depth and width of sentences (only sentences > 3 words)."""
    all_depths = []
    all_widths = []
    valid_sentences = get_valid_sentences(doc)

    for sent in valid_sentences:
        roots = [token for token in sent if token.head == token]
        if not roots:
            continue

        root = roots[0]
        all_depths.append(compute_depth(root))

        levels = count_nodes_per_level(root)
        if levels:
            all_widths.append(max(levels.values()))

    avg_depth = round(sum(all_depths) / len(all_depths), 2) if all_depths else 0.0
    avg_width = round(sum(all_widths) / len(all_widths), 2) if all_widths else 0.0

    return {"avg_depth": avg_depth, "avg_width": avg_width}

def avg_subordinates_per_sentence(doc):
    """Calculates the average number of subordinate clauses per sentence."""
    if doc is None or not hasattr(doc, "sents"):
        return 0.0

    total_subordinates = 0
    total_sentences = 0
    for sent in doc.sents:
        if not hasattr(sent, "__iter__"):
            continue
        total_sentences += 1
        for token in sent:
            if hasattr(token, "dep_") and token.dep_ in {"advcl", "ccomp", "xcomp", "acl:relcl", "acl"}:
                total_subordinates += 1

    return round(total_subordinates / total_sentences, 2) if total_sentences > 0 else 0.0

def avg_passive_verbs(doc):
    """Calculates the average number of passive verbs per sentence."""
    if doc is None or not hasattr(doc, "sents"):
        return 0.0

    passive_verb_count = 0
    sentence_count = 0
    for sent in doc.sents:
        if not hasattr(sent, "__iter__"):
            continue
        sentence_count += 1
        for token in sent:
            if not (hasattr(token, "pos_") and hasattr(token, "dep_") and hasattr(token, "children")):
                continue
            if token.pos_ == "VERB":
                is_passive = token.dep_ == "aux:pass" or any(child.dep_ == "aux:pass" for child in token.children)
                if is_passive:
                    passive_verb_count += 1

    return round(passive_verb_count / sentence_count, 2) if sentence_count > 0 else 0.0

def run_complexity_analysis(docs, ids):
    """
    Runs the complexity analysis suite on a list of spaCy documents.
    Returns a DataFrame with one row per document.
    """
    logging.info("Running complexity analysis module...")
    results = []
    for doc in tqdm(docs, desc="Complexity Analysis"):
        if doc is None:
            # Add an empty record to maintain alignment
            results.append({
                "cttr": 0.0,
                "avg_sent_len": 0.0,
                "avg_depth": 0.0,
                "avg_width": 0.0,
                "avg_subordinates": 0.0,
                "avg_passives": 0.0
            })
            continue

        depth_width = calculate_avg_depth_width(doc)

        results.append({
            "cttr": corrected_type_token_ratio(doc),
            "avg_sent_len": avg_sentence_length(doc),
            "avg_depth": depth_width["avg_depth"],
            "avg_width": depth_width["avg_width"],
            "avg_subordinates": avg_subordinates_per_sentence(doc),
            "avg_passives": avg_passive_verbs(doc)
        })

    df = pd.DataFrame(results)
    df.insert(0, 'id', ids)
    return df

def run_pos_analysis(docs, ids):
    """
    Extracts normalized Part-of-Speech (POS) frequencies.
    """
    logging.info("Running POS analysis module...")
    all_pos_tags = set()
    document_pos_counts = []

    # First pass: collect all POS tags and count
    for doc in tqdm(docs, desc="POS Analysis (Pass 1)"):
        if doc is None:
            document_pos_counts.append(Counter())
            continue

        pos = [token.pos_ for token in doc if token.pos_ != "PUNCT"]
        all_pos_tags.update(pos)
        document_pos_counts.append(Counter(pos))

    sorted_all_pos = sorted(all_pos_tags)
    rows = []

    # Second pass: calculate normalized frequencies
    for counter in tqdm(document_pos_counts, desc="POS Analysis (Pass 2)"):
        length = sum(counter.values()) if counter else 1
        row = [counter.get(p, 0) / length for p in sorted_all_pos]
        rows.append(row)

    df = pd.DataFrame(rows, columns=sorted_all_pos)
    df.insert(0, 'id', ids)
    return df

def run_hapax_analysis(docs, ids):
    """
    Extracts hapax legomena (lemmas with frequency 1) per document.
    """
    logging.info("Running Hapax analysis module...")
    doc_hapax_list = []
    global_vocabulary = set()

    for doc in tqdm(docs, desc="Hapax Analysis"):
        if doc is None:
            doc_hapax_list.append({})
            continue

        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        counter = Counter(lemmas)
        hapax = {lemma: 1 for lemma, freq in counter.items() if freq == 1}
        global_vocabulary.update(hapax.keys())
        doc_hapax_list.append(hapax)

    sorted_vocab = sorted(global_vocabulary)
    matrix = []
    for hapax_dict in doc_hapax_list:
        row = [hapax_dict.get(lemma, 0) for lemma in sorted_vocab]
        matrix.append(row)

    df = pd.DataFrame(matrix, columns=sorted_vocab)
    df.insert(0, 'id', ids)
    return df

def run_pos_trigrams_analysis(docs, ids):
    """
    Extracts normalized POS trigrams.
    """
    logging.info("Running POS Trigrams analysis module...")
    all_trigrams = set()
    document_trigrams_list = []

    for doc in tqdm(docs, desc="POS Trigrams (Pass 1)"):
        if doc is None:
            document_trigrams_list.append([])
            continue

        pos_tags = [token.pos_ for token in doc if token.pos_ != "PUNCT"]
        trigrams = [tuple(pos_tags[i:i+3]) for i in range(len(pos_tags)-2)]
        document_trigrams_list.append(trigrams)
        all_trigrams.update(trigrams)

    sorted_trigrams = sorted(all_trigrams)
    # Rename columns to be valid names (e.g., ('NOUN', 'ADP', 'DET') -> 'NOUN_ADP_DET')
    trigram_columns = ["_".join(t) for t in sorted_trigrams]

    rows = []
    for trigrams in tqdm(document_trigrams_list, desc="POS Trigrams (Pass 2)"):
        counter = Counter(trigrams)
        total = sum(counter.values()) if counter else 1
        row = [counter.get(t, 0) / total for t in sorted_trigrams]
        rows.append(row)

    df = pd.DataFrame(rows, columns=trigram_columns)
    df.insert(0, 'id', ids)
    return df

def run_char_trigrams_analysis(texts, ids):
    """
    Extracts character trigrams (letters only) and their counts.
    """
    logging.info("Running Character Trigrams analysis module...")
    rows = []
    total_trigrams = set()

    for text in tqdm(texts, desc="Character Trigrams"):
        if not isinstance(text, str):
            rows.append({})
            continue

        filtered_text = re.sub(r'[^a-zA-ZÀ-ÿ]', '', text)
        trigrams = [filtered_text[i:i+3] for i in range(len(filtered_text)-2)]
        counter = Counter(trigrams)
        total_trigrams.update(counter.keys())

        # Add the counter to the list
        rows.append(counter)

    # Create DataFrame and fill NaNs with 0
    df_trigrams = pd.DataFrame(rows).fillna(0)

    # Sort columns for consistency
    trigram_columns = sorted([col for col in df_trigrams.columns])
    df_trigrams = df_trigrams[trigram_columns]

    df_trigrams.insert(0, 'id', ids)
    return df_trigrams

def run_function_words_analysis(docs, ids):
    """
    Extracts the normalized frequency of function words.
    """
    logging.info("Running Function Words analysis module...")
    all_function_words = set()
    document_function_words = []

    for doc in tqdm(docs, desc="Function Words (Pass 1)"):
        if doc is None:
            document_function_words.append(([], 1))
            continue

        function_words = [
            token.text.lower()
            for token in doc
            if token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ"}
        ]
        num_valid_tokens = sum(1 for token in doc if token.is_alpha)
        document_function_words.append((function_words, num_valid_tokens))
        all_function_words.update(function_words)

    sorted_words = sorted(all_function_words)
    rows = []

    for function_words, num_valid_tokens in tqdm(document_function_words, desc="Function Words (Pass 2)"):
        counter = Counter(function_words)
        denom = num_valid_tokens if num_valid_tokens > 0 else 1
        row = [counter.get(p, 0) / denom for p in sorted_words]
        rows.append(row)

    df = pd.DataFrame(rows, columns=sorted_words)
    df.insert(0, 'id', ids)
    return df

def run_punctuation_analysis(docs, ids):
    """
    Extracts the normalized frequency of punctuation.
    """
    logging.info("Running Punctuation analysis module...")
    unique_punctuation = set()
    document_punctuation_list = []

    for doc in tqdm(docs, desc="Punctuation (Pass 1)"):
        if doc is None:
            document_punctuation_list.append((Counter(), 1))
            continue

        punctuation = [token.text for token in doc if token.is_punct]
        unique_punctuation.update(punctuation)
        document_punctuation_list.append((Counter(punctuation), len(doc)))

    sorted_punctuation = sorted(unique_punctuation)
    rows = []

    for counter, num_tokens in tqdm(document_punctuation_list, desc="Punctuation (Pass 2)"):
        denom = num_tokens if num_tokens > 0 else 1
        row = [counter.get(p, 0) / denom for p in sorted_punctuation]
        rows.append(row)

    df = pd.DataFrame(rows, columns=sorted_punctuation)
    df.insert(0, 'id', ids)
    return df

def run_tfidf_vocab_analysis(docs, ids, top_n = 500):
    """
    Extracts the filtered TF-IDF vocabulary (lemmas with freq >= 2 in at least one doc)
    and reduced to the top_n most frequent lemmas.
    """
    logging.info("Running TF-IDF Vocabulary analysis module...")
    filtered_vocabulary = set()
    filtered_texts_for_pass_2 = []

    # Build filtered vocabulary
    for doc in tqdm(docs, desc="TF-IDF Vocab (Pass 1)"):
        if doc is None:
            continue
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        counter = Counter(lemmas)
        frequent_lemmas = [lemma for lemma, count in counter.items() if count >= 2]
        filtered_vocabulary.update(frequent_lemmas)

    # Filter texts using the vocabulary
    for doc in tqdm(docs, desc="TF-IDF Vocab (Pass 2)"):
        if doc is None:
            filtered_texts_for_pass_2.append("")
            continue
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
               and token.lemma_.lower() in filtered_vocabulary
        ]
        filtered_texts_for_pass_2.append(" ".join(lemmas))

    # Calculate TF-IDF and select Top N
    vectorizer = TfidfVectorizer(vocabulary=sorted(filtered_vocabulary))
    tfidf_matrix = vectorizer.fit_transform(filtered_texts_for_pass_2)

    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Select Top N lemmas
    logging.info(f"Selecting the top {top_n} most frequent lemmas from {df_tfidf.shape[1]}...")
    lemma_sum = df_tfidf.sum(axis=0)
    top_lemmas = lemma_sum.nlargest(top_n).index.tolist()
    df_top = df_tfidf[top_lemmas]

    df_top.insert(0, 'id', ids)
    return df_top

def load_data(filepath):
    """Loads data from CSV or Parquet."""
    logging.info(f"Loading data from: {filepath}")
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .csv or .parquet.")

def save_data(df, filepath):
    """Saves data to CSV or Parquet."""
    logging.info(f"Saving data to: {filepath}")
    if filepath.endswith(".csv"):
        df.to_csv(filepath, index=False)
    elif filepath.endswith(".parquet"):
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .csv or .parquet.")

def main(args):
    """
    Main function that orchestrates the entire process.
    """

    # Load data
    try:
        df = load_data(args.input_file)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Check for necessary columns
    if args.text_column not in df.columns:
        logging.error(f"Text column '{args.text_column}' not found in the file.")
        return
    if args.id_column not in df.columns:
        logging.error(f"ID column '{args.id_column}' not found in the file.")
        logging.error("Analysis requires a unique ID column (e.g., 'nome' or 'id').")
        return
    if df[args.id_column].duplicated().any():
        logging.warning(f"WARNING: The ID column '{args.id_column}' contains duplicates. Merging may not work correctly.")

    # Pre-process texts
    logging.info("Applying pre-processing (regex cleaning)...")
    texts_raw = df[args.text_column].fillna("").apply(text_preprocessing).tolist()
    ids = df[args.id_column].tolist()

    # Load and apply spaCy (ONLY ONCE)
    logging.info(f"Loading spaCy model '{args.spacy_model}'...")
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        logging.error(f"spaCy model '{args.spacy_model}' not found.")
        logging.error(f"Run: python -m spacy download {args.spacy_model}")
        return

    logging.info(f"Processing {len(texts_raw)} texts with spaCy... (This may take time)")
    # Disable unnecessary pipes to speed up
    # We keep 'parser' for syntactic analyses (depth, subordinates)
    disable_pipes = ["ner"]
    docs = list(tqdm(nlp.pipe(texts_raw, disable=disable_pipes), total=len(texts_raw), desc="spaCy Processing"))

    # List of feature DataFrames to merge
    features_dfs = [df] # Start with the original DataFrame

    # Run the requested analysis modules

    if args.run_complexity:
        df_complexity = run_complexity_analysis(docs, ids)
        features_dfs.append(df_complexity)

    if args.run_pos:
        df_pos = run_pos_analysis(docs, ids)
        if args.normalize_features:
            logging.info("Applying L2 normalization to POS features...")
            df_pos = normalizza_l2(df_pos, args.id_column)
        features_dfs.append(df_pos)

    if args.run_hapax:
        df_hapax = run_hapax_analysis(docs, ids)
        if args.normalize_features:
            logging.info("Applying L2 normalization to Hapax features...")
            df_hapax = normalizza_l2(df_hapax, args.id_column)
        features_dfs.append(df_hapax)

    if args.run_pos_trigrams:
        df_pos_trigrams = run_pos_trigrams_analysis(docs, ids)
        if args.normalize_features:
            logging.info("Applying L2 normalization to POS Trigram features...")
            df_pos_trigrams = normalizza_l2(df_pos_trigrams, args.id_column)
        features_dfs.append(df_pos_trigrams)

    if args.run_char_trigrams:
        # Note: this uses 'texts_raw' not 'docs', as it's regex-based
        df_char_trigrams = run_char_trigrams_analysis(texts_raw, ids)
        if args.normalize_features:
            logging.info("Applying L2 normalization to Character Trigram features...")
            df_char_trigrams = normalizza_l2(df_char_trigrams, args.id_column)
        features_dfs.append(df_char_trigrams)

    if args.run_function_words:
        df_func_words = run_function_words_analysis(docs, ids)
        if args.normalize_features:
            logging.info("Applying L2 normalization to Function Word features...")
            df_func_words = normalizza_l2(df_func_words, args.id_column)
        features_dfs.append(df_func_words)

    if args.run_punctuation:
        df_punct = run_punctuation_analysis(docs, ids)
        if args.normalize_features:
            logging.info("Applying L2 normalization to Punctuation features...")
            df_punct = normalizza_l2(df_punct, args.id_column)
        features_dfs.append(df_punct)

    if args.run_vocab_tfidf:
        df_tfidf = run_tfidf_vocab_analysis(docs, ids, args.vocab_top_n)
        # TF-IDF is already a normalized measure, L2 is not usually necessary
        features_dfs.append(df_tfidf)

    # Merge all feature DataFrames
    if len(features_dfs) > 1:
        logging.info("Merging all feature DataFrames...")
        try:
            df_finale = reduce(lambda left, right: pd.merge(left, right, on=args.id_column, how='left'), features_dfs)
        except Exception as e:
            logging.error(f"Error during DataFrame merge. Ensure the ID column '{args.id_column}' is correct. Error: {e}")
            return
    else:
        logging.warning("No analysis modules were selected. The output file will be a copy of the input.")
        df_finale = df

    # 6. Save the results
    try:
        save_data(df_finale, args.output_file)
        logging.info(f"Analysis completed successfully! Results saved in {args.output_file}")
    except Exception as e:
        logging.error(f"Error saving the output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stylometric Analysis and Textual Complexity Toolkit")

    # I/O Arguments
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input file (.csv or .parquet)")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output file (.csv or .parquet)")

    # Column Arguments
    parser.add_argument("--text_column", type=str, default="testo", help="Name of the column containing the text (default: 'testo')")
    parser.add_argument("--id_column", type=str, default="nome", help="Name of the unique ID column (default: 'nome')")

    # Model Arguments
    parser.add_argument("--spacy_model", type=str, default="it_core_news_lg", help="Name of the spaCy model to download and use (default: 'it_core_news_lg')")

    # Arguments for Analysis Modules (Flags)
    parser.add_argument("--run_complexity", action="store_true", help="Run complexity metrics (TTR, depth, subordinates, etc.)")
    parser.add_argument("--run_pos", action="store_true", help="Run POS frequency analysis")
    parser.add_argument("--run_hapax", action="store_true", help="Run Hapax Legomena (lemmas) analysis")
    parser.add_argument("--run_pos_trigrams", action="store_true", help="Run POS trigram analysis")
    parser.add_argument("--run_char_trigrams", action="store_true", help="Run character trigram analysis")
    parser.add_argument("--run_function_words", action="store_true", help="Run function word analysis")
    parser.add_argument("--run_punctuation", action="store_true", help="Run punctuation analysis")
    parser.add_argument("--run_vocab_tfidf", action="store_true", help="Run TF-IDF vocabulary analysis (filtered and top-n)")

    # Analysis Options
    parser.add_argument("--vocab_top_n", type=int, default=500, help="Number of top lemmas to keep for TF-IDF analysis (default: 500)")
    parser.add_argument("--normalize_features", action="store_true", help="Apply L2 normalization to high-dimensionality feature modules (POS, Hapax, etc.)")

    args = parser.parse_args()
    main(args)
