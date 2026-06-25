# STEK 2035 — NLP Topic Modelling Pipeline

Computational analysis of the **Stadtentwicklungskonzept Heidelberg 2035** document corpus.
The pipeline extracts, cleans, and chunks the source PDFs, then runs **LDA** and **LSA** topic models in parallel, evaluates them with c_v coherence, and writes a quantitative comparison report.

---

## Repository structure

```
stek-2035/
├── stek_pipeline.py          # v1 — baseline pipeline
├── stek_pipeline_v2.py       # v2 — adds inline_patterns + lemma_corrections
├── config.yaml               # all runtime parameters (paths, model settings)
├── stek_output/
│   └── texts/                # input: extracted .txt files (one per PDF)
└── stek_results/             # output: all generated files (auto-created)
    ├── chunks_stats.json
    ├── coherence_sweep.csv
    ├── lda_topics_k{K}.txt
    ├── lsa_topics_k{K}.txt
    ├── doc_topic_examples.txt
    └── comparison_report.txt
```

---

## Pipeline overview

```
config.yaml + *.txt files
        │
        ▼
  clean_raw()          ← remove footers, fix hyphens, normalize whitespace
        │
        ▼
  chunk_document()     ← split each doc into ~N-word pseudo-documents
        │
        ▼
  preprocess_chunks()  ← spaCy lemmatize, POS filter, stopwords, bigrams
        │
        ▼
  Dictionary + corpus_bow   ← Gensim vocab, filter extremes
        │
       / \
      /   \
run_lda() run_lsa()    ← both models run for each k in topic_counts
      \   /
       \ /
        │
  coherence sweep      ← c_v coherence, diversity, Jaccard per k
        │
        ▼
  refit best k models
        │
        ▼
  output files         ← topic tables, comparison report, examples
```

---

## Installation

```bash
pip install pymupdf gensim scikit-learn spacy pyyaml pandas numpy
python -m spacy download de_core_news_lg
```

> Python 3.10+ required (uses `list[str] | None` type hints).

---

## Usage

```bash
# run with default config
python stek_pipeline_v2.py --config config.yaml

# run v1 (no inline_patterns / lemma_corrections)
python stek_pipeline.py --config config.yaml
```

The script reads all `*.txt` files from `paths.text_dir`, processes them, and writes all results to `paths.results_dir`.

---

## config.yaml reference

```yaml
paths:
  text_dir: stek_output/texts      # folder containing extracted .txt files
  results_dir: stek_results        # output folder (created automatically)

cleaning:
  footer_patterns:                 # regex patterns matched against whole lines
    - '^\s*\d+\s*$'               # standalone page numbers
    - 'Stadtentwicklungskonzept'   # running header example
  inline_patterns:                 # (v2 only) regex removed anywhere in text
    - '\bSTEK\s*2035\b'

chunking:
  target_tokens: 300               # words per chunk
  min_tokens: 50                   # chunks shorter than this are dropped

preprocessing:
  spacy_model: de_core_news_lg
  keep_pos: [NOUN, VERB, ADJ]      # only these POS tags survive
  min_token_len: 3
  custom_stopwords: [stadt, heidelberg, jahr]
  lemma_corrections:               # (v2 only) fix spaCy lemma errors
    wohnungen: wohnung
  bigrams:
    enabled: true
    min_count: 5
    threshold: 10.0

dictionary:
  no_below: 3                      # drop words appearing in fewer than N chunks
  no_above: 0.8                    # drop words appearing in more than 80% chunks

models:
  topic_counts: [5, 7, 10, 12, 15] # k values to sweep
  random_seed: 42
  seeds_stability: [1, 7, 21]      # (v2 only) seeds for LDA stability check
  lda:
    passes: 10
    iterations: 400
    alpha: auto
    eta: auto
    chunksize: 100
  lsa:
    use_tfidf: true                # false = raw counts

evaluation:
  coherence_metric: c_v
  top_n_words: 10
```

---

## Function reference

### `clean_raw(text, footer_patterns, inline_patterns=None)`
Structural cleaning **before** any NLP. Three steps in order:
1. Splits text into lines, drops any line matching a `footer_patterns` regex (page numbers, running headers).
2. *(v2 only)* Applies `inline_patterns` as `re.sub` anywhere in the text — catches footers that PDF extraction merged into body paragraphs.
3. Rejoins words hyphenated across line breaks (`"gemes-\nsen"` → `"gemessen"`), then collapses all whitespace to single spaces.

### `chunk_document(text, target_tokens)`
Splits a cleaned document into fixed-size word windows of `target_tokens` words each. Needed because one STEK document accounts for ~46% of the total corpus — without chunking the topic model would be dominated by it. Each chunk becomes one pseudo-document for LDA/LSA.

### `preprocess_chunks(chunks, cfg)`
Full NLP preprocessing using spaCy's German model:
- loads `de_core_news_lg` with parser and NER disabled (speed)
- keeps only tokens whose POS tag is in `keep_pos`
- removes spaCy built-in stopwords and `custom_stopwords`
- lowercases and lemmatizes each token (`t.lemma_`)
- *(v2 only)* applies `lemma_corrections` dict to fix known spaCy errors
- optionally detects bigrams with Gensim `Phrases` (e.g. `bezahlbarer_wohnraum`)

### `run_lda(docs, dictionary, corpus_bow, k, cfg)`
Wraps Gensim `LdaModel`. Probabilistic model — each topic is a distribution over words, each chunk is a mixture of topics. Parameters (`passes`, `iterations`, `alpha`, `eta`) come from config.

### `run_lsa(texts_joined, k, cfg, top_n)`
Implements LSA as `TfidfVectorizer` + `TruncatedSVD`. Algebraic approach — topics are principal components (directions of maximum variance in the TF-IDF matrix). Applies sign correction so each component points toward its dominant positive direction. Returns topics, the fitted SVD object, doc-topic matrix, and the TF-IDF matrix.

### `coherence(topics, docs, dictionary, metric)`
Wraps Gensim `CoherenceModel`. Measures how often the top words of each topic co-occur in the corpus. Higher c_v = more coherent, interpretable topics.

### `topic_diversity(topics)`
Fraction of unique words across all topics' top-N words. Value of 1.0 means no word appears in more than one topic. Higher is better.

### `mean_pairwise_jaccard(topics)`
Mean Jaccard overlap between every pair of topics. Lower is better — topics with low Jaccard are well-separated from each other.

---

## Output files

| File | Description |
|---|---|
| `chunks_stats.json` | Number of chunks, chunks per document, dictionary size, mean tokens per chunk |
| `coherence_sweep.csv` | c_v coherence, diversity, Jaccard for every k for both LDA and LSA |
| `lda_topics_k{K}.txt` | Top 15 words per topic at best LDA k |
| `lsa_topics_k{K}.txt` | Top 15 words per topic at best LSA k, with explained variance ratio |
| `doc_topic_examples.txt` | 10 random chunks with their dominant LDA topic and probability (sanity check) |
| `comparison_report.txt` | Full quantitative LDA vs LSA summary — the main deliverable |

---

## v1 vs v2 differences

| Feature | v1 `stek_pipeline.py` | v2 `stek_pipeline_v2.py` |
|---|---|---|
| Footer removal | Whole-line regex only | + `inline_patterns` via `re.sub` anywhere in text |
| Lemma fixes | None | `lemma_corrections` dict in config |
| LDA stability | Not implemented | Multi-seed Jaccard stability check |

Use **v2** for all new runs. v1 is kept for reproducibility of earlier results.

---

## Tech stack

| Library | Role |
|---|---|
| `spacy` + `de_core_news_lg` | German lemmatization and POS tagging |
| `gensim` | LDA model, bigram detection, coherence scoring |
| `scikit-learn` | TF-IDF vectorizer, TruncatedSVD (LSA) |
| `numpy` / `pandas` | Numerical ops, results tabulation |
| `pyyaml` | Config loading |
| `pathlib` | Cross-platform file handling |

---

## Interpreting results

**Coherence (c_v):** Values above 0.5 indicate coherent topics. For small German-language corpora, 0.4–0.55 is typical.

**Topic diversity:** Above 0.7 is good. Below 0.5 suggests topics are overlapping and k may be too high.

**Jaccard overlap:** Below 0.1 means topics are well-separated. Above 0.2 suggests redundant topics.

**LDA vs LSA:** LDA tends to produce more interpretable topics on short text. LSA captures more global variance but topics can be harder to label. The `comparison_report.txt` gives the quantitative verdict.

---

## Contact

Project: STEK 2035 — Stadtentwicklungskonzept Heidelberg
Course: AI Strategy module — final examination project
