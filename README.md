# STEK 2035 — Heidelberg Chatbot & Topic Modelling

A German-language RAG chatbot over the **Stadtentwicklungskonzept Heidelberg
2035** (STEK 2035) document corpus, plus the NLP pipeline that analyzes the
same corpus with LDA/LSA topic modelling. This repo is the data + backend
half of the project; the chat UI lives in a separate Next.js repo
(`stek-chatbot`).

---

## Architecture

Three tiers, plus an offline data/modelling pipeline that feeds them:

```
Browser (Next.js UI, separate repo "stek-chatbot")
       │  POST /api/chat   { message, topic? }
       ▼
Next.js API route (app/api/chat/route.ts)
       │  proxies to
       ▼
FastAPI RAG backend — backend/rag_server.py  (localhost:8000)
       │
       ├─ GET  /health   sanity check
       ├─ GET  /topics   list the 8 LDA topics (id, label, top words)
       └─ POST /chat
             ├─ 1. Embed the question (intfloat/multilingual-e5-base)
             ├─ 2. If `topic` given, restrict candidates to that topic's chunks
             ├─ 3. Cosine-similarity search over vector_store/embeddings.npy
             ├─ 4. Build a German prompt from the top-k chunks
             └─ 5. POST to Ollama (local LLM) → answer + source chunks
                        │
                        ▼
                 Ollama (localhost:11434) — e.g. qwen2.5:1.5b
```

The backend is stateless per request — everything it needs (chunks,
embeddings, topic assignments) is loaded once at startup from
`vector_store/`.

---

## Whole flow

### A. Offline: turning PDFs into a servable, topic-tagged corpus

```
pdfs/*.pdf  (12 source documents)
     │  stek_explore.py  (PyMuPDF extraction)
     ▼
stek_output/texts/*.txt
     │
     ├─────────────────────────────┐
     ▼                              ▼
stek_pipeline_v2.py            merge_and_embedding/*.ipynb
(research pipeline:            (serving pipeline: sentence-aware
 LDA + LSA sweep, k=5..20,     chunking + intfloat/multilingual-e5-base
 config.yaml)                   embeddings)
     │                              │
     ▼                              ▼
stek_output/texts/stek_results/   vector_store/
 - coherence_sweep.csv              chunks.jsonl      ← live RAG corpus (1307 chunks)
 - lda_topics_k8.txt (best, c_v=0.46)  embeddings.npy  ← their 768-dim vectors
 - lsa_topics_k5.txt                 meta.json         ← embedding model name/dim
 - comparison_report.txt
```

Each chunk in `chunks.jsonl` already carries a `"topics"` / `"relevance_score"`
field from `input_topic_tagging_extraction.ipynb` — a **keyword heuristic**
(does the chunk contain any top word from the k=5 LDA topic list?), not real
model inference. `rag_server.py` does not use it.

### B. Assigning real LDA topics to the live corpus

```
vector_store/chunks.jsonl (1307 chunks)
     │  assign_lda_topics.py
     │  spaCy (de_core_news_sm) lemmatize + POS-filter + bigrams
     │  → Gensim LdaModel(num_topics=8), same hyperparams as config.yaml
     ▼
chunks.jsonl            + "lda_topic": <0-7>   (real posterior inference,
                                                 one dominant topic per chunk)
vector_store/lda_topics.json    topic id → label, top words, chunk count
vector_store/lda_model/         persisted model + dictionary + phraser
```

Run this once (or whenever the chunk corpus changes) — `rag_server.py` just
reads its output, it never runs spaCy/Gensim itself at serve time.

### C. Runtime: answering a question

1. User types a question in the browser, optionally picks a topic.
2. `Browser → Next.js /api/chat → FastAPI /chat`.
3. `rag_server.py` embeds the question with the `"query: "` prefix (chunks
   were embedded with `"passage: "` — required by `multilingual-e5-base`).
4. If a `topic` was given, retrieval is restricted to that topic's chunk
   indices before ranking; otherwise all 1307 chunks are candidates.
5. Top-k chunks (cosine similarity, dot product — embeddings are
   pre-normalized) go into a German prompt template.
6. The prompt is sent to Ollama's `/api/generate`; the answer is returned
   together with the source chunks (origin file, chunk id, score).

---

## Repository structure

```
STEK2035-chatbot/
├── pdfs/                        12 source PDFs only (gitignored, kept locally)
├── stek_explore.py              PDF → text extraction
├── stek_output/
│   ├── exploration_report.txt
│   └── texts/                   extracted *.txt (input to both pipelines below)
│       ├── config.yaml, config_v2.yaml
│       ├── stek_pipeline.py, stek_pipeline_v2.py    (LDA/LSA research pipeline)
│       ├── bert_topic_stek.py
│       └── stek_results/        coherence sweep, topic tables, comparison report
├── merge_and_embedding/         notebooks that built vector_store/ (serving pipeline)
├── input_topic_tagging_extraction.ipynb   builds the keyword-heuristic "topics" field
├── input_data_topics/           manifest + text used by the notebook above
├── Input text files/            extracted text (duplicate of stek_output/texts, kept for the keyword-tagging notebook)
├── Output_Clusters/              topic clusters from citizen-participation comments
│                                  — a different dataset from the 12 STEK PDFs
├── scraped_data_topics/         scraped heidelberg.de pages, folded into chunks.jsonl
├── assign_lda_topics.py         trains the real LDA model used for topic filtering (see B above)
├── vector_store/
│   ├── chunks.jsonl             live RAG corpus — origin, text, embeddings index, lda_topic
│   ├── chunks_v2.jsonl          experimental re-chunking, never embedded — not used
│   ├── embeddings.npy           (1307, 768) float32, L2-normalized
│   ├── meta.json                embedding model name/dim/count
│   ├── lda_topics.json          topic id → label/top_words/chunk_count
│   └── lda_model/                persisted Gensim LDA model + dictionary + phraser
└── backend/
    ├── rag_server.py            FastAPI app — the only thing that runs at request time
    ├── requirements.txt
    └── SETUP_GUIDE.md           step-by-step local setup (Ollama + backend + frontend)
```

---

## API reference (`backend/rag_server.py`)

| Endpoint | Method | Body / params | Returns |
|---|---|---|---|
| `/health` | GET | — | `{status, chunks_loaded, ollama_model}` |
| `/topics` | GET | — | `[{id, label, top_words, chunk_count}, ...]` (8 topics) |
| `/chat` | POST | `{message: str, top_k?: int, topic?: int}` | `{reply: str, sources: [{origin, chunk_id, text, score}]}` |

`topic` is the LDA topic id from `/topics`; omit it to search the whole
corpus.

---

## Setup & running

See **[backend/SETUP_GUIDE.md](backend/SETUP_GUIDE.md)** for the full
step-by-step (Ollama, Python env, Next.js). Short version:

```bash
# 1. Ollama (separate install): ollama pull qwen2.5:1.5b

# 2. Python backend — run from the repo ROOT, not backend/, since
#    rag_server.py uses paths relative to vector_store/
python -m venv venv && venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.rag_server:app --port 8000

# 3. (One-time / when the corpus changes) regenerate LDA topic assignments
pip install gensim spacy pyyaml pandas
python -m spacy download de_core_news_sm
python assign_lda_topics.py

# 4. Frontend (separate repo)
cd ../stek-chatbot && npm run dev
```

Open **http://localhost:3000**.

---

## Topic modelling pipeline reference (`stek_output/texts/`)

The research pipeline that found k=8 as the best-coherence LDA topic count
(used to configure `assign_lda_topics.py` above). Two versions:

| Feature | v1 `stek_pipeline.py` | v2 `stek_pipeline_v2.py` |
|---|---|---|
| Footer removal | Whole-line regex only | + `inline_patterns` via `re.sub` anywhere in text |
| Lemma fixes | None | `lemma_corrections` dict in config |
| LDA stability | Not implemented | Multi-seed Jaccard stability check |

Use v2 for new runs; v1 is kept for reproducibility of earlier results.

```bash
pip install pymupdf gensim scikit-learn spacy pyyaml pandas
python -m spacy download de_core_news_lg
cd stek_output/texts
python stek_pipeline_v2.py --config config.yaml
```

**Pipeline stages:** `clean_raw()` (strip footers/headers, fix hyphenation)
→ `chunk_document()` (~150-token pseudo-documents) → `preprocess_chunks()`
(spaCy lemmatize, POS-filter to NOUN/PROPN/ADJ, stopwords, bigrams) →
Gensim `Dictionary` → `run_lda()` / `run_lsa()` swept over
`k = [5, 6, 8, 10, 12, 15, 18, 20]` → c_v coherence, topic diversity,
Jaccard overlap → best-k topic tables + `comparison_report.txt`.

**Interpreting results:** c_v above 0.5 is coherent (0.4–0.55 typical for
small German corpora); topic diversity above 0.7 is good; Jaccard overlap
below 0.1 means topics are well-separated. LDA (best c_v = 0.4585 at k=8)
outperformed LSA on interpretability for this corpus; LSA had higher raw
coherence (0.635 at k=5) but topics are harder to label — see
`stek_output/texts/stek_results/comparison_report.txt` for the full
quantitative comparison.

---

## Tech stack

| Library | Role |
|---|---|
| `fastapi` + `uvicorn` | RAG backend server |
| `sentence-transformers` | Query/chunk embeddings (`multilingual-e5-base`) |
| `spacy` + `de_core_news_sm`/`_lg` | German lemmatization and POS tagging |
| `gensim` | LDA model, bigram detection, coherence scoring |
| `scikit-learn` | TF-IDF vectorizer, TruncatedSVD (LSA) |
| Next.js (separate repo) | Chat UI |
| Ollama | Local LLM inference |

---

## Contact

Project: STEK 2035 — Stadtentwicklungskonzept Heidelberg
Course: AI Strategy module — final examination project
