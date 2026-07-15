"""
STEK 2035 -- Assign LDA topics to the live RAG chunk corpus
=============================================================
Trains a fresh Gensim LDA model (k=8, matching the best-coherence
configuration found by stek_output/texts/stek_pipeline_v2.py) directly on
vector_store/chunks.jsonl -- the exact corpus rag_server.py serves from --
and writes back a single dominant topic id per chunk via real
LdaModel.get_document_topics() inference.

This is a different, more direct signal than the keyword-heuristic "topics"
field already present in chunks.jsonl (built by input_topic_tagging_extraction.ipynb
/ the merge_and_embedding notebooks, which just checks whether LDA top-words
appear as substrings). That field and "relevance_score" are left untouched;
this script only adds a new "lda_topic" field alongside them.

Usage:
    stekenv\\Scripts\\python assign_lda_topics.py
    python -m spacy download de_core_news_sm   # first run only

Reads:  vector_store/chunks.jsonl
Writes: vector_store/chunks.jsonl        (adds "lda_topic" field, in place)
        vector_store/lda_topics.json     (topic id -> label/top_words/chunk_count)
        vector_store/lda_model/          (persisted model + dictionary + phraser)
"""

import json
import logging
from pathlib import Path

import spacy
from gensim import corpora
from gensim.models import LdaModel, Phrases
from gensim.models.phrases import Phraser

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("assign_lda_topics")

VECTOR_STORE = Path("vector_store")
CHUNKS_PATH = VECTOR_STORE / "chunks.jsonl"
TOPICS_PATH = VECTOR_STORE / "lda_topics.json"
MODEL_DIR = VECTOR_STORE / "lda_model"

NUM_TOPICS = 8          # best-coherence k found by stek_pipeline_v2.py (c_v = 0.4585)
RANDOM_SEED = 42
TOP_N_WORDS = 12

# Preprocessing settings mirror stek_output/texts/config.yaml (the config
# that produced the best k=8 result), minus the cleaning/chunking steps --
# vector_store chunks are already cleaned and chunked upstream.
SPACY_MODEL = "de_core_news_sm"
KEEP_POS = {"NOUN", "PROPN", "ADJ"}
MIN_TOKEN_LEN = 3
CUSTOM_STOPWORDS = {
    "heidelberg", "heidelberger", "stadt", "stek", "seite",
    "dokumentation", "auftrag", "gmbh", "thema", "frage", "beispiel",
}
BIGRAM_MIN_COUNT = 10
BIGRAM_THRESHOLD = 12.0
DICT_NO_BELOW = 5
DICT_NO_ABOVE = 0.5

# Provisional labels -- refine after inspecting the printed top words below.
TOPIC_LABELS = {
    0: "Bildung, Wohnen & Alltag",
    1: "Flaechennutzung & fachliche Grundlagen",
    2: "Stadtgesellschaft & Wirtschaft",
    3: "Ziele, Ressourcen & Region",
    4: "Stadtentwicklung & Klima",
    5: "Gruenflaechen & Natur",
    6: "Kultur, Vielfalt & Quartier",
    7: "Bevoelkerung & Wohnraum-Statistik",
}


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def preprocess(texts: list[str], nlp) -> list[list[str]]:
    docs = []
    for doc in nlp.pipe(texts, batch_size=64):
        toks = []
        for t in doc:
            if t.pos_ not in KEEP_POS or t.is_stop or not t.is_alpha:
                continue
            lemma = t.lemma_.lower()
            if len(lemma) < MIN_TOKEN_LEN or lemma in CUSTOM_STOPWORDS:
                continue
            toks.append(lemma)
        docs.append(toks)
    return docs


def main():
    log.info("Loading %s ...", CHUNKS_PATH)
    chunks = load_chunks()
    log.info("Loaded %d chunks", len(chunks))

    log.info("Loading spaCy model %s ...", SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])

    docs = preprocess([c["text"] for c in chunks], nlp)
    log.info("Lemmatized %d chunks", len(docs))

    phrases = Phrases(docs, min_count=BIGRAM_MIN_COUNT, threshold=BIGRAM_THRESHOLD)
    bigram = Phraser(phrases)
    docs = [bigram[d] for d in docs]

    dictionary = corpora.Dictionary(docs)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE)
    log.info("Dictionary: %d -> %d terms after filter_extremes", before, len(dictionary))

    corpus_bow = [dictionary.doc2bow(d) for d in docs]

    log.info("Training LDA (k=%d) ...", NUM_TOPICS)
    lda = LdaModel(
        corpus=corpus_bow, id2word=dictionary, num_topics=NUM_TOPICS,
        passes=15, iterations=200, alpha="auto", eta="auto",
        chunksize=200, random_state=RANDOM_SEED, eval_every=None,
    )

    log.info("Top words per topic:")
    topic_words = {}
    for i in range(NUM_TOPICS):
        words = [w for w, _ in lda.show_topic(i, topn=TOP_N_WORDS)]
        topic_words[i] = words
        log.info("  Topic %d: %s", i, ", ".join(words))

    # ---- dominant topic per chunk: real LDA posterior inference
    counts = {i: 0 for i in range(NUM_TOPICS)}
    for chunk, bow in zip(chunks, corpus_bow):
        dist = lda.get_document_topics(bow)
        if dist:
            top_topic, _ = max(dist, key=lambda x: x[1])
        else:
            top_topic = -1  # chunk had no surviving tokens after preprocessing
        chunk["lda_topic"] = top_topic
        if top_topic >= 0:
            counts[top_topic] += 1

    n_unassigned = sum(1 for c in chunks if c["lda_topic"] == -1)
    log.info("Chunks per topic: %s (unassigned: %d)", counts, n_unassigned)

    # ---- write augmented chunks.jsonl (existing fields untouched)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log.info("Updated %s with 'lda_topic' field", CHUNKS_PATH)

    # ---- topic metadata for the /topics API endpoint
    topics_out = [
        {
            "id": i,
            "label": TOPIC_LABELS.get(i, f"Topic {i}: " + ", ".join(topic_words[i][:3])),
            "top_words": topic_words[i],
            "chunk_count": counts[i],
        }
        for i in range(NUM_TOPICS)
    ]
    TOPICS_PATH.write_text(json.dumps(topics_out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", TOPICS_PATH)

    # ---- persist model for reproducibility (not required at serve time)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lda.save(str(MODEL_DIR / "lda.model"))
    dictionary.save(str(MODEL_DIR / "dictionary.dict"))
    bigram.save(str(MODEL_DIR / "bigram.phraser"))
    log.info("Saved model to %s", MODEL_DIR)

    log.info("DONE.")


if __name__ == "__main__":
    main()
