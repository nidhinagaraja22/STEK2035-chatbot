"""
STEK 2035 — Recompute LDA and LSA Topics (Aligned to Current Corpus)
========================================================================
Rebuilds topic models from scratch against your CURRENT corpus
(corpus_v2_chunks_l4l5split.jsonl, 1258 chunks) — the previous LDA
output (8 topics, chunk counts summing to 1307) was computed against
a DIFFERENT, earlier corpus snapshot and cannot be reliably aligned
to current chunk positions.

Critically, unlike the earlier LDA output you have, this script
outputs PER-CHUNK topic assignments (which topic each of your 1258
chunks belongs to), not just topic-level summaries — the per-chunk
mapping is what stek_topic_concentration_signal.py and any future
topic-based analysis actually needs.

Methodology (matching your project's established choices):
  - German preprocessing: spaCy lemmatization, POS filtering
    (NOUN/VERB/ADJ only — removes articles/prepositions that carry
    no topic signal), custom domain stopwords
  - Bigram detection via Gensim Phrases (captures compound concepts
    like "bezahlbarer_wohnraum")
  - LDA via Gensim (probabilistic, interpretable per-chunk
    distributions)
  - LSA via TF-IDF + TruncatedSVD (algebraic, captures global
    variance)
  - k=5 topics for both, matching your project's established
    coherence/diversity/Jaccard evaluation from earlier work

Usage:
    pip install spacy gensim scikit-learn --break-system-packages
    python -m spacy download de_core_news_lg
    python stek_recompute_topics.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl

Outputs:
    corpus/corpus_v2/lda_topics_v2.json   — per-chunk LDA topic id + full topic summaries
    corpus/corpus_v2/lsa_topics_v2.json   — per-chunk LSA topic id + full topic summaries
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
LDA_OUTPUT_PATH = Path("corpus/corpus_v2/lda_topics_v2.json")
LSA_OUTPUT_PATH = Path("corpus/corpus_v2/lsa_topics_v2.json")

N_TOPICS = 5  # matches this project's established best-k finding
N_TOP_WORDS = 12

# Domain-specific stopwords, on top of spaCy's built-in German list —
# these appear so frequently they'd otherwise dominate every topic
# without adding topical signal (same principle as the corpus-wide
# generic-term stripping used elsewhere in this project)
DOMAIN_STOPWORDS = {
    "heidelberg", "stadt", "stek", "2035", "stadtentwicklungskonzept",
    "konzept", "seite", "abbildung", "quelle", "jahr", "z", "b", "bzw",
    "erstellt", "durch", "auftrag", "co", "kg",
}


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def preprocess_texts(texts):
    """spaCy lemmatization + POS filtering (NOUN/VERB/ADJ only) +
    domain stopword removal + bigram detection."""
    try:
        import spacy
    except ImportError:
        raise SystemExit("Missing dependency: pip install spacy --break-system-packages")

    try:
        nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
    except OSError:
        print("⚠️  de_core_news_lg not found, trying de_core_news_sm instead")
        print("   (lg is more accurate — install with:")
        print("    python -m spacy download de_core_news_lg)")
        nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])

    print(f"Lemmatizing and POS-filtering {len(texts)} chunks...")
    tokenized = []
    for doc in nlp.pipe(texts, batch_size=64):
        tokens = [
            tok.lemma_.lower() for tok in doc
            if tok.pos_ in ("NOUN", "VERB", "ADJ")
            and not tok.is_stop
            and tok.lemma_.lower() not in DOMAIN_STOPWORDS
            and len(tok.lemma_) > 2
            and tok.is_alpha
        ]
        tokenized.append(tokens)

    # bigram detection — captures compound concepts as single tokens
    from gensim.models.phrases import Phrases, Phraser
    print("Detecting bigrams...")
    bigram_model = Phrases(tokenized, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram_model)
    tokenized_bigrams = [bigram_phraser[doc] for doc in tokenized]

    return tokenized_bigrams


def compute_lda(tokenized, n_topics=N_TOPICS):
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    print(f"\nComputing LDA (k={n_topics})...")
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized]

    lda_model = LdaModel(corpus_bow, num_topics=n_topics, id2word=dictionary,
                           random_state=42, passes=10, alpha="auto")

    # per-chunk dominant topic assignment
    chunk_topics = []
    for bow in corpus_bow:
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        chunk_topics.append(int(dominant_topic))

    # topic summaries
    topic_summaries = []
    for tid in range(n_topics):
        top_words = [w for w, _ in lda_model.show_topic(tid, topn=N_TOP_WORDS)]
        chunk_count = chunk_topics.count(tid)
        topic_summaries.append({"id": tid, "top_words": top_words, "chunk_count": chunk_count})

    return chunk_topics, topic_summaries


def compute_lsa(tokenized, n_topics=N_TOPICS):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print(f"\nComputing LSA (k={n_topics})...")
    joined_texts = [" ".join(doc) for doc in tokenized]

    vectorizer = TfidfVectorizer(min_df=3, max_df=0.5)
    tfidf_matrix = vectorizer.fit_transform(joined_texts)

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    # per-chunk dominant topic = highest-loading component
    chunk_topics = [int(np.argmax(row)) for row in lsa_matrix]

    feature_names = vectorizer.get_feature_names_out()
    topic_summaries = []
    for tid in range(n_topics):
        top_indices = svd.components_[tid].argsort()[::-1][:N_TOP_WORDS]
        top_words = [feature_names[i] for i in top_indices]
        chunk_count = chunk_topics.count(tid)
        topic_summaries.append({"id": tid, "top_words": top_words, "chunk_count": chunk_count})

    return chunk_topics, topic_summaries


def main():
    print("=" * 70)
    print("STEK 2035 — Recompute LDA and LSA Topics")
    print("=" * 70)

    if not CHUNKS_PATH.exists():
        print(f"\n❌ {CHUNKS_PATH} not found.")
        return

    chunks = load_chunks()
    print(f"\nLoaded {len(chunks)} chunks from {CHUNKS_PATH}")
    texts = [c.get("text", "") for c in chunks]

    tokenized = preprocess_texts(texts)

    # ── LDA ───────────────────────────────────────────────────────────────────
    lda_topics, lda_summaries = compute_lda(tokenized)
    lda_output = {
        "topics": lda_summaries,
        "chunk_topic_assignment": [
            {"idx": i, "chunk_uid": chunks[i].get("chunk_uid"), "lda_topic": t}
            for i, t in enumerate(lda_topics)
        ],
    }
    with open(LDA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lda_output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ LDA saved: {LDA_OUTPUT_PATH}")
    total_lda = sum(t["chunk_count"] for t in lda_summaries)
    print(f"   Total chunks across topics: {total_lda} (must equal {len(chunks)})")
    assert total_lda == len(chunks), "LDA chunk count mismatch — something is wrong"

    # ── LSA ───────────────────────────────────────────────────────────────────
    lsa_topics, lsa_summaries = compute_lsa(tokenized)
    lsa_output = {
        "topics": lsa_summaries,
        "chunk_topic_assignment": [
            {"idx": i, "chunk_uid": chunks[i].get("chunk_uid"), "lsa_topic": t}
            for i, t in enumerate(lsa_topics)
        ],
    }
    with open(LSA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lsa_output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ LSA saved: {LSA_OUTPUT_PATH}")
    total_lsa = sum(t["chunk_count"] for t in lsa_summaries)
    print(f"   Total chunks across topics: {total_lsa} (must equal {len(chunks)})")
    assert total_lsa == len(chunks), "LSA chunk count mismatch — something is wrong"

    print(f"\n{'='*70}")
    print("Both outputs are PER-CHUNK assignments, aligned to your CURRENT")
    print(f"{len(chunks)}-chunk corpus — safe to use with")
    print(f"stek_topic_concentration_signal.py (point LDA_TOPICS_PATH at")
    print(f"{LDA_OUTPUT_PATH} and adjust it to read the 'chunk_topic_assignment' list).")


if __name__ == "__main__":
    main()
