"""
STEK 2035 — Tasks 2–5: Preprocessing, LDA, LSA, Comparative Evaluation
=======================================================================
Usage:
    pip install pymupdf gensim scikit-learn spacy pyyaml pandas
    python -m spacy download de_core_news_lg
    python stek_pipeline.py --config config.yaml

Reads:   stek_output/texts/*.txt   (from stek_explore.py)
Writes:  stek_results/
    chunks_stats.json          chunking + cleaning statistics
    coherence_sweep.csv        c_v coherence for LDA & LSA per topic count
    lda_topics_k{K}.txt        topic tables (best K)
    lsa_topics_k{K}.txt
    doc_topic_examples.txt     sample chunks with dominant topics (sanity check)
    comparison_report.txt      quantitative LDA-vs-LSA summary  -> send back
"""

import argparse
import json
import logging
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("stek_pipeline")

try:
    import spacy
    from gensim import corpora
    from gensim.models import LdaModel, Phrases, CoherenceModel, TfidfModel
    from gensim.models.phrases import Phraser
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\nInstall: pip install gensim scikit-learn spacy pyyaml pandas")


# ---------------------------------------------------------------- cleaning --

def clean_raw(text: str, footer_patterns: list[str]) -> str:
    """Structural cleaning BEFORE NLP: footers, dehyphenation, whitespace."""
    lines = text.split("\n")
    pats = [re.compile(p) for p in footer_patterns]
    lines = [ln for ln in lines if not any(p.search(ln) for p in pats)]
    text = "\n".join(lines)
    # rejoin words hyphenated across line breaks: "gemes-\nsen" -> "gemessen"
    text = re.sub(r"([a-zäöüß])-\s*\n\s*([a-zäöüß])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_document(text: str, target_tokens: int) -> list[str]:
    """Split a long document into pseudo-documents of ~target_tokens words.

    Rationale: the corpus has 12 documents with a 766–110k token range; one
    document is 46% of the corpus. Topic models need many comparable units,
    so we model chunks, not whole documents.
    """
    words = text.split()
    return [" ".join(words[i:i + target_tokens])
            for i in range(0, len(words), target_tokens)]


# ----------------------------------------------------------- preprocessing --

def preprocess_chunks(chunks: list[str], cfg: dict) -> list[list[str]]:
    """spaCy lemmatization + POS filter + stopwords + optional bigrams."""
    p = cfg["preprocessing"]
    log.info("Loading spaCy model %s ...", p["spacy_model"])
    nlp = spacy.load(p["spacy_model"], disable=["parser", "ner"])
    keep_pos = set(p["keep_pos"])
    custom_stop = set(w.lower() for w in p["custom_stopwords"])
    min_len = p["min_token_len"]

    docs = []
    for doc in nlp.pipe(chunks, batch_size=64):
        toks = []
        for t in doc:
            if t.pos_ not in keep_pos or t.is_stop or not t.is_alpha:
                continue
            lemma = t.lemma_.lower()
            if len(lemma) < min_len or lemma in custom_stop:
                continue
            toks.append(lemma)
        docs.append(toks)
    log.info("Lemmatized %d chunks", len(docs))

    if p["bigrams"]["enabled"]:
        phrases = Phrases(docs, min_count=p["bigrams"]["min_count"],
                          threshold=p["bigrams"]["threshold"])
        bigram = Phraser(phrases)
        docs = [bigram[d] for d in docs]
        n_bi = len({t for d in docs for t in d if "_" in t})
        log.info("Bigram pass added %d distinct bigrams", n_bi)
    return docs


# ----------------------------------------------------------------- models --

def topic_diversity(topics: list[list[str]]) -> float:
    """Fraction of unique words across all topics' top-N (1.0 = no overlap)."""
    all_words = [w for t in topics for w in t]
    return len(set(all_words)) / max(len(all_words), 1)


def mean_pairwise_jaccard(topics: list[list[str]]) -> float:
    """Mean Jaccard overlap between topic pairs (lower = better separated)."""
    if len(topics) < 2:
        return 0.0
    vals = [len(set(a) & set(b)) / len(set(a) | set(b))
            for a, b in combinations(topics, 2)]
    return float(np.mean(vals))


def run_lda(docs, dictionary, corpus_bow, k, cfg):
    m = cfg["models"]["lda"]
    return LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=k,
                    passes=m["passes"], iterations=m["iterations"],
                    alpha=m["alpha"], eta=m["eta"], chunksize=m["chunksize"],
                    random_state=cfg["models"]["random_seed"], eval_every=None)


def lda_topics(model, top_n) -> list[list[str]]:
    return [[w for w, _ in model.show_topic(i, topn=top_n)]
            for i in range(model.num_topics)]


def run_lsa(texts_joined, k, cfg, top_n):
    """LSA = TF-IDF + TruncatedSVD. Returns topics + model objects."""
    vec = TfidfVectorizer(max_features=20000) if cfg["models"]["lsa"]["use_tfidf"] \
        else CountVectorizer(max_features=20000)
    X = vec.fit_transform(texts_joined)
    svd = TruncatedSVD(
        n_components=k, random_state=cfg["models"]["random_seed"])
    doc_topic = svd.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for comp in svd.components_:
        # sign correction: orient each component to its dominant direction
        if comp[np.argmax(np.abs(comp))] < 0:
            comp = -comp
        topics.append(list(terms[np.argsort(comp)[::-1][:top_n]]))
    return topics, svd, doc_topic, X


def coherence(topics, docs, dictionary, metric) -> float:
    cm = CoherenceModel(topics=topics, texts=docs, dictionary=dictionary,
                        coherence=metric, processes=1)
    return cm.get_coherence()


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    text_dir = Path(cfg["paths"]["text_dir"])
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    top_n = cfg["evaluation"]["top_n_words"]

    files = sorted(text_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files in {text_dir.resolve()}")

    # ---- structural cleaning + chunking
    chunks, chunk_sources = [], []
    for f in files:
        cleaned = clean_raw(f.read_text(encoding="utf-8"),
                            cfg["cleaning"]["footer_patterns"])
        cs = chunk_document(cleaned, cfg["chunking"]["target_tokens"])
        chunks.extend(cs)
        chunk_sources.extend([f.stem] * len(cs))
    log.info("Created %d raw chunks from %d documents",
             len(chunks), len(files))

    # ---- NLP preprocessing
    docs = preprocess_chunks(chunks, cfg)
    keep = [i for i, d in enumerate(docs)
            # POS filter shrinks ~3x
            if len(d) >= cfg["chunking"]["min_tokens"] // 3]
    docs = [docs[i] for i in keep]
    chunks = [chunks[i] for i in keep]
    chunk_sources = [chunk_sources[i] for i in keep]
    log.info("Kept %d chunks after min-length filter", len(docs))

    dictionary = corpora.Dictionary(docs)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=cfg["dictionary"]["no_below"],
                               no_above=cfg["dictionary"]["no_above"])
    log.info("Dictionary: %d -> %d terms after filter_extremes",
             before, len(dictionary))
    corpus_bow = [dictionary.doc2bow(d) for d in docs]
    texts_joined = [" ".join(d) for d in docs]

    (out / "chunks_stats.json").write_text(json.dumps({
        "n_chunks": len(docs),
        "chunks_per_doc": pd.Series(chunk_sources).value_counts().to_dict(),
        "dict_size": len(dictionary),
        "mean_tokens_per_chunk": float(np.mean([len(d) for d in docs])),
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- coherence sweep
    rows = []
    metric = cfg["evaluation"]["coherence_metric"]
    for k in cfg["models"]["topic_counts"]:
        log.info("=== k=%d ===", k)
        lda = run_lda(docs, dictionary, corpus_bow, k, cfg)
        t_lda = lda_topics(lda, top_n)
        c_lda = coherence(t_lda, docs, dictionary, metric)

        t_lsa, _, _, _ = run_lsa(texts_joined, k, cfg, top_n)
        c_lsa = coherence(t_lsa, docs, dictionary, metric)

        rows.append({"k": k,
                     "lda_coherence": round(c_lda, 4),
                     "lsa_coherence": round(c_lsa, 4),
                     "lda_diversity": round(topic_diversity(t_lda), 4),
                     "lsa_diversity": round(topic_diversity(t_lsa), 4),
                     "lda_jaccard": round(mean_pairwise_jaccard(t_lda), 4),
                     "lsa_jaccard": round(mean_pairwise_jaccard(t_lsa), 4)})
        log.info("k=%2d  LDA c_v=%.4f  LSA c_v=%.4f", k, c_lda, c_lsa)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "coherence_sweep.csv", index=False)

    # ---- refit best models, dump full topic tables
    best_k_lda = int(sweep.loc[sweep.lda_coherence.idxmax(), "k"])
    best_k_lsa = int(sweep.loc[sweep.lsa_coherence.idxmax(), "k"])
    log.info("Best k: LDA=%d, LSA=%d", best_k_lda, best_k_lsa)

    lda = run_lda(docs, dictionary, corpus_bow, best_k_lda, cfg)
    t_lda = lda_topics(lda, 15)
    (out / f"lda_topics_k{best_k_lda}.txt").write_text(
        "\n".join(f"Topic {i:2d}: " + ", ".join(ws)
                  for i, ws in enumerate(t_lda)),
        encoding="utf-8")

    t_lsa, svd, doc_topic, _ = run_lsa(texts_joined, best_k_lsa, cfg, 15)
    (out / f"lsa_topics_k{best_k_lsa}.txt").write_text(
        "\n".join(f"Topic {i:2d} (var {svd.explained_variance_ratio_[i]:.3f}): "
                  + ", ".join(ws) for i, ws in enumerate(t_lsa)),
        encoding="utf-8")

    # ---- qualitative sanity check: dominant topic for sample chunks
    samples = []
    rng = np.random.default_rng(cfg["models"]["random_seed"])
    for idx in rng.choice(len(docs), size=min(10, len(docs)), replace=False):
        dist = lda.get_document_topics(corpus_bow[idx])
        top_topic, prob = max(dist, key=lambda x: x[1])
        samples.append(f"[{chunk_sources[idx]}] LDA topic {top_topic} (p={prob:.2f})\n"
                       f"  words: {', '.join(t_lda[top_topic][:8])}\n"
                       f"  text:  {chunks[idx][:220]}...\n")
    (out / "doc_topic_examples.txt").write_text("\n".join(samples), encoding="utf-8")

    # ---- comparison report
    rep = ["STEK 2035 — LDA vs LSA QUANTITATIVE COMPARISON", "=" * 60,
           sweep.to_string(index=False), "",
           f"Best LDA: k={best_k_lda}, c_v={sweep.lda_coherence.max():.4f}",
           f"Best LSA: k={best_k_lsa}, c_v={sweep.lsa_coherence.max():.4f}", "",
           f"--- LDA topics (k={best_k_lda}) ---"]
    rep += [f"T{i:2d}: " + ", ".join(ws[:10]) for i, ws in enumerate(t_lda)]
    rep += ["", f"--- LSA topics (k={best_k_lsa}) ---"]
    rep += [f"T{i:2d} (var {svd.explained_variance_ratio_[i]:.3f}): " + ", ".join(ws[:10])
            for i, ws in enumerate(t_lsa)]
    (out / "comparison_report.txt").write_text("\n".join(rep), encoding="utf-8")

    log.info("All results in %s", out.resolve())
    print("\nDONE. Send back: stek_results/comparison_report.txt "
          "(+ doc_topic_examples.txt and chunks_stats.json if possible)")


if __name__ == "__main__":
    main()
