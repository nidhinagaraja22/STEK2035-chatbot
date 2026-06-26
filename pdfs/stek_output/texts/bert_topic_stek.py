"""
STEK 2035 — BERTopic Standalone Pipeline (German-only, CPU-optimised)
======================================================================
Designed for:
  - German-only documents
  - CPU-only machine (no GPU required)
  - STEK 2035 corpus (extracted .txt files)

Usage:
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn spacy
    python -m spacy download de_core_news_sm
    python stek_bertopic.py

Reads:   F:/Downloads/STEK Heidelberg/output_text_1/*.txt
Writes:  F:/Downloads/STEK Heidelberg/bertopic_results/
    topic_summary.txt          top words per topic
    topic_chunks.txt           which chunks belong to which topic
    chunks_stats.txt           how many chunks per source file
    bertopic_comparison.txt    BERTopic vs LDA/LSA discussion
"""

import argparse
import os
import re
import json
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("stek_bertopic")
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
text_dir = Path(cfg["paths"]["text_dir"])

# ── 0. Install check ──────────────────────────────────────────────────────────
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Run: pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these paths to match your machine
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FOLDER  = Path(cfg["paths"]["text_dir"])
OUTPUT_FOLDER = Path(cfg["paths"]["results_dir"])

# German stopwords — domain-specific STEK terms added
GERMAN_STOPWORDS = [
    # articles + pronouns
    "der","die","das","den","dem","des","ein","eine","einem","einen","einer",
    "eines","und","oder","aber","auch","sich","bei","mit","von","zu","zur",
    "zum","auf","in","an","im","ist","sind","wird","wurde","werden","war",
    "haben","hat","haben","ich","wir","sie","er","es","ihr","uns","man",
    "nicht","noch","wie","wenn","dann","dass","als","aus","nach","vor",
    "über","unter","zwischen","durch","für","gegen","ohne","um","bis","seit",
    "schon","nur","so","aber","doch","sehr","mehr","alle","viele","kann",
    "muss","soll","hier","dort","wo","was","wer","wie","ob","dass","weil",
    # STEK domain — too common to be topic-discriminating
    "heidelberg","stadt","stek","stadtentwicklung","konzept","entwicklung",
    "jahr","jahre","bereich","thema","rahmen","ziel","maßnahme","prozess",
    "beitrag","bürger","bürgerinnen","verwaltung","gemeinderat","stadtrat",
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Read and clean all TXT files
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove SOURCE headers, page numbers, and normalize whitespace."""
    text = re.sub(r'SOURCE:.*\n', '', text)
    text = re.sub(r'={10,}', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # fix German hyphenation across line breaks
    text = re.sub(r'([a-zäöüß])-\s*\n\s*([a-zäöüß])', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── chunking config ────────────────────────────────────────────────────────
CHUNK_SIZE    = 200   # words per chunk — increase for more context
CHUNK_OVERLAP = 50    # words overlap between chunks — avoids cutting ideas

def sliding_window_chunks(text: str,
                          window: int = CHUNK_SIZE,
                          overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into fixed-size word windows with overlap.

    Example (window=5, overlap=2):
      words = [A B C D E F G H]
      chunk1 = A B C D E
      chunk2 =     C D E F G
      chunk3 =         E F G H

    Overlap ensures that ideas split across boundaries
    are still captured in at least one chunk.
    """
    words = text.split()
    if len(words) < window:
        # document too short — return it as one chunk
        return [text] if len(text.strip()) > 40 else []

    step = window - overlap   # how many words to advance each time
    chunks = []
    for i in range(0, len(words) - window + 1, step):
        chunk = " ".join(words[i:i + window])
        chunks.append(chunk)

    # catch the final leftover words if any
    remainder = words[-(len(words) % step or step):]
    if len(remainder) >= window // 2:  # only if at least half a window
        chunks.append(" ".join(remainder))

    return chunks


def read_and_chunk(filepath: str) -> list[str]:
    """Read one TXT file, clean it, and split into sliding window chunks."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_text(text)

    chunks = sliding_window_chunks(text, window=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    return chunks


def load_corpus(folder: str) -> tuple[list[str], list[str]]:
    """Load all TXT files → returns (all_chunks, file_origins)."""
    txt_files = sorted(
        f for f in os.listdir(folder) if f.endswith('.txt')
    )
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")

    all_chunks, origins = [], []
    log.info("Found %d TXT files", len(txt_files))

    for fname in txt_files:
        path = os.path.join(folder, fname)
        chunks = read_and_chunk(path)
        log.info("  %-60s → %d chunks", fname, len(chunks))
        all_chunks.extend(chunks)
        origins.extend([fname] * len(chunks))

    log.info("Total chunks: %d", len(all_chunks))
    return all_chunks, origins


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build BERTopic model (German + CPU optimised)
# ─────────────────────────────────────────────────────────────────────────────

def build_bertopic_model(n_chunks: int) -> BERTopic:
    """
    Assemble BERTopic with components tuned for:
      - German language
      - CPU-only inference
      - Small-to-medium corpus (hundreds to low thousands of chunks)
    """

    # ── Embedding model ───────────────────────────────────────────────────────
    # paraphrase-multilingual-MiniLM-L12-v2:
    #   • trained on 50+ languages including German
    #   • 384-dim dense vectors — fast on CPU
    #   • understands synonyms: "Mietpreis" ≈ "Wohnkosten"
    log.info("Loading SentenceTransformer (German, CPU) ...")
    embedding_model = SentenceTransformer(
        "PM-AI/bi-encoder_msmarco_bert-base_german",  # German BERT
        device="cpu"               # force CPU — safe for your 920MX
    )

    # ── UMAP — dimensionality reduction ───────────────────────────────────────
    # reduces 384-dim embeddings to 5-dim before clustering
    # n_neighbors: lower = more local structure preserved
    # metric: cosine is better than euclidean for sentence embeddings
    umap_model = UMAP(
        n_neighbors   = min(15, n_chunks // 10),  # adapt to corpus size
        n_components  = 5,          # reduce to 5D before HDBSCAN
        min_dist      = 0.0,        # pack clusters tightly
        metric        = "cosine",   # best for sentence embeddings
        random_state  = 42,
        low_memory    = True        # important for CPU
    )

    # ── HDBSCAN — clustering ──────────────────────────────────────────────────
    # min_cluster_size: minimum chunks to form one topic
    # prediction_data=True: needed for topic assignment of new documents
    min_cluster = max(10, n_chunks // 50)
    log.info("HDBSCAN min_cluster_size = %d", min_cluster)
    hdbscan_model = HDBSCAN(
        min_cluster_size    = min_cluster,
        min_samples         = 3,
        metric              = "euclidean",   # operates on UMAP output (5D)
        cluster_selection_epsilon = 0.2,     # merge nearby micro-clusters
        prediction_data     = True
    )

    # ── CountVectorizer — topic word extraction ───────────────────────────────
    # BERTopic uses c-TF-IDF on top of cluster assignments
    # ngram_range=(1,2): allows bigrams like "bezahlbarer_wohnraum"
    # min_df=2: ignore words appearing in only 1 chunk
    vectorizer_model = CountVectorizer(
        stop_words    = GERMAN_STOPWORDS,
        ngram_range   = (1, 2),      # unigrams + bigrams
        min_df        = 2,
        max_features  = 10000
    )

    # ── Assemble BERTopic ─────────────────────────────────────────────────────
    topic_model = BERTopic(
        embedding_model   = embedding_model,
        umap_model        = umap_model,
        hdbscan_model     = hdbscan_model,
        vectorizer_model  = vectorizer_model,
        top_n_words       = 10,       # keywords per topic
        nr_topics         = "auto",   # let HDBSCAN decide
        calculate_probabilities = False,  # faster on CPU
        verbose           = True
    )

    return topic_model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run and save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    topic_model: BERTopic,
    topics: list[int],
    chunks: list[str],
    origins: list[str],
    out_folder: str
):
    """Write all result files to out_folder."""
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info.Topic != -1])
    n_noise    = int((np.array(topics) == -1).sum())

    log.info("Topics found  : %d", n_topics)
    log.info("Noise chunks  : %d / %d", n_noise, len(chunks))

    # ── topic_summary.txt ─────────────────────────────────────────────────────
    lines = [
        "STEK 2035 — BERTopic Topic Summary",
        "=" * 60,
        f"Total chunks   : {len(chunks)}",
        f"Topics found   : {n_topics}",
        f"Noise chunks   : {n_noise} ({100*n_noise/len(chunks):.1f}%)",
        "",
    ]
    for _, row in topic_info.iterrows():
        if row.Topic == -1:
            continue
        words = topic_model.get_topic(row.Topic)
        word_str = ", ".join(w for w, _ in words[:10])
        lines.append(f"Topic {row.Topic:02d} ({row.Count:4d} chunks): {word_str}")

    summary_path = os.path.join(out_folder, "topic_summary.txt")
    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved: %s", summary_path)

    # ── topic_chunks.txt ──────────────────────────────────────────────────────
    chunk_lines = ["STEK 2035 — Chunks per Topic", "=" * 60, ""]
    topic_ids = sorted(set(topics))

    for tid in topic_ids:
        label = "NOISE" if tid == -1 else f"Topic {tid:02d}"
        idxs  = [i for i, t in enumerate(topics) if t == tid]

        if tid != -1:
            words = topic_model.get_topic(tid)
            word_str = ", ".join(w for w, _ in words[:6])
            chunk_lines.append(f"{'─'*60}")
            chunk_lines.append(f"{label} | keywords: {word_str} | {len(idxs)} chunks")
        else:
            chunk_lines.append(f"{'─'*60}")
            chunk_lines.append(f"{label} | {len(idxs)} chunks (not assigned to any topic)")

        chunk_lines.append("")
        for rank, i in enumerate(idxs[:15], 1):   # show max 15 chunks per topic
            chunk_lines.append(f"  [{rank:02d}] [{origins[i]}]")
            chunk_lines.append(f"       {chunks[i][:200]}...")
            chunk_lines.append("")

        if len(idxs) > 15:
            chunk_lines.append(f"  ... and {len(idxs)-15} more chunks\n")

    chunks_path = os.path.join(out_folder, "topic_chunks.txt")
    Path(chunks_path).write_text("\n".join(chunk_lines), encoding="utf-8")
    log.info("Saved: %s", chunks_path)

    # ── chunks_stats.txt ──────────────────────────────────────────────────────
    origin_series  = pd.Series(origins)
    topic_series   = pd.Series(topics)
    stats_lines    = ["STEK 2035 — Chunk Statistics", "=" * 60, ""]

    stats_lines.append("Chunks per source file:")
    for fname, count in origin_series.value_counts().items():
        stats_lines.append(f"  {fname:<60} {count:>4} chunks")

    stats_lines += ["", "Chunks per topic:"]
    for tid, count in topic_series.value_counts().sort_index().items():
        label = "Noise (-1)" if tid == -1 else f"Topic {tid:02d}  "
        stats_lines.append(f"  {label}  {count:>4} chunks")

    stats_path = os.path.join(out_folder, "chunks_stats.txt")
    Path(stats_path).write_text("\n".join(stats_lines), encoding="utf-8")
    log.info("Saved: %s", stats_path)

    # ── bertopic_comparison.txt ───────────────────────────────────────────────
    comp = [
        "STEK 2035 — BERTopic vs LDA vs LSA",
        "=" * 60,
        "",
        "APPROACH",
        "  LDA    : probabilistic, each chunk = mixture of topics",
        "  LSA    : algebraic (TF-IDF + SVD), captures global variance",
        "  BERTopic: semantic embeddings + HDBSCAN clustering + c-TF-IDF",
        "",
        "GERMAN LANGUAGE HANDLING",
        "  LDA    : word counts only, misses synonyms",
        "  LSA    : word counts only, misses synonyms",
        "  BERTopic: multilingual transformer — understands that",
        "           'Mietpreis' and 'Wohnkosten' mean the same thing",
        "",
        "NUMBER OF TOPICS",
        "  LDA    : must specify k manually",
        "  LSA    : must specify k manually",
        "  BERTopic: auto-detected by HDBSCAN",
        f"           → found {n_topics} topics in your corpus",
        "",
        "NOISE HANDLING",
        "  LDA    : every chunk is assigned to a topic (no noise)",
        "  LSA    : every chunk is assigned to a topic (no noise)",
        "  BERTopic: chunks that don't fit any topic → label -1 (noise)",
        f"           → {n_noise} chunks ({100*n_noise/len(chunks):.1f}%) marked as noise",
        "",
        "SPEED (CPU only)",
        "  LDA    : fast",
        "  LSA    : very fast",
        "  BERTopic: slow (embedding 2000+ chunks takes ~5-10 min on CPU)",
        "",
        "WHEN TO USE WHICH",
        "  LDA    : small corpus, need topic probabilities per chunk",
        "  LSA    : very fast baseline, interpretable components",
        "  BERTopic: large corpus, German synonyms matter, no k needed",
        "",
        "RESULTS SUMMARY",
        f"  Topics found   : {n_topics}",
        f"  Noise chunks   : {n_noise} / {len(chunks)}",
        "",
        "TOP TOPICS:",
    ]
    for _, row in topic_info[topic_info.Topic != -1].head(10).iterrows():
        words = topic_model.get_topic(row.Topic)
        word_str = ", ".join(w for w, _ in words[:8])
        comp.append(f"  Topic {row.Topic:02d} ({row.Count} chunks): {word_str}")

    comp_path = os.path.join(out_folder, "bertopic_comparison.txt")
    Path(comp_path).write_text("\n".join(comp), encoding="utf-8")
    log.info("Saved: %s", comp_path)

    print("\n" + "="*60)
    print("BERTOPIC RESULTS SUMMARY")
    print("="*60)
    print(f"  Topics found : {n_topics}")
    print(f"  Noise chunks : {n_noise} / {len(chunks)} ({100*n_noise/len(chunks):.1f}%)")
    print(f"  Output saved : {out_folder}")
    print("="*60)
    print("\nTop topics:")
    for _, row in topic_info[topic_info.Topic != -1].head(8).iterrows():
        words = topic_model.get_topic(row.Topic)
        word_str = ", ".join(w for w, _ in words[:6])
        print(f"  Topic {row.Topic:02d} ({row.Count:3d} chunks): {word_str}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("STEK 2035 — BERTopic Pipeline starting")
    log.info("Input  : %s", INPUT_FOLDER)
    log.info("Output : %s", OUTPUT_FOLDER)

    # step 1 — load corpus
    chunks, origins = load_corpus(INPUT_FOLDER)

    if len(chunks) < 20:
        raise ValueError(
            f"Only {len(chunks)} chunks found — too few for BERTopic. "
            "Check your INPUT_FOLDER path."
        )

    # step 2 — build model
    topic_model = build_bertopic_model(len(chunks))

    # step 3 — fit and transform
    log.info("Fitting BERTopic on %d chunks (this may take several minutes on CPU)...", len(chunks))
    topics, _ = topic_model.fit_transform(chunks)

    # step 4 — save all results
    save_results(topic_model, topics, chunks, origins, OUTPUT_FOLDER)

    log.info("Done. All results in: %s", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()