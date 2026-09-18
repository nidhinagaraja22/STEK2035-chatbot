"""
STEK 2035 — Topic-Concentration Signal for Unanswerable Detection
======================================================================
Hypothesis (different from, and more targeted than, the whole-corpus
top_score/spread test that failed earlier): an ANSWERABLE question's
similarity should CONCENTRATE in one dominant LDA topic, with
multiple supporting chunks. An UNANSWERABLE question's similarity
should SCATTER across many topics roughly evenly, since nothing is
genuinely relevant to concentrate around.

Three signals computed PER TOPIC (k=8 LDA topics), then combined:

  topic_relevance = the single highest per-topic max-similarity
                     (how well does the BEST topic match, at all)

  topic_diversity = how many DIFFERENT topics have a max-similarity
                     within a small margin of the overall best score
                     (competing topics, not one clear winner)

  topic_counts     = within the WINNING topic only, how many chunks
                     exceed a support threshold (breadth of backing
                     for the best topic, not just one lucky chunk)

Prediction:
  ANSWERABLE:    LOW topic_diversity (one clear winner),
                 HIGH topic_counts within that winner
  UNANSWERABLE:  HIGH topic_diversity (no clear winner),
                 LOW topic_counts everywhere

Usage:
    python stek_topic_concentration_signal.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    vector_store/lda_topics.json   (per-chunk topic labels, k=8 —
        falls back to a per-chunk "topic"/"lda_topic" field inside
        the chunks file itself if this separate file isn't found)
    stek_task4_5_master_ground_truth.json
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
LDA_TOPICS_PATH = Path("corpus/corpus_v2/lda_topics_v2.json")  # from stek_recompute_topics.py
LSA_TOPICS_PATH = Path("corpus/corpus_v2/lsa_topics_v2.json")  # from stek_recompute_topics.py
TOPIC_SOURCE = "lda"  # "lsa" or "lda" — switched to LDA: LSA's topic 0 absorbed
                        # 980 of 1258 chunks (78%), making diversity/concentration
                        # measurements meaningless for most of the corpus. LDA's
                        # distribution (421/194/275/115/253) is far more balanced.
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

DIVERSITY_MARGIN = 0.02       # a topic "competes" with the best if within this margin
SUPPORT_THRESHOLD_MARGIN = 0.05  # a chunk "supports" its topic if within this of that topic's max


def load_chunks_with_topics():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    # use the recomputed LSA/LDA output (correct format: a top-level
    # "chunk_topic_assignment" list, aligned by row index to the chunks
    # file — produced by stek_recompute_topics.py)
    recomputed_path = LSA_TOPICS_PATH if TOPIC_SOURCE == "lsa" else LDA_TOPICS_PATH
    topic_field = "lsa_topic" if TOPIC_SOURCE == "lsa" else "lda_topic"

    if recomputed_path.exists():
        with open(recomputed_path, encoding="utf-8") as f:
            topic_data = json.load(f)
        assignments = topic_data["chunk_topic_assignment"]
        if len(assignments) != len(chunks):
            print(f"⚠️  WARNING: {len(assignments)} topic assignments vs {len(chunks)} "
                  f"chunks — mismatch, results will be unreliable.")
        for c, a in zip(chunks, assignments):
            c["_topic"] = a[topic_field]
        print(f"Loaded {TOPIC_SOURCE.upper()} topic assignments from: {recomputed_path}")
    else:
        print(f"⚠️  {recomputed_path} not found — falling back to inline "
              f"'lda_topic'/'topic' field, or document_id as a crude proxy.")
        for c in chunks:
            c["_topic"] = c.get("lda_topic", c.get("topic", c.get("document_id")))

    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def compute_topic_signals(question, model, chunk_embeddings, chunks):
    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec

    # group similarities by topic
    topic_sims = defaultdict(list)
    for i, c in enumerate(chunks):
        topic_sims[c["_topic"]].append(sims[i])

    topic_max = {t: max(s) for t, s in topic_sims.items()}
    best_topic = max(topic_max, key=topic_max.get)
    topic_relevance = topic_max[best_topic]

    # diversity: how many topics compete within DIVERSITY_MARGIN of the best
    competing = sum(1 for v in topic_max.values() if v >= topic_relevance - DIVERSITY_MARGIN)
    topic_diversity = competing

    # counts: within the WINNING topic, how many chunks support it
    winning_topic_sims = topic_sims[best_topic]
    support_count = sum(1 for v in winning_topic_sims if v >= topic_relevance - SUPPORT_THRESHOLD_MARGIN)

    return {"topic_relevance": float(topic_relevance), "topic_diversity": topic_diversity,
            "topic_counts": support_count, "best_topic": best_topic, "n_topics_total": len(topic_max)}


def categorize_question(q):
    if q["category"] in ("vague",):
        return "vague"
    if q["category"] in ("unanswerable", "semi_answerable") or not q.get("answerable", True):
        return "unanswerable"
    if q.get("relevant_chunks"):
        return "answerable"
    return None


def main():
    print("=" * 70)
    print("Topic-Concentration Signal for Unanswerable Detection")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks_with_topics()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    n_unique_topics = len(set(c["_topic"] for c in chunks))
    print(f"  {len(chunks)} chunks across {n_unique_topics} distinct topic labels")

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    groups = {"answerable": [], "vague": [], "unanswerable": []}

    for q in gt_data["questions"]:
        category = categorize_question(q)
        if category is None:
            continue
        signals = compute_topic_signals(q["question"], model, chunk_embeddings, chunks)
        groups[category].append({"q_id": q["q_id"], **signals})
        print(f"  [{category:<12}] {q['q_id']}: relevance={signals['topic_relevance']:.4f}  "
              f"diversity={signals['topic_diversity']}/{signals['n_topics_total']}  "
              f"counts={signals['topic_counts']}")

    print(f"\n{'='*70}")
    print("GROUP STATISTICS")
    print(f"{'='*70}")
    for cat in ["answerable", "vague", "unanswerable"]:
        rows = groups[cat]
        if not rows:
            continue
        rel = [r["topic_relevance"] for r in rows]
        div = [r["topic_diversity"] for r in rows]
        cnt = [r["topic_counts"] for r in rows]
        print(f"\n{cat.upper()} (n={len(rows)}):")
        print(f"  topic_relevance: mean={statistics.mean(rel):.4f}  min={min(rel):.4f}  max={max(rel):.4f}")
        print(f"  topic_diversity: mean={statistics.mean(div):.2f}  min={min(div)}  max={max(div)}")
        print(f"  topic_counts:    mean={statistics.mean(cnt):.2f}  min={min(cnt)}  max={max(cnt)}")

    with open("topic_concentration_results.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: topic_concentration_results.json")


if __name__ == "__main__":
    main()