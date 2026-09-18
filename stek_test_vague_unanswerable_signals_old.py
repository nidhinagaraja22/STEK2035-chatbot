"""
STEK 2035 — Testing Candidate-Pool Statistics for Vague/Unanswerable Detection
===================================================================================
Tests whether two PRE-SELECTION candidate-pool statistics can
distinguish vague questions from unanswerable questions from normal
answerable questions — using the same real ground-truth categories
(6 vague, 6 unanswerable, 48 answerable) already used to calibrate
Task 10's word-count threshold.

Two DIFFERENT hypothesized signals, since vague and unanswerable are
different failure modes:

  top_score    = highest similarity across ALL corpus chunks
                 Hypothesis: UNANSWERABLE questions show a distinctly
                 LOW top_score — nothing in the corpus is genuinely
                 relevant, so even the best match is weak.

  score_spread = standard deviation of similarity across the top-20
                 candidates
                 Hypothesis: VAGUE questions show a distinctly LOW
                 spread — nothing discriminates itself as clearly
                 more relevant than anything else. This matches a
                 real, already-observed finding: Q046 ("Ich möchte
                 etwas über Heidelberg erfahren") showed retrieval
                 scores clustered in a 0.863-0.868 range, spread of
                 only 0.005.

Usage:
    python stek_test_vague_unanswerable_signals.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
"""

import json
import statistics
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

TOP_N_FOR_SPREAD = 20


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def compute_pool_statistics(question, model, chunk_embeddings):
    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec
    top_sorted = np.sort(sims)[::-1]

    top_score = float(top_sorted[0])
    top_n = top_sorted[:TOP_N_FOR_SPREAD]
    score_spread = float(np.std(top_n))

    return top_score, score_spread


def categorize_question(q):
    if q["category"] in ("vague",):
        return "vague"
    if q["category"] in ("unanswerable",) or not q.get("answerable", True):
        return "unanswerable"
    if q.get("relevant_chunks"):
        return "answerable"
    return None


def main():
    print("=" * 70)
    print("Testing Candidate-Pool Statistics for Vague/Unanswerable Detection")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    groups = {"vague": [], "unanswerable": [], "answerable": []}

    for q in gt_data["questions"]:
        category = categorize_question(q)
        if category is None:
            continue
        top_score, score_spread = compute_pool_statistics(q["question"], model, chunk_embeddings)
        groups[category].append({"q_id": q["q_id"], "question": q["question"],
                                    "top_score": top_score, "score_spread": score_spread})
        print(f"  [{category:<12}] {q['q_id']}: top_score={top_score:.4f}  spread={score_spread:.4f}")

    print(f"\n{'='*70}")
    print("GROUP STATISTICS")
    print(f"{'='*70}")
    for cat in ["answerable", "vague", "unanswerable"]:
        rows = groups[cat]
        if not rows:
            continue
        top_scores = [r["top_score"] for r in rows]
        spreads = [r["score_spread"] for r in rows]
        print(f"\n{cat.upper()} (n={len(rows)}):")
        print(f"  top_score:    mean={statistics.mean(top_scores):.4f}  "
              f"min={min(top_scores):.4f}  max={max(top_scores):.4f}")
        print(f"  score_spread: mean={statistics.mean(spreads):.4f}  "
              f"min={min(spreads):.4f}  max={max(spreads):.4f}")

    with open("vague_unanswerable_signal_test.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: vague_unanswerable_signal_test.json")
    print(f"\n👉 Look for CLEAN SEPARATION between group ranges above —")
    print(f"   if unanswerable's top_score max is below answerable's top_score")
    print(f"   min, that's a usable threshold. Same check for vague's")
    print(f"   score_spread vs the other two groups.")


if __name__ == "__main__":
    main()
