"""
STEK 2035 — Does Lowering the Threshold from 0.75 to 0.50 Actually Do Anything?
====================================================================================
Before changing the threshold, check where official-tier best scores
ACTUALLY fall across all default_cascade questions. Given this
session's repeated finding that embeddings cluster tightly
(anisotropy — most content scores 0.75-0.90 regardless of true
relevance), it's possible official-tier content already almost
always clears 0.75, meaning the threshold rarely binds at all —
in which case lowering it to 0.50 would change nothing measurable.

Usage:
    python stek_check_threshold_sensitivity.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py (imported, for classify_request_type)
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import classify_request_type

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")


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


def main():
    print("=" * 70)
    print("Does Lowering the Threshold Actually Do Anything?")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    default_cascade_qs = [q for q in questions if classify_request_type(q["question"]) == "default_cascade"]
    print(f"\nChecking official-tier best score for {len(default_cascade_qs)} default_cascade questions\n")

    above_075 = 0
    between_050_075 = 0
    below_050 = 0
    scores = []

    for q in default_cascade_qs:
        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        official_best = float(np.max(sims[official_idx]))
        scores.append(official_best)

        if official_best >= 0.75:
            above_075 += 1
            band = "≥0.75 (unaffected by lowering — already passes both)"
        elif official_best >= 0.50:
            between_050_075 += 1
            band = "0.50-0.75 (WOULD CHANGE — now passes at 0.50, didn't at 0.75)"
        else:
            below_050 += 1
            band = "<0.50 (unaffected — falls through either way)"

        print(f"  {q['q_id']:<6}: official_best_score={official_best:.4f}  [{band}]")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  ≥0.75 (threshold change has NO effect):        {above_075}/{len(default_cascade_qs)}")
    print(f"  0.50-0.75 (threshold change WOULD flip these): {between_050_075}/{len(default_cascade_qs)}")
    print(f"  <0.50 (threshold change has NO effect):        {below_050}/{len(default_cascade_qs)}")
    print(f"\n  Mean official-tier best score: {np.mean(scores):.4f}")
    print(f"  Min: {min(scores):.4f}  Max: {max(scores):.4f}")

    print(f"\n👉 Only the middle band ({between_050_075} questions) would actually")
    print(f"   change behavior if you lower the threshold to 0.50. If that number")
    print(f"   is small or zero, lowering the threshold will have little to no")
    print(f"   measurable effect on your results — confirming or refuting the")
    print(f"   anisotropy concern with real data before you spend time re-testing.")


if __name__ == "__main__":
    main()
