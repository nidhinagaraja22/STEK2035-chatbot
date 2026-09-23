"""
STEK 2035 — Semantic Nearest-Neighbor Vague Detection (Third Signal Test)
==============================================================================
Tests a DIFFERENT kind of vague-detection signal from Task 10's two
word-based signals: does a question's EMBEDDING sit close to the
embeddings of KNOWN vague example questions?

Rationale: the word-based signals (Task 10) work by structural
analysis (strip frame words, count what remains) — they can miss a
NEW vague phrasing that doesn't match the known frame-word patterns.
Semantic similarity to known vague EXAMPLES could generalize
differently — catching vague questions that "feel" similar in
meaning even if worded completely differently.

Method: for each test question, compute its max cosine similarity
against the 6 known vague example questions' embeddings. If that
max similarity exceeds a threshold, flag as vague.

This is tested the SAME rigorous way as Task 10's signals — against
all 6 known vague questions (leave-one-out, so a question is never
compared against itself) and all 48 real answerable questions, to
see if it achieves comparable or complementary accuracy.

Usage:
    python stek_semantic_vague_detection_test.py

Requires:
    corpus/corpus_v2/embeddings_v2_e5base.npy (not used directly —
        this only needs the embedding MODEL, not the corpus)
    stek_task4_5_master_ground_truth.json
"""

from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

VAGUE_EXAMPLES = [
    "Ich möchte etwas über Heidelberg erfahren.",
    "Was plant Heidelberg?",
    "Was sagt STEK 2035?",
    "Wie ist Heidelberg?",
    "Was gibt's Neues?",
    "Erzähl mir was.",
]


def normalise(v):
    return v / max(np.linalg.norm(v), 1e-8)


def main():
    print("=" * 70)
    print("Semantic Nearest-Neighbor Vague Detection — Third Signal Test")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding the 6 known vague examples...")
    vague_embeddings = np.array([normalise(model.encode(f"query: {q}")) for q in VAGUE_EXAMPLES])

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    answerable = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    vague_qs = [q for q in gt_data["questions"] if q["category"] == "vague"]

    print(f"\nTesting against {len(vague_qs)} vague and {len(answerable)} answerable questions\n")

    # ── vague questions: leave-one-out (never compare a question to itself) ──
    print("=" * 70)
    print("VAGUE QUESTIONS (leave-one-out max similarity to the OTHER 5)")
    print("=" * 70)
    vague_max_sims = []
    for i, q in enumerate(vague_qs):
        q_vec = normalise(model.encode(f"query: {q['question']}"))
        other_embeddings = np.delete(vague_embeddings, i, axis=0)
        max_sim = float(np.max(other_embeddings @ q_vec))
        vague_max_sims.append(max_sim)
        print(f"  {q['q_id']}: max_sim_to_other_vague={max_sim:.4f}  '{q['question']}'")

    # ── answerable questions: max similarity to ALL 6 vague examples ─────────
    print(f"\n{'='*70}")
    print("ANSWERABLE QUESTIONS (max similarity to all 6 known vague examples)")
    print(f"{'='*70}")
    answerable_max_sims = []
    for q in answerable:
        q_vec = normalise(model.encode(f"query: {q['question']}"))
        max_sim = float(np.max(vague_embeddings @ q_vec))
        answerable_max_sims.append(max_sim)
        print(f"  {q['q_id']}: max_sim_to_vague_examples={max_sim:.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Vague questions:      min={min(vague_max_sims):.4f}  "
          f"max={max(vague_max_sims):.4f}  mean={np.mean(vague_max_sims):.4f}")
    print(f"Answerable questions: min={min(answerable_max_sims):.4f}  "
          f"max={max(answerable_max_sims):.4f}  mean={np.mean(answerable_max_sims):.4f}")

    overlap = max(answerable_max_sims) >= min(vague_max_sims)
    print(f"\nDo the ranges OVERLAP: {overlap}")
    if overlap:
        n_overlap = sum(1 for s in answerable_max_sims if s >= min(vague_max_sims))
        print(f"  {n_overlap} of {len(answerable)} real answerable questions fall")
        print(f"  at or above the lowest vague question's score — this signal")
        print(f"  alone would create false positives at any threshold that")
        print(f"  catches all 6 vague questions.")
    else:
        print(f"  ✅ Clean separation — this signal could work as a genuine")
        print(f"  usable threshold on its own.")

    with open("semantic_vague_detection_results.json", "w", encoding="utf-8") as f:
        json.dump({"vague_max_sims": vague_max_sims, "answerable_max_sims": answerable_max_sims,
                    "ranges_overlap": overlap}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: semantic_vague_detection_results.json")


if __name__ == "__main__":
    main()
