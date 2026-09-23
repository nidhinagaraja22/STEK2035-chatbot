"""
STEK 2035 — The Real Test: Citizen-Content Contamination on Policy Questions
================================================================================
Document-level recall/MRR/nDCG (used throughout this project's earlier
comparisons) only measures "did the target document appear somewhere
in top-5" — it says NOTHING about what the OTHER retrieved chunks are,
or whether a POLICY question ends up with citizen-opinion-dominated
context. This directly tests THAT specific, real concern instead.

For every question explicitly marked is_official_policy=True in the
ground truth, this measures: what FRACTION of each strategy's
retrieved top-5 chunks come from citizen tiers (L4/L5) rather than
official tiers (L1/L2)?

This is the metric that actually matters for the misattribution risk
this project exists to prevent — not the document-recall metric used
in earlier comparisons, which was measuring a different (real, but
narrower) question.

Usage:
    python stek_test_policy_question_contamination.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py (imported)
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve

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


def citizen_fraction(authority_levels: list) -> float:
    """Fraction of retrieved chunks from L4/L5 (citizen/workshop tiers)."""
    if not authority_levels:
        return 0.0
    citizen_count = sum(1 for a in authority_levels if a >= 4)
    return citizen_count / len(authority_levels)


def main():
    print("=" * 70)
    print("The Real Test: Citizen-Content Contamination on Policy Questions")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    doc_ids = [c["document_id"] for c in chunks]
    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    policy_qs = [q for q in gt_data["questions"] if q.get("is_official_policy") is True and q.get("relevant_chunks")]
    print(f"\nTesting {len(policy_qs)} questions explicitly marked as OFFICIAL POLICY questions\n")

    plain_fractions = []
    cascade_fractions = []
    per_question = []

    for q in policy_qs:
        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec

        # plain top-k
        plain_top5_idx = np.argsort(-sims)[:5]
        plain_authorities = [chunks[i]["authority_level"] for i in plain_top5_idx]
        plain_frac = citizen_fraction(plain_authorities)
        plain_fractions.append(plain_frac)

        # threshold cascade
        cascade_result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model,
                                              official_idx, l4_idx, l5_idx)
        cascade_authorities = [chunks[i]["authority_level"] for i in cascade_result["result_idx"]]
        cascade_frac = citizen_fraction(cascade_authorities)
        cascade_fractions.append(cascade_frac)

        marker = "⚠️ " if plain_frac > 0.5 else "  "
        print(f"{marker}{q['q_id']:<6} plain_citizen_frac={plain_frac:.2f}  "
              f"cascade_citizen_frac={cascade_frac:.2f}  "
              f"plain_authorities={plain_authorities}")

        per_question.append({"q_id": q["q_id"], "plain_citizen_fraction": plain_frac,
                                "cascade_citizen_fraction": cascade_frac,
                                "plain_authorities": plain_authorities, "cascade_authorities": cascade_authorities})

    print(f"\n{'='*70}")
    print("SUMMARY — Citizen-Content Fraction on POLICY Questions Specifically")
    print(f"{'='*70}")
    print(f"Plain top-k:        mean_citizen_fraction={np.mean(plain_fractions):.3f}  "
          f"({sum(1 for f in plain_fractions if f > 0.5)}/{len(plain_fractions)} questions MAJORITY citizen content)")
    print(f"Threshold cascade:  mean_citizen_fraction={np.mean(cascade_fractions):.3f}  "
          f"({sum(1 for f in cascade_fractions if f > 0.5)}/{len(cascade_fractions)} questions MAJORITY citizen content)")

    print(f"\n👉 This is the metric that actually tests the misattribution concern —")
    print(f"   NOT document-level recall, which was silent on this question entirely.")
    print(f"   A high plain-top-k citizen fraction here would directly confirm the")
    print(f"   real, specific risk being raised: policy questions receiving")
    print(f"   citizen-dominated context regardless of document-recall performance.")

    with open("policy_contamination_results.json", "w", encoding="utf-8") as f:
        json.dump({"plain_mean_citizen_fraction": float(np.mean(plain_fractions)),
                    "cascade_mean_citizen_fraction": float(np.mean(cascade_fractions)),
                    "per_question": per_question}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: policy_contamination_results.json")


if __name__ == "__main__":
    main()
