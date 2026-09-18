"""
STEK 2035 — Precision@k, Recall@k, MRR for the Cascading Retrieval System
=============================================================================
Computes real IR metrics for all 45 verified ground-truth questions,
using the SAME cascading retrieval logic already validated (71%
document-level recall on the last full run).

Uses DOCUMENT-LEVEL relevance, not exact chunk_id — established
earlier this session as the reliable metric given the v2/v3 chunk
numbering mismatch (ground truth chunk_ids were verified against
corpus_v3's chunking, not v2's).

Multi-document questions (e.g. Q034 needs BOTH zukunftsreise_2035
AND online_beteiligung_2024) are handled properly: Recall@k is the
FRACTION of required documents actually represented in the top-k,
not a binary hit/miss — this correctly penalizes partial coverage
without treating it as a total failure.

Usage:
    python stek_precision_recall_cascading.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

TOP_K = 5
COVERAGE_THRESHOLD = 0.75

CITIZEN_REQUEST_KEYWORDS = [
    "wünschen sich bürger", "wünscht sich", "bürger fordern", "fordern bürger",
    "was denken bürger", "bürgermeinung", "was wollen bürger",
    "was war der zweck", "wie wurden bürgerinnen", "räume für kulturelle nutzung fehlten",
]
COMPARISON_KEYWORDS = [
    "im vergleich", "verglichen mit", "unterschied zwischen",
    "bürgerwünsche und offizielle", "einerseits", "andererseits",
]


def classify_request_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in COMPARISON_KEYWORDS):
        return "comparison"
    if any(kw in q for kw in CITIZEN_REQUEST_KEYWORDS):
        return "citizen_explicit"
    return "default_cascade"


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


def search_tier(sims, tier_idx, k):
    if len(tier_idx) == 0:
        return [], 0.0
    tier_sims = sims[tier_idx]
    ranked = tier_idx[np.argsort(-tier_sims)][:k]
    return list(ranked), float(tier_sims.max())


def cascading_retrieve(question, sims, official_idx, l4_idx, l5_idx, k=TOP_K):
    request_type = classify_request_type(question)

    if request_type == "citizen_explicit":
        citizen_idx = np.concatenate([l4_idx, l5_idx])
        result_idx, _ = search_tier(sims, citizen_idx, k)
        return result_idx, request_type

    if request_type == "comparison":
        off_result, _ = search_tier(sims, official_idx, k - 2)
        cit_result, _ = search_tier(sims, np.concatenate([l4_idx, l5_idx]), 2)
        return off_result + cit_result, request_type

    off_result, off_score = search_tier(sims, official_idx, k)
    if off_score >= COVERAGE_THRESHOLD:
        return off_result, request_type
    l4_result, l4_score = search_tier(sims, l4_idx, k)
    if l4_score >= COVERAGE_THRESHOLD:
        return l4_result, request_type
    l5_result, _ = search_tier(sims, l5_idx, k)
    return l5_result, request_type


def precision_at_k(retrieved_docs, target_docs, k):
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in target_docs)
    return hits / len(top_k)


def recall_at_k(retrieved_docs, target_docs, k):
    """Fraction of REQUIRED documents represented in top-k — correctly
    handles multi-document questions with partial credit."""
    if not target_docs:
        return 0.0
    top_k_docs = set(retrieved_docs[:k])
    covered = len(target_docs & top_k_docs)
    return covered / len(target_docs)


def reciprocal_rank(retrieved_docs, target_docs):
    for rank, d in enumerate(retrieved_docs, start=1):
        if d in target_docs:
            return 1.0 / rank
    return 0.0


def main():
    print("=" * 70)
    print("STEK 2035 — Precision/Recall/MRR for Cascading Retrieval")
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
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]
    doc_ids = [c["document_id"] for c in chunks]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    print(f"Evaluating {len(questions)} verified questions\n")

    all_precision, all_recall, all_rr = [], [], []
    per_question = []

    for q in questions:
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}

        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec

        result_idx, req_type = cascading_retrieve(q["question"], sims, official_idx, l4_idx, l5_idx)
        retrieved_docs = [doc_ids[i] for i in result_idx]

        p = precision_at_k(retrieved_docs, target_docs, TOP_K)
        r = recall_at_k(retrieved_docs, target_docs, TOP_K)
        rr = reciprocal_rank(retrieved_docs, target_docs)

        all_precision.append(p)
        all_recall.append(r)
        all_rr.append(rr)

        per_question.append({
            "q_id": q["q_id"], "request_type": req_type,
            "precision_at_5": round(p, 3), "recall_at_5": round(r, 3),
            "reciprocal_rank": round(rr, 3),
            "target_docs": list(target_docs), "retrieved_docs": retrieved_docs,
        })

        marker = "✅" if r >= 0.99 else ("🟡" if r > 0 else "❌")
        print(f"{marker} {q['q_id']:<6} P@5={p:.2f} R@5={r:.2f} RR={rr:.2f}  "
              f"target={target_docs}")

    mean_p = np.mean(all_precision)
    mean_r = np.mean(all_recall)
    mrr = np.mean(all_rr)

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"  Mean Precision@5 : {mean_p:.4f}")
    print(f"  Mean Recall@5    : {mean_r:.4f}")
    print(f"  MRR              : {mrr:.4f}")
    print(f"  Questions evaluated: {len(questions)}")

    with open("precision_recall_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "mean_precision_at_5": mean_p, "mean_recall_at_5": mean_r, "mrr": mrr,
            "n_questions": len(questions), "per_question": per_question,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: precision_recall_results.json")

    print(f"\n⚠️  Reminder: Recall@5 for multi-document questions (e.g. Q034,")
    print(f"   Q043, Q045) is PARTIAL CREDIT — 0.5 means half the required")
    print(f"   documents were found, not a failure. See per_question detail")
    print(f"   in the saved JSON for exact document-level breakdown.")


if __name__ == "__main__":
    main()
