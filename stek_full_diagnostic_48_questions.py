"""
STEK 2035 — Full Diagnostic: All 48 Questions, Real Scores
================================================================
For every answerable question in the ground truth, using the
SELECTED, deployed strategy (threshold-gated cascade):

  - Request classification (default_cascade / citizen_explicit / comparison)
  - Actual retrieved chunks with real similarity scores and authority levels
  - Document-level hit/miss against ground truth
  - Citizen-content fraction (the misattribution-risk metric)

One comprehensive report, reusing all already-validated components
from this project — no new logic, just combining what's already
been separately confirmed correct into a single per-question view.

Usage:
    python stek_full_diagnostic_48_questions.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl (the CORRECTLY L4/L5-split version)
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py (imported)
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve, classify_request_type, CITATION_PREFIX

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
    if not authority_levels:
        return 0.0
    return sum(1 for a in authority_levels if a >= 4) / len(authority_levels)


def main():
    print("=" * 70)
    print("Full Diagnostic — All 48 Answerable Questions, Real Scores")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    # sanity check — confirm the L4/L5 split is actually correctly applied
    # before trusting anything downstream (this exact check caught a real,
    # silent corpus bug earlier this session)
    online_bet_levels = {c.get("authority_level") for c in chunks if c["document_id"] == "online_beteiligung_2024"}
    if online_bet_levels != {5}:
        print(f"\n⚠️  WARNING: online_beteiligung_2024 shows authority_level(s) {online_bet_levels},")
        print(f"   expected only {{5}}. The L4/L5 split may not be correctly applied to this")
        print(f"   corpus file — results below may not be trustworthy. Verify CHUNKS_PATH")
        print(f"   points to the genuinely split corpus before trusting this diagnostic.\n")
    else:
        print(f"  ✅ L4/L5 split confirmed correct (online_beteiligung_2024 = L5 only)\n")

    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]
    doc_ids = [c["document_id"] for c in chunks]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    print(f"Diagnosing {len(questions)} answerable questions\n")

    results = []
    hits = 0

    for q in questions:
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}
        request_type = classify_request_type(q["question"])
        is_policy = q.get("is_official_policy") is True

        result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model,
                                      official_idx, l4_idx, l5_idx)
        retrieved_idx = result["result_idx"]
        retrieved_docs = [doc_ids[i] for i in retrieved_idx]
        retrieved_authorities = [chunks[i]["authority_level"] for i in retrieved_idx]
        retrieved_scores = [float(result["sims"][i]) for i in retrieved_idx]

        hit = bool(target_docs & set(retrieved_docs))
        if hit:
            hits += 1
        cit_frac = citizen_fraction(retrieved_authorities)

        print(f"\n{'='*70}")
        print(f"{q['q_id']} [{'POLICY' if is_policy else 'non-policy'}] [{request_type}]")
        print(f"Q: {q['question']}")
        print(f"{'='*70}")
        print(f"Target document(s): {target_docs}")
        print(f"Tier used: {result['tier_used']}")
        print(f"{'✅ HIT' if hit else '❌ MISS'}   citizen_fraction={cit_frac:.2f}")
        print(f"\nRetrieved (rank : score : authority : document):")
        for rank, (doc, auth, score) in enumerate(zip(retrieved_docs, retrieved_authorities, retrieved_scores), 1):
            marker = "→" if doc in target_docs else " "
            print(f"  {marker} #{rank}: score={score:.4f}  L{auth}  {doc}")

        results.append({
            "q_id": q["q_id"], "is_policy": is_policy, "request_type": request_type,
            "target_docs": list(target_docs), "hit": hit, "citizen_fraction": cit_frac,
            "tier_used": result["tier_used"],
            "retrieved": [{"rank": r+1, "document_id": d, "authority_level": a, "score": s}
                           for r, (d, a, s) in enumerate(zip(retrieved_docs, retrieved_authorities, retrieved_scores))],
        })

    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Document-level hit rate: {hits}/{len(questions)} ({hits/len(questions)*100:.1f}%)")

    policy_results = [r for r in results if r["is_policy"]]
    policy_hits = sum(1 for r in policy_results if r["hit"])
    print(f"Policy-question hit rate: {policy_hits}/{len(policy_results)} "
          f"({policy_hits/len(policy_results)*100:.1f}%)")
    print(f"Policy-question mean citizen_fraction: "
          f"{np.mean([r['citizen_fraction'] for r in policy_results]):.4f}")

    misses = [r["q_id"] for r in results if not r["hit"]]
    print(f"\nMissed questions ({len(misses)}): {misses}")

    with open("full_diagnostic_48_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: full_diagnostic_48_results.json")


if __name__ == "__main__":
    main()
