"""
STEK 2035 — Inspect the 6 Unexplained Citizen Retrieval Failures
====================================================================
Q010, Q016, Q021, Q023, Q024, Q025 all have CORRECTLY-tiered (L4)
relevant chunks, yet strict citizen-only retrieval still misses them.
Rather than guess why, this script prints the ACTUAL top-5 retrieved
text for each question, alongside the ground-truth chunk's text, so
a human can directly judge which of three hypotheses is true:

  (a) Ground truth incompleteness — the top-5 contains a genuinely
      good alternative citizen answer, just not the cited idx
  (b) Embedding weakness — the top-5 is generic/off-topic, suggesting
      e5-base failed to find good citizen-register matches at all
  (c) Near-duplicate crowding — the top-5 contains several very
      similar citizen comments on the same theme, none exactly
      matching the cited chunk

Usage:
    python stek_diagnose_citizen_misses.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl  (use the FIXED version —
      corpus_v2_chunks_fixed.jsonl — so zukunftsreise_2035 is correctly
      tiered before this diagnostic runs)
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR   = Path("corpus/corpus_v2")
CHUNKS_PATH  = CORPUS_DIR / "corpus_v2_chunks.jsonl"   # point this at the FIXED file
EMB_PATH     = CORPUS_DIR / "embeddings_v2_e5base.npy"
EMBED_MODEL  = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

TOP_K = 5

# the 6 unexplained failures, identified from the strict-filter test run
TARGET_QIDS = ["Q010", "Q016", "Q021", "Q023", "Q024", "Q025"]


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append({
                "text": obj.get("text", ""),
                "document_id": obj.get("document_id", "unknown"),
                "authority_level": obj.get("authority_level", 3),
            })
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def main():
    print("=" * 70)
    print("STEK 2035 — Diagnosing the 6 Unexplained Citizen Failures")
    print("=" * 70)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    target_questions = [q for q in gt_data["questions"] if q["q_id"] in TARGET_QIDS]

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    authority = np.array([c["authority_level"] for c in chunks])
    citizen_idx = np.where(authority >= 4)[0]
    print(f"  Citizen-tier pool size: {len(citizen_idx)} chunks\n")

    for q in target_questions:
        print(f"\n{'='*70}")
        print(f"{q['q_id']}: {q['question']}")
        print(f"{'='*70}")

        gt_idx_list = [r["idx"] for r in q["relevant_chunks"]]
        print(f"\nGROUND TRUTH chunk(s) [{gt_idx_list}]:")
        for idx in gt_idx_list:
            print(f"  [{idx}] ({chunks[idx]['document_id']}): {chunks[idx]['text'][:200]}")

        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec

        citizen_sims = sims[citizen_idx]
        top5_within_citizen = citizen_idx[np.argsort(-citizen_sims)][:TOP_K]

        print(f"\nACTUAL TOP-5 within citizen-only pool (what the system found instead):")
        for rank, idx in enumerate(top5_within_citizen, 1):
            hit_marker = " ← THIS IS THE GROUND TRUTH CHUNK" if int(idx) in gt_idx_list else ""
            print(f"  #{rank} [{idx}] score={sims[idx]:.4f} ({chunks[idx]['document_id']}){hit_marker}")
            print(f"      {chunks[idx]['text'][:200]}")

        print(f"\n👉 MANUAL JUDGMENT NEEDED: does the top-5 above contain a genuinely")
        print(f"   reasonable alternative answer to the ground truth (→ hypothesis A,")
        print(f"   ground truth incompleteness), or does it look off-topic/generic")
        print(f"   (→ hypothesis B, embedding weakness), or several very similar")
        print(f"   citizen comments on the same theme (→ hypothesis C, near-duplicate")
        print(f"   crowding)?")

    print(f"\n\n{'='*70}")
    print("NEXT STEP")
    print(f"{'='*70}")
    print("Read through the printed comparisons above for all 6 questions and")
    print("classify each one into hypothesis A/B/C. If most are A, the retrieval")
    print("system may be working better than the strict recall@5 metric suggests —")
    print("the fix would be expanding ground truth (Verification Guide Parameter 3),")
    print("not changing the retrieval strategy. If most are B, this is direct")
    print("evidence to test e5-large on these specific questions. If most are C,")
    print("consider a near-duplicate deduplication pass on Online-Beteiligung")
    print("before re-testing.")


if __name__ == "__main__":
    main()
