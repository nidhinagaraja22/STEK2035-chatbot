"""
STEK 2035 — Diagnose All Remaining Unverified Failures
==========================================================
Following the same pattern as stek_diagnose_citizen_misses.py, which
found 5 of 6 "failures" were actually ground truth incompleteness, not
real retrieval failures. This script runs the identical diagnostic
against the 19 REMAINING unverified "ALL MISS" questions from the last
full retrieval run — mostly policy-classified questions, so this
searches the FULL corpus (not citizen-tier-restricted).

Usage:
    python stek_diagnose_remaining_misses.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR   = Path("corpus/corpus_v2")
CHUNKS_PATH  = CORPUS_DIR / "corpus_v2_chunks.jsonl"
EMB_PATH     = CORPUS_DIR / "embeddings_v2_e5base.npy"
EMBED_MODEL  = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

TOP_K = 5

# the 19 remaining unverified ALL MISS questions (Q021 excluded — already
# confirmed as a genuine failure via boilerplate contamination)
TARGET_QIDS = [
    "Q001", "Q006", "Q007", "Q008", "Q009", "Q011", "Q012", "Q013", "Q015",
    "Q017", "Q019", "Q022", "Q026", "Q029", "Q037", "Q041", "Q043", "Q044", "Q045",
]


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
    print("STEK 2035 — Diagnosing 19 Remaining Unverified Failures")
    print("=" * 70)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    target_questions = [q for q in gt_data["questions"] if q["q_id"] in TARGET_QIDS]
    print(f"\nFound {len(target_questions)} of {len(TARGET_QIDS)} target questions")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return
    print(f"  {len(chunks)} chunks (full corpus, no tier restriction)\n")

    # classification tally for the summary at the end
    summary = []

    for q in target_questions:
        print(f"\n{'='*70}")
        print(f"{q['q_id']} [{q['category']}]: {q['question']}")
        print(f"{'='*70}")

        gt_idx_list = [r["idx"] for r in q["relevant_chunks"]]
        print(f"\nGROUND TRUTH chunk(s) {gt_idx_list}:")
        for idx in gt_idx_list:
            print(f"  [{idx}] ({chunks[idx]['document_id']}): {chunks[idx]['text'][:180]}")

        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec

        top5 = np.argsort(-sims)[:TOP_K]

        print(f"\nACTUAL TOP-5 (full corpus, what the system found instead):")
        for rank, idx in enumerate(top5, 1):
            idx = int(idx)
            hit_marker = " ← THIS IS THE GROUND TRUTH CHUNK" if idx in gt_idx_list else ""
            print(f"  #{rank} [{idx}] score={sims[idx]:.4f} ({chunks[idx]['document_id']}, "
                  f"L{chunks[idx]['authority_level']}){hit_marker}")
            print(f"      {chunks[idx]['text'][:180]}")

        print(f"\n👉 MANUAL JUDGMENT: A (ground truth incomplete — good alternative found),")
        print(f"   B (embedding weakness — top-5 looks off-topic/generic), or")
        print(f"   C (boilerplate/header contamination — top-5 is titles/agendas)?")

        summary.append(q["q_id"])

    print(f"\n\n{'='*70}")
    print(f"DIAGNOSED {len(summary)} QUESTIONS — READ THROUGH ALL RESULTS ABOVE")
    print(f"{'='*70}")
    print("For each question, classify into hypothesis A/B/C based on the")
    print("printed top-5 content. This will reveal the TRUE recall rate,")
    print("as distinct from the current 41.4% lower-bound estimate.")


if __name__ == "__main__":
    main()
