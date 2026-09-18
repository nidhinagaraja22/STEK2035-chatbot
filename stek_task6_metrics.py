"""
STEK 2035 — Task 6: Precision@k/Recall@k/MRR/nDCG@k Across Embedding Models
================================================================================
Uses ONLY stek_task4_5_master_ground_truth.json (the 45-verified-question
file) — no separate Task 6 ground truth file needed.

Uses DOCUMENT-LEVEL relevance, not exact chunk_id: the master ground
truth's chunk_id values were verified against corpus_v3 (100-word
chunking), which has entirely different chunk boundaries than this
v2 corpus (~270-word chunking) actually in use. Document-level
matching is stable across both and was already cross-validated
earlier this session (two independent methods gave an identical
32/45 result).

Multi-document questions (e.g. one question requiring both
zukunftsreise_2035 AND online_beteiligung_2024) get PARTIAL credit
on Recall@k — the fraction of required documents actually found,
not a binary hit/miss.

nDCG@k uses BINARY relevance (1 if correct document, 0 otherwise),
since the master ground truth does not carry graded (1 vs 2)
relevance the way the older, now-unused Task 6 file did.

Usage:
    python stek_task6_metrics.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/embeddings_v2_e5large.npy
    corpus/corpus_v2/embeddings_v2_bgem3.npy
    stek_task4_5_master_ground_truth.json
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR   = Path("corpus/corpus_v2")
CHUNKS_PATH  = CORPUS_DIR / "corpus_v2_chunks_l4l5split.jsonl"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

EMBEDDING_FILES = {
    "multilingual-e5-base":  {"emb": CORPUS_DIR / "embeddings_v2_e5base.npy",
                                "model": "intfloat/multilingual-e5-base", "prefix": "query: "},
    "multilingual-e5-large": {"emb": CORPUS_DIR / "embeddings_v2_e5large.npy",
                                "model": "intfloat/multilingual-e5-large", "prefix": "query: "},
    "bge-m3":                {"emb": CORPUS_DIR / "embeddings_v2_bgem3.npy",
                                "model": "BAAI/bge-m3", "prefix": ""},
}

K_VALUES = [1, 3, 5, 10]
CANDIDATE_K = 20

OUTPUT_JSON = "task6_metrics_results.json"
OUTPUT_CSV = "task6_metrics_summary.csv"


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                chunks.append({"text": obj.get("text", ""), "document_id": obj.get("document_id", "unknown")})
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def load_questions():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data["questions"] if q.get("relevant_chunks")]


def precision_at_k(retrieved_docs, target_docs, k):
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in target_docs)
    return hits / len(top_k)


def recall_at_k(retrieved_docs, target_docs, k):
    """Partial credit: fraction of REQUIRED documents found in top-k."""
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


def ndcg_at_k(retrieved_docs, target_docs, k):
    """Binary relevance nDCG — master ground truth has no graded relevance.
    Each target document counts ONLY ONCE toward DCG, at its best (first)
    occurrence — since multiple chunks can come from the same document,
    without this dedup a repeated document would be double-counted and
    nDCG could exceed 1.0, which is mathematically impossible."""
    counted = set()
    dcg = 0.0
    for rank, d in enumerate(retrieved_docs[:k], start=1):
        if d in target_docs and d not in counted:
            dcg += 1.0 / math.log2(rank + 1)
            counted.add(d)
    ideal_hits = min(len(target_docs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def run_model(model_key, config, chunks, questions):
    emb_path = config["emb"]
    if not emb_path.exists():
        print(f"  ⚠️  SKIPPED — embeddings file not found: {emb_path}")
        return None

    chunk_embeddings = load_normalised(emb_path)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"  ⚠️  SKIPPED — row mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks")
        return None

    print(f"  Loading model: {config['model']}")
    model = SentenceTransformer(config["model"])
    doc_ids = [c["document_id"] for c in chunks]

    metrics_by_k = defaultdict(list)
    mrr_scores = []

    for q in questions:
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}

        q_vec = model.encode(f"{config['prefix']}{q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        top_idx = np.argsort(-sims)[:CANDIDATE_K]
        retrieved_docs = [doc_ids[i] for i in top_idx]

        mrr_scores.append(reciprocal_rank(retrieved_docs, target_docs))

        for k in K_VALUES:
            metrics_by_k[k].append({
                "precision": precision_at_k(retrieved_docs, target_docs, k),
                "recall": recall_at_k(retrieved_docs, target_docs, k),
                "ndcg": ndcg_at_k(retrieved_docs, target_docs, k),
            })

    del model
    import gc
    gc.collect()

    summary = {"mrr": round(float(np.mean(mrr_scores)), 4)}
    for k in K_VALUES:
        rows = metrics_by_k[k]
        summary[f"precision@{k}"] = round(float(np.mean([r["precision"] for r in rows])), 4)
        summary[f"recall@{k}"] = round(float(np.mean([r["recall"] for r in rows])), 4)
        summary[f"ndcg@{k}"] = round(float(np.mean([r["ndcg"] for r in rows])), 4)

    return summary


def main():
    print("=" * 70)
    print("STEK 2035 — Task 6: Metrics Across Embedding Models")
    print("=" * 70)

    print(f"\nLoading chunks from: {CHUNKS_PATH}")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks")

    questions = load_questions()
    print(f"Loaded {len(questions)} verified questions from {GROUND_TRUTH_PATH}")

    all_results = {}
    for model_key, config in EMBEDDING_FILES.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_key}")
        print(f"{'='*70}")
        result = run_model(model_key, config, chunks, questions)
        if result:
            all_results[model_key] = result
            print(f"\n  MRR = {result['mrr']}")
            for k in K_VALUES:
                print(f"  k={k:<3} P={result[f'precision@{k}']:.4f}  "
                      f"R={result[f'recall@{k}']:.4f}  nDCG={result[f'ndcg@{k}']:.4f}")

    if not all_results:
        print("\n⚠️  No models produced results — check embedding file paths.")
        return

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {OUTPUT_JSON}")

    import csv
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["model", "mrr"] + [f"{m}@{k}" for k in K_VALUES for m in ["precision", "recall", "ndcg"]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_key, result in all_results.items():
            writer.writerow({"model": model_key, **result})
    print(f"✅ Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()