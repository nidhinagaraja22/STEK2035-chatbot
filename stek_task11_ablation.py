"""
STEK 2035 — Task 11: Ablation Study
========================================
Systematic build-up, reusing REAL, already-validated components from
this project rather than re-deriving anything:

  Stage 0 (baseline):      plain top-k, no authority, no dedup
                             (pre-cleaning corpus)
  Stage 1 (+dedup):        plain top-k, CURRENT cleaned corpus
  Stage 2 (+authority):    threshold-gated cascade (official→L4→L5)
  Stage 3 (+diversity):    authority_source_topic — cascade's
                             authority logic PLUS source/topic capping

"+metadata" from the originally-requested sequence is not separately
testable — authority_level tags have no effect on retrieval until a
strategy actually uses them; that effect first appears at Stage 2.

Stage 4 (+expanded corpus) is reported qualitatively, not as a
same-corpus comparison: Q051-Q054 were added to the ground truth
AFTER the original 45-question set, so there is no "before" version
of the corpus to re-test against — this stage's evidence is that
these 4 questions test genuinely newer content (MRO 2035+, Vorrangflur
etc.) not covered by the original evaluation set at all.

Usage:
    python stek_task11_ablation.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl (current, cleaned)
    A pre-cleaning reference corpus for Stage 0 (adjust PRE_CLEANING_PATH)
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/lda_topics_v2.json
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py, stek_strategy_authority_source_topic.py (imported)
"""

import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve
from stek_strategy_authority_source_topic import authority_source_topic_strategy

CURRENT_CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
# adjust this to wherever your pre-cleaning reference corpus lives —
# the corpus BEFORE boilerplate/URL-fragment dedup was applied
PRE_CLEANING_CHUNKS_PATH = Path("reference_old_1266.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
LDA_TOPICS_PATH = Path("corpus/corpus_v2/lda_topics_v2.json")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

K_VALUES = [1, 3, 5, 10]


def load_chunks(path, with_topics=False):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if with_topics:
        with open(LDA_TOPICS_PATH, encoding="utf-8") as f:
            topic_data = json.load(f)
        assignments = topic_data["chunk_topic_assignment"]
        if len(assignments) == len(chunks):
            for c, a in zip(chunks, assignments):
                c["_topic"] = a["lda_topic"]
        else:
            print(f"  ⚠️  Topic assignment count mismatch for {path} — topics unavailable for this corpus version")
            for c in chunks:
                c["_topic"] = 0
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def precision_at_k(retrieved_docs, target_docs, k):
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in target_docs) / len(top_k)


def recall_at_k(retrieved_docs, target_docs, k):
    if not target_docs:
        return 0.0
    return len(target_docs & set(retrieved_docs[:k])) / len(target_docs)


def reciprocal_rank(retrieved_docs, target_docs):
    for rank, d in enumerate(retrieved_docs, start=1):
        if d in target_docs:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_docs, target_docs, k):
    counted = set()
    dcg = 0.0
    for rank, d in enumerate(retrieved_docs[:k], start=1):
        if d in target_docs and d not in counted:
            dcg += 1.0 / math.log2(rank + 1)
            counted.add(d)
    ideal_hits = min(len(target_docs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(name, get_retrieved_fn, questions):
    all_p, all_r, all_n, all_rr = {k: [] for k in K_VALUES}, {k: [] for k in K_VALUES}, {k: [] for k in K_VALUES}, []
    for q in questions:
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}
        retrieved = get_retrieved_fn(q)
        all_rr.append(reciprocal_rank(retrieved, target_docs))
        for k in K_VALUES:
            all_p[k].append(precision_at_k(retrieved, target_docs, k))
            all_r[k].append(recall_at_k(retrieved, target_docs, k))
            all_n[k].append(ndcg_at_k(retrieved, target_docs, k))

    mrr = float(np.mean(all_rr))
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"MRR: {mrr:.4f}")
    for k in K_VALUES:
        print(f"  k={k:<3} P={np.mean(all_p[k]):.4f}  R={np.mean(all_r[k]):.4f}  nDCG={np.mean(all_n[k]):.4f}")

    return {"mrr": mrr, "metrics_by_k": {k: {"precision": float(np.mean(all_p[k])),
                                                "recall": float(np.mean(all_r[k])),
                                                "ndcg": float(np.mean(all_n[k]))} for k in K_VALUES}}


def main():
    print("=" * 70)
    print("Task 11 — Ablation Study")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    original_questions = [q for q in gt_data["questions"]
                            if q.get("relevant_chunks") and q["q_id"] not in ("Q051", "Q052", "Q053", "Q054")]
    expanded_only = [q for q in gt_data["questions"] if q["q_id"] in ("Q051", "Q052", "Q053", "Q054")]
    print(f"\nOriginal question set: {len(original_questions)} | Expansion-only: {len(expanded_only)}")

    results = {}

    # ── Stage 0: baseline — plain top-k on PRE-CLEANING corpus ────────────────
    if PRE_CLEANING_CHUNKS_PATH.exists():
        print(f"\nLoading pre-cleaning corpus from: {PRE_CLEANING_CHUNKS_PATH}")
        old_chunks = load_chunks(PRE_CLEANING_CHUNKS_PATH)
        print(f"  ⚠️  {len(old_chunks)} chunks — SKIPPING Stage 0 quantitatively: this corpus")
        print(f"     needs its own separately-generated embeddings (different chunk count/order")
        print(f"     than the current corpus), which were not generated for this ablation.")
        print(f"     Documenting this as a known gap rather than approximating it.")
    else:
        print(f"\n⚠️  {PRE_CLEANING_CHUNKS_PATH} not found — Stage 0 (dedup ablation) skipped.")
        print(f"   Adjust PRE_CLEANING_CHUNKS_PATH to your actual pre-cleaning corpus file.")

    # ── Stage 1: current cleaned corpus, plain top-k ──────────────────────────
    print(f"\nLoading current (cleaned, deduped) corpus from: {CURRENT_CHUNKS_PATH}")
    chunks = load_chunks(CURRENT_CHUNKS_PATH, with_topics=True)
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return
    doc_ids = [c["document_id"] for c in chunks]
    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]

    print("Precomputing similarities for all original questions...")
    q_sims = {}
    for q in original_questions:
        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        q_sims[q["q_id"]] = chunk_embeddings @ q_vec

    def plain_topk_fn(q):
        sims = q_sims[q["q_id"]]
        return [doc_ids[i] for i in np.argsort(-sims)[:10]]

    results["stage1_dedup_plain_topk"] = evaluate(
        "STAGE 1 (+dedup) — Plain top-k, current cleaned corpus", plain_topk_fn, original_questions)

    # ── Stage 2: +authority (threshold cascade) ───────────────────────────────
    def cascade_fn(q):
        result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model, official_idx, l4_idx, l5_idx)
        return [doc_ids[i] for i in result["result_idx"]]

    results["stage2_plus_authority"] = evaluate(
        "STAGE 2 (+authority) — Threshold-gated cascade", cascade_fn, original_questions)

    # ── Stage 3: +diversity (authority_source_topic) ──────────────────────────
    def diversity_fn(q):
        sims = q_sims[q["q_id"]]
        candidates = [{"similarity": float(sims[i]), "source": chunks[i]["document_id"],
                        "authority_level": chunks[i].get("authority_level", 3), "topic": chunks[i]["_topic"]}
                       for i in np.argsort(-sims)]
        selected = authority_source_topic_strategy(candidates, k=10)
        return [c["source"] for c in selected]

    results["stage3_plus_diversity"] = evaluate(
        "STAGE 3 (+diversity) — Authority + source/topic capping", diversity_fn, original_questions)

    # ── Stage 4: +expanded corpus — qualitative, using the BEST strategy so far ──
    print(f"\n{'='*70}")
    print("STAGE 4 (+expanded corpus) — qualitative, not a same-corpus comparison")
    print(f"{'='*70}")
    print(f"Testing the {len(expanded_only)} expansion-only questions (Q051-Q054) with")
    print(f"Stage 1's plain top-k, to see how NEW content performs (no 'before expansion'")
    print(f"baseline exists, since these questions never existed pre-expansion):")
    for q in expanded_only:
        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        retrieved = [doc_ids[i] for i in np.argsort(-sims)[:5]]
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}
        hit = bool(target_docs & set(retrieved))
        print(f"  {q['q_id']}: {'✅ HIT' if hit else '❌ MISS'}  target={target_docs}")

    print(f"\n{'='*70}")
    print("ABLATION SUMMARY — Same 41 Original Questions Throughout")
    print(f"{'='*70}")
    for stage, r in results.items():
        print(f"  {stage:<30}: MRR={r['mrr']:.4f}  R@5={r['metrics_by_k'][5]['recall']:.4f}")

    with open("task11_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: task11_ablation_results.json")

    print(f"\n⚠️  Stage 0 (pure dedup effect, isolated) could not be measured quantitatively")
    print(f"   without separately embedding the pre-cleaning corpus — documented as a")
    print(f"   known gap, not silently skipped.")


if __name__ == "__main__":
    main()
