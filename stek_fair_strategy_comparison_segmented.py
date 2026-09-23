"""
STEK 2035 — Fair, Same-Question-Set Comparison of Both Strategies
=======================================================================
Runs BOTH candidate strategies — the threshold-gated cascade and
authority_source_topic — on the EXACT SAME set of answerable
questions, with the EXACT SAME metric formulas, in ONE script.

This exists specifically because earlier comparisons used DIFFERENT
question counts (45 vs 48 — Q051-Q054 were added to the ground truth
between the two separate runs), making the earlier "cascade wins"
conclusion potentially confounded by which questions were tested,
not just which strategy was used. This script removes that
confound entirely by evaluating both on the identical set, in the
same run.

Usage:
    python stek_fair_strategy_comparison.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/lda_topics_v2.json
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py (imported)
    stek_strategy_authority_source_topic.py (imported)
"""

import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve
from stek_strategy_authority_source_topic import authority_source_topic_strategy
from stek_test_vague_unanswerable_signals import strip_generic_terms

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
LDA_TOPICS_PATH = Path("corpus/corpus_v2/lda_topics_v2.json")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

K_VALUES = [1, 3, 5, 10]


def load_chunks_with_topics():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    with open(LDA_TOPICS_PATH, encoding="utf-8") as f:
        topic_data = json.load(f)
    assignments = topic_data["chunk_topic_assignment"]
    if len(assignments) != len(chunks):
        raise SystemExit(f"❌ Topic assignment count ({len(assignments)}) != chunk count ({len(chunks)})")
    for c, a in zip(chunks, assignments):
        c["_topic"] = a["lda_topic"]
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
    hits = sum(1 for d in top_k if d in target_docs)
    return hits / len(top_k)


def recall_at_k(retrieved_docs, target_docs, k):
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
    counted = set()
    dcg = 0.0
    for rank, d in enumerate(retrieved_docs[:k], start=1):
        if d in target_docs and d not in counted:
            dcg += 1.0 / math.log2(rank + 1)
            counted.add(d)
    ideal_hits = min(len(target_docs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_strategy(name, get_retrieved_docs_fn, questions):
    """get_retrieved_docs_fn(q, sims, chunks) -> list of document_ids, ranked"""
    all_precision = {k: [] for k in K_VALUES}
    all_recall = {k: [] for k in K_VALUES}
    all_ndcg = {k: [] for k in K_VALUES}
    all_rr = []
    hits = 0

    for q, sims in questions:
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}
        retrieved_docs = get_retrieved_docs_fn(q, sims)

        rr = reciprocal_rank(retrieved_docs, target_docs)
        all_rr.append(rr)
        if recall_at_k(retrieved_docs, target_docs, 5) >= 0.99:
            hits += 1

        for k in K_VALUES:
            all_precision[k].append(precision_at_k(retrieved_docs, target_docs, k))
            all_recall[k].append(recall_at_k(retrieved_docs, target_docs, k))
            all_ndcg[k].append(ndcg_at_k(retrieved_docs, target_docs, k))

    print(f"\n{'='*70}")
    print(f"STRATEGY: {name}")
    print(f"{'='*70}")
    print(f"MRR: {np.mean(all_rr):.4f}   Full R@5 hits: {hits}/{len(questions)}")
    for k in K_VALUES:
        print(f"  k={k:<3} P={np.mean(all_precision[k]):.4f}  "
              f"R={np.mean(all_recall[k]):.4f}  nDCG={np.mean(all_ndcg[k]):.4f}")

    return {"mrr": float(np.mean(all_rr)), "full_hits": hits, "n_questions": len(questions),
            "metrics_by_k": {k: {"precision": float(np.mean(all_precision[k])),
                                   "recall": float(np.mean(all_recall[k])),
                                   "ndcg": float(np.mean(all_ndcg[k]))} for k in K_VALUES}}


def main():
    print("=" * 70)
    print("FAIR COMPARISON — Same Question Set, Same Metrics, Both Strategies")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus + LDA topics from: {CHUNKS_PATH}")
    chunks = load_chunks_with_topics()
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
    all_answerable = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    policy_questions = [q for q in all_answerable if q.get("is_official_policy") is True]
    citizen_questions = [q for q in all_answerable if q.get("is_official_policy") is not True]
    print(f"\n✅ Segmented: {len(policy_questions)} POLICY questions, "
          f"{len(citizen_questions)} CITIZEN-OPINION questions "
          f"({len(all_answerable)} total, none removed)")

    def run_segment(segment_name, answerable_questions):
        print(f"\n\n{'#'*70}")
        print(f"# SEGMENT: {segment_name} ({len(answerable_questions)} questions)")
        print(f"{'#'*70}")

        print("Precomputing similarities — both RAW and STRIPPED questions...")
        questions_with_sims = []
        questions_with_sims_stripped = []
        for q in answerable_questions:
            q_vec = model.encode(f"query: {q['question']}")
            q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
            sims = chunk_embeddings @ q_vec
            questions_with_sims.append((q, sims))

            stripped = strip_generic_terms(q["question"])
            if not stripped:
                stripped = q["question"]
            q_vec_s = model.encode(f"query: {stripped}")
            q_vec_s = q_vec_s / max(np.linalg.norm(q_vec_s), 1e-8)
            sims_s = chunk_embeddings @ q_vec_s
            questions_with_sims_stripped.append((q, sims_s))

        def cascade_fn(q, sims):
            result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model,
                                          official_idx, l4_idx, l5_idx)
            return [doc_ids[i] for i in result["result_idx"]]

        cascade_results = evaluate_strategy(f"[{segment_name}] Threshold-Gated Cascade", cascade_fn, questions_with_sims)

        def plain_topk_fn(q, sims):
            top_idx = np.argsort(-sims)[:10]
            return [doc_ids[i] for i in top_idx]

        plain_results = evaluate_strategy(f"[{segment_name}] Plain Top-K (UNSTRIPPED)", plain_topk_fn, questions_with_sims)
        plain_results_stripped = evaluate_strategy(f"[{segment_name}] Plain Top-K (STRIPPED)", plain_topk_fn, questions_with_sims_stripped)

        def authority_fn(q, sims):
            candidates = [
                {"similarity": float(sims[i]), "source": chunks[i]["document_id"],
                 "authority_level": chunks[i].get("authority_level", 3), "topic": chunks[i]["_topic"]}
                for i in np.argsort(-sims)
            ]
            selected = authority_source_topic_strategy(candidates, k=10)
            return [c["source"] for c in selected]

        authority_results = evaluate_strategy(f"[{segment_name}] Authority Source Topic", authority_fn, questions_with_sims)

        print(f"\n\n{'='*70}")
        print(f"HEAD-TO-HEAD — {segment_name} ({len(answerable_questions)} QUESTIONS)")
        print(f"{'='*70}")
        print(f"{'Metric':<15} {'Plain TopK':<12} {'Cascade':<12} {'Authority':<12} {'Winner'}")
        print("-" * 68)

        def winner(vals_dict):
            return max(vals_dict, key=vals_dict.get)

        mrr_vals = {"Plain": plain_results["mrr"], "Cascade": cascade_results["mrr"], "Authority": authority_results["mrr"]}
        print(f"{'MRR':<15} {plain_results['mrr']:<12.4f} {cascade_results['mrr']:<12.4f} "
              f"{authority_results['mrr']:<12.4f} {winner(mrr_vals)}")
        for k in K_VALUES:
            for metric in ["precision", "recall", "ndcg"]:
                p = plain_results["metrics_by_k"][k][metric]
                c = cascade_results["metrics_by_k"][k][metric]
                a = authority_results["metrics_by_k"][k][metric]
                vals = {"Plain": p, "Cascade": c, "Authority": a}
                print(f"{metric}@{k:<10} {p:<12.4f} {c:<12.4f} {a:<12.4f} {winner(vals)}")

        segment_filename = f"fair_strategy_comparison_{segment_name.lower().replace(' ', '_').replace('-', '_')}.json"
        with open(segment_filename, "w", encoding="utf-8") as f:
            json.dump({"segment": segment_name, "n_questions": len(answerable_questions),
                        "plain_topk": plain_results, "plain_topk_stripped": plain_results_stripped,
                        "cascade": cascade_results, "authority_source_topic": authority_results},
                       f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved: {segment_filename}")

        return {"mrr": {"Plain": plain_results["mrr"], "Cascade": cascade_results["mrr"], "Authority": authority_results["mrr"]},
                "r5": {"Plain": plain_results["metrics_by_k"][5]["recall"],
                       "Cascade": cascade_results["metrics_by_k"][5]["recall"],
                       "Authority": authority_results["metrics_by_k"][5]["recall"]}}

    policy_summary = run_segment("POLICY", policy_questions)
    citizen_summary = run_segment("CITIZEN-OPINION", citizen_questions)

    print(f"\n\n{'='*70}")
    print("FINAL SIDE-BY-SIDE — POLICY vs CITIZEN-OPINION SEGMENTS")
    print(f"{'='*70}")
    print(f"{'Strategy':<12} {'Policy MRR':<12} {'Policy R@5':<12} {'Citizen MRR':<13} {'Citizen R@5'}")
    for strat in ["Plain", "Cascade", "Authority"]:
        print(f"{strat:<12} {policy_summary['mrr'][strat]:<12.4f} {policy_summary['r5'][strat]:<12.4f} "
              f"{citizen_summary['mrr'][strat]:<13.4f} {citizen_summary['r5'][strat]:<12.4f}")

    print(f"\n👉 Both segments are reported in full — no questions were removed from")
    print(f"   evaluation. This shows how each strategy performs on your PRIMARY")
    print(f"   use case (policy) and SECONDARY use case (citizen opinion) separately,")
    print(f"   rather than blending them into one number that represents neither well.")


if __name__ == "__main__":
    main()
