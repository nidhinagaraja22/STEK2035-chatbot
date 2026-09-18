"""
STEK 2035 — Real Accuracy of the SELECTED Strategy (authority_source_topic)
================================================================================
Measures Precision@k, Recall@k, MRR, and nDCG@k for the strategy
actually selected earlier in this project — authority-primary sort
+ Task 3's validated combined_source_topic dual capping — NOT the
older threshold-gated cascade, which is what Task 6 was previously
run against by mistake.

Reuses, unmodified:
  - authority_source_topic_strategy() from
    stek_strategy_authority_source_topic.py (already verified)
  - precision_at_k/recall_at_k/reciprocal_rank/ndcg_at_k formulas
    from stek_task6_metrics.py (already verified, including the
    nDCG double-counting bug fix)

Uses document-level matching throughout, consistent with every
other evaluation this session, given the ground truth's chunk_id
values are not reliably comparable to this corpus's own chunk
positions.

Usage:
    python stek_measure_selected_strategy_accuracy.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/lda_topics_v2.json
    stek_task4_5_master_ground_truth.json
    stek_strategy_authority_source_topic.py (imported, not duplicated)
"""

import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_strategy_authority_source_topic import authority_source_topic_strategy

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
LDA_TOPICS_PATH = Path("corpus/corpus_v2/lda_topics_v2.json")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

CANDIDATE_K = None  # REMOVED pre-filtering by pure similarity — that was a
                      # real bug: filtering to top-30 by raw similarity BEFORE
                      # authority-sorting meant official-tier chunks could be
                      # excluded from the pool entirely if they didn't score in
                      # the raw top-30, so authority gating never got a chance
                      # to promote them. Candidates are now built from the
                      # FULL corpus every time.
K_VALUES = [1, 3, 5, 10]

ABSTENTION_THRESHOLD = 0.75  # if best retrieved similarity < this, system SHOULD
                                # decline to answer rather than force a response.
                                # NOTE: earlier testing this session found no
                                # similarity-based threshold cleanly separates
                                # answerable from unanswerable questions — this
                                # metric measures EXACTLY HOW GOOD/BAD that known
                                # limitation actually is, in standard terms,
                                # rather than leaving it as a qualitative finding


def compute_abstention_metrics(all_results):
    """
    Standard binary classification metrics for the abstention decision.
    Positive class = "should abstain" (question is unanswerable/vague).

    TP: correctly abstained on a genuinely unanswerable/vague question
    TN: correctly attempted an answer on a genuinely answerable question
    FP: wrongly abstained on a genuinely answerable question (false alarm —
        the WORSE error in practice, since it denies a real answer to a
        real user)
    FN: wrongly attempted to answer a genuinely unanswerable question
        (the hallucination-risk error)
    """
    tp = sum(1 for r in all_results if r["should_abstain"] and r["did_abstain"])
    tn = sum(1 for r in all_results if not r["should_abstain"] and not r["did_abstain"])
    fp = sum(1 for r in all_results if not r["should_abstain"] and r["did_abstain"])
    fn = sum(1 for r in all_results if r["should_abstain"] and not r["did_abstain"])

    accuracy = (tp + tn) / len(all_results) if all_results else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


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


# ── Task 6 metric formulas — unchanged from the already-verified,
#    bug-fixed stek_task6_metrics.py ─────────────────────────────────────────
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
    """Each target document counts ONLY ONCE, at its best occurrence —
    the exact fix applied earlier after finding scores could exceed 1.0."""
    counted = set()
    dcg = 0.0
    for rank, d in enumerate(retrieved_docs[:k], start=1):
        if d in target_docs and d not in counted:
            dcg += 1.0 / math.log2(rank + 1)
            counted.add(d)
    ideal_hits = min(len(target_docs), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    print("=" * 70)
    print("Real Accuracy of the SELECTED Strategy (authority_source_topic)")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus + LDA topics from: {CHUNKS_PATH}")
    chunks = load_chunks_with_topics()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return
    print(f"  {len(chunks)} chunks loaded")

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    answerable_questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    abstention_questions = [q for q in gt_data["questions"]
                              if q["category"] in ("vague", "unanswerable", "semi_answerable")
                              or q.get("relevant_chunks")]
    print(f"Evaluating {len(answerable_questions)} verified answerable questions for P/R/MRR/nDCG")
    print(f"Evaluating {len(abstention_questions)} total questions for abstention accuracy\n")

    all_precision = {k: [] for k in K_VALUES}
    all_recall = {k: [] for k in K_VALUES}
    all_ndcg = {k: [] for k in K_VALUES}
    all_rr = []
    per_question = []

    for q in answerable_questions:
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}

        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        top_idx = np.argsort(-sims)  # ALL chunks, no pre-filtering — authority gating needs the full pool

        candidates = [
            {"similarity": float(sims[i]), "source": chunks[i]["document_id"],
             "authority_level": chunks[i].get("authority_level", 3), "topic": chunks[i]["_topic"]}
            for i in top_idx
        ]

        # THE ACTUAL SELECTED STRATEGY — authority-primary sort +
        # combined_source_topic dual capping, run for real here
        selected = authority_source_topic_strategy(candidates, k=max(K_VALUES))
        retrieved_docs = [c["source"] for c in selected]

        rr = reciprocal_rank(retrieved_docs, target_docs)
        all_rr.append(rr)

        for k in K_VALUES:
            p = precision_at_k(retrieved_docs, target_docs, k)
            r = recall_at_k(retrieved_docs, target_docs, k)
            n = ndcg_at_k(retrieved_docs, target_docs, k)
            all_precision[k].append(p)
            all_recall[k].append(r)
            all_ndcg[k].append(n)

        p5, r5 = all_precision[5][-1], all_recall[5][-1]
        marker = "✅" if r5 >= 0.99 else ("🟡" if r5 > 0 else "❌")
        print(f"{marker} {q['q_id']:<6} P@5={p5:.2f} R@5={r5:.2f} RR={rr:.2f}  target={target_docs}")

        per_question.append({"q_id": q["q_id"], "precision_at_5": round(p5, 3),
                                "recall_at_5": round(r5, 3), "reciprocal_rank": round(rr, 3),
                                "target_docs": list(target_docs), "retrieved_docs": retrieved_docs})

    # ── Abstention accuracy — separate pass over ALL questions,
    #    including unanswerable/vague, since it needs both to compute
    #    a real confusion matrix ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ABSTENTION ACCURACY (threshold={ABSTENTION_THRESHOLD})")
    print(f"{'='*70}")
    abstention_results = []
    for q in abstention_questions:
        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        top_idx = np.argsort(-sims)  # ALL chunks, no pre-filtering — authority gating needs the full pool
        candidates = [
            {"similarity": float(sims[i]), "source": chunks[i]["document_id"],
             "authority_level": chunks[i].get("authority_level", 3), "topic": chunks[i]["_topic"]}
            for i in top_idx
        ]
        selected = authority_source_topic_strategy(candidates, k=5)
        best_score = max(c["similarity"] for c in selected)

        should_abstain = q["category"] in ("vague", "unanswerable", "semi_answerable")
        did_abstain = best_score < ABSTENTION_THRESHOLD

        marker = "✅" if should_abstain == did_abstain else "❌"
        print(f"{marker} {q['q_id']:<6} [{q['category']:<16}] best_score={best_score:.4f}  "
              f"should_abstain={should_abstain}  did_abstain={did_abstain}")

        abstention_results.append({"q_id": q["q_id"], "category": q["category"],
                                      "best_score": best_score, "should_abstain": should_abstain,
                                      "did_abstain": did_abstain})

    abstention_metrics = compute_abstention_metrics(abstention_results)
    print(f"\nConfusion matrix: TP={abstention_metrics['tp']} TN={abstention_metrics['tn']} "
          f"FP={abstention_metrics['fp']} FN={abstention_metrics['fn']}")
    print(f"Accuracy:  {abstention_metrics['accuracy']:.4f}")
    print(f"Precision: {abstention_metrics['precision']:.4f}  (of questions flagged to abstain, "
          f"how many truly needed it)")
    print(f"Recall:    {abstention_metrics['recall']:.4f}  (of questions that truly needed abstention, "
          f"how many were caught)")
    print(f"F1:        {abstention_metrics['f1']:.4f}")
    print(f"\n⚠️  Given this session's own extensive testing found no similarity-based")
    print(f"   threshold cleanly separates answerable from unanswerable questions,")
    print(f"   expect this to show real, meaningful errors (both FP and FN) —")
    print(f"   that is the honest, already-anticipated result, not a bug.")

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS — authority_source_topic_strategy")
    print(f"{'='*70}")
    print(f"MRR: {np.mean(all_rr):.4f}\n")
    for k in K_VALUES:
        print(f"  k={k:<3} P={np.mean(all_precision[k]):.4f}  "
              f"R={np.mean(all_recall[k]):.4f}  nDCG={np.mean(all_ndcg[k]):.4f}")

    with open("selected_strategy_accuracy_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "mrr": float(np.mean(all_rr)),
            "metrics_by_k": {k: {"precision": float(np.mean(all_precision[k])),
                                   "recall": float(np.mean(all_recall[k])),
                                   "ndcg": float(np.mean(all_ndcg[k]))} for k in K_VALUES},
            "per_question": per_question,
            "abstention_metrics": abstention_metrics,
            "abstention_per_question": abstention_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: selected_strategy_accuracy_results.json")


if __name__ == "__main__":
    main()