"""
STEK 2035 — Fit Weighted Marginal Strategy on REAL Data
============================================================
Wires the already-built and synthetic-data-verified marginal scoring
logic (stek_weighted_marginal_strategy.py) to your ACTUAL corpus,
embeddings, real LDA topics (lda_topics_v2.json), and real ground
truth — replacing the synthetic flooding-scenario proof-of-concept
with genuinely fitted weights.

Reuses marginal_selection(), evaluate_weights_on_questions(), and
fit_weights_grid_search_loo_cv() UNCHANGED from the already-verified
file — only this script's data-loading and candidate-building logic
is new.

For tractability, each question's candidate pool is limited to its
top-30 chunks by raw similarity (CANDIDATE_K) rather than all 1258 —
consistent with the CANDIDATE_K=20 convention already used elsewhere
in this project (e.g. Task 6's metrics script).

Usage:
    python stek_fit_marginal_weights_real.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/lda_topics_v2.json
    stek_task4_5_master_ground_truth.json
    stek_weighted_marginal_strategy.py (imported, not duplicated)
"""

import json
import itertools
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stek_weighted_marginal_strategy import fit_weights_grid_search_loo_cv, evaluate_weights_on_questions

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
LDA_TOPICS_PATH = Path("corpus/corpus_v2/lda_topics_v2.json")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

CANDIDATE_K = 30  # each question's candidate pool size, for tractability


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


def build_real_questions_data(gt_data, chunks, chunk_embeddings, model):
    """Build the questions_data format fit_weights_grid_search_loo_cv expects:
    [{"candidates": [...], "target_docs": set(...)}], using REAL similarity,
    REAL source documents, REAL LDA topics — only answerable questions,
    since weight-fitting needs a real correctness label to optimize against."""
    questions_data = []
    q_ids = []

    for q in gt_data["questions"]:
        if not q.get("relevant_chunks"):
            continue  # only fit against real, verified answerable questions

        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        top_idx = np.argsort(-sims)[:CANDIDATE_K]

        candidates = [
            {"similarity": float(sims[i]), "source": chunks[i]["document_id"], "topic": chunks[i]["_topic"]}
            for i in top_idx
        ]
        target_docs = {t["document_id"] for t in q["relevant_chunks"]}

        questions_data.append({"candidates": candidates, "target_docs": target_docs})
        q_ids.append(q["q_id"])

    return questions_data, q_ids


def main():
    print("=" * 70)
    print("STEK 2035 — Fit Weighted Marginal Strategy on REAL Data")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus + LDA topics from: {CHUNKS_PATH}")
    chunks = load_chunks_with_topics()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return
    print(f"  {len(chunks)} chunks, topics aligned correctly")

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"\nBuilding real candidate pools (top-{CANDIDATE_K} per question)...")
    questions_data, q_ids = build_real_questions_data(gt_data, chunks, chunk_embeddings, model)
    print(f"  {len(questions_data)} answerable questions with real candidates built")

    # baseline: pure relevance, no dominance/diversity/topic weighting —
    # same sanity check pattern as the synthetic test
    baseline_weights = {"relevance": 1.0, "dominance": 0.0, "diversity": 0.0, "topic": 0.0}
    baseline_recall = evaluate_weights_on_questions(baseline_weights, questions_data)
    print(f"\nBaseline (pure relevance, no other weights): recall={baseline_recall:.4f}")

    print(f"\nRunning leave-one-out grid search across {len(questions_data)} questions...")
    print(f"(this may take a while — {len(questions_data)} folds, each searching a weight grid)")

    avg_weights, folds = fit_weights_grid_search_loo_cv(questions_data)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nAverage fitted weights across {len(folds)} real folds:")
    for k, v in avg_weights.items():
        print(f"  {k:<12}: {v:.3f}")

    print(f"\nPer-fold best weights (checking real-data stability):")
    for f, qid in zip(folds, q_ids):
        print(f"  {qid} (held out): {f['weights']} -> test_score={f['test_score']:.2f}")

    # stability diagnostic — same principle as the earlier α-fitting failure:
    # check if weights swing wildly across folds, which would indicate
    # overfitting to a small/skewed sample rather than a genuine pattern
    import statistics
    print(f"\nWeight stability (stdev across folds — HIGH stdev = unstable/overfit):")
    for key in ["relevance", "dominance", "diversity", "topic"]:
        values = [f["weights"][key] for f in folds]
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {key:<12}: mean={statistics.mean(values):.3f}  stdev={stdev:.3f}")

    with open("marginal_weights_real_fit_results.json", "w", encoding="utf-8") as f:
        json.dump({"baseline_recall": baseline_recall, "avg_weights": avg_weights,
                    "per_fold": [{"q_id": qid, **f} for f, qid in zip(folds, q_ids)]},
                   f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: marginal_weights_real_fit_results.json")

    print(f"\n⚠️  IMPORTANT: if any weight shows HIGH stdev across folds, or if")
    print(f"   this recall doesn't clearly beat the baseline, treat the fitted")
    print(f"   weights as PROVISIONAL — this is the exact same caution that")
    print(f"   caught the earlier authority/similarity fitting bias.")


if __name__ == "__main__":
    main()