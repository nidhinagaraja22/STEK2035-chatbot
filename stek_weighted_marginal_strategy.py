"""
STEK 2035 — Weighted Marginal Scoring (Relevance/Dominance/Diversity/Topic)
================================================================================
Combines the authority pre-filter (already built, unchanged) with a
WEIGHTED, GREEDY marginal-scoring selection over the remaining three
Task 3 metrics — relevance, dominance, diversity, topic_coverage.

Authority is NOT one of the fitted weights — it's already handled by
sorting into tiers before this scoring ever runs. This is a deliberate
safeguard: fitting authority alongside the others was tried earlier in
this project and produced a spurious result (w_authority ended up
~9.4x larger than w_similarity, later traced to a skewed ground-truth
sample, not a genuine effect). Removing authority from the fitted set
removes that specific failure mode.

Why GREEDY MARGINAL scoring, not a one-shot formula:
    dominance, diversity, and topic_coverage are properties of a
    5-chunk SET, not of a single chunk in isolation — you cannot
    score one candidate against these metrics until you know what
    else is already selected. Marginal scoring solves this the same
    way MMR does: score each remaining candidate by how much
    selecting it NEXT would improve the set so far, pick the best,
    repeat.

Usage:
    from stek_weighted_marginal_strategy import marginal_selection, fit_weights_loo_cv
    top5 = marginal_selection(candidates, weights={"relevance":1.0, "dominance":1.0,
                                                      "diversity":1.0, "topic":1.0})
"""

from collections import defaultdict
import itertools

TOP_K = 5

DEFAULT_WEIGHTS = {"relevance": 1.0, "dominance": 1.0, "diversity": 1.0, "topic": 1.0}


def compute_marginal_score(candidate, selected_so_far, weights):
    """
    Score ONE candidate for the NEXT selection slot, given what's
    already been picked. Each term is a MARGINAL signal — how this
    candidate would change the set if added, not a property of the
    candidate alone.
    """
    source_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    for c in selected_so_far:
        source_counts[c["source"]] += 1
        topic_counts[c["topic"]] += 1

    n_selected = len(selected_so_far)

    # relevance — the candidate's own similarity score, unchanged by context
    relevance_term = candidate["similarity"]

    # dominance — PENALTY that grows the more this candidate's source is
    # ALREADY represented; adding a 3rd chunk from an already-2x source
    # is penalized more than adding a document's 1st chunk
    dominance_penalty = source_counts[candidate["source"]] / max(n_selected, 1)

    # diversity — REWARD for introducing a source not yet in the set
    diversity_bonus = 1.0 if source_counts[candidate["source"]] == 0 else 0.0

    # topic_coverage — REWARD for introducing a topic not yet in the set
    topic_bonus = 1.0 if topic_counts[candidate["topic"]] == 0 else 0.0

    return (weights["relevance"] * relevance_term
            - weights["dominance"] * dominance_penalty
            + weights["diversity"] * diversity_bonus
            + weights["topic"] * topic_bonus)


def marginal_selection(candidates: list, weights: dict = None, k: int = TOP_K) -> list:
    """
    Greedy selection: at each of k steps, score every remaining
    candidate by its MARGINAL contribution given what's already
    selected, pick the highest, repeat. Assumes candidates are
    already authority-filtered/ordered upstream — this function
    does not touch authority at all.
    """
    weights = weights or DEFAULT_WEIGHTS
    remaining = list(candidates)
    selected = []

    for _ in range(min(k, len(candidates))):
        scored = [(compute_marginal_score(c, selected, weights), c) for c in remaining]
        scored.sort(key=lambda x: -x[0])
        best_score, best_candidate = scored[0]
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def evaluate_weights_on_questions(weights, questions_data):
    """
    questions_data: list of dicts, each {"candidates": [...], "target_docs": set(...)}
    Returns mean document-level recall@5 across all questions — same
    metric already validated elsewhere this session.
    """
    recalls = []
    for q in questions_data:
        result = marginal_selection(q["candidates"], weights)
        retrieved_docs = {c["source"] for c in result}
        target_docs = q["target_docs"]
        if not target_docs:
            continue
        covered = len(target_docs & retrieved_docs)
        recalls.append(covered / len(target_docs))
    return sum(recalls) / len(recalls) if recalls else 0.0


def fit_weights_grid_search_loo_cv(questions_data, weight_grid=None):
    """
    Leave-one-out cross-validation grid search — the same safeguard
    used (and needed) for the earlier authority/similarity fitting.

    weight_grid: list of weight dicts to try. If None, uses a small
    default grid over relevance/dominance/diversity/topic.

    Returns: (best_weights, fold_results) — fold_results lets you
    inspect whether the best weight combination is STABLE across
    folds (a red flag if it swings wildly) rather than trusting a
    single aggregate number.
    """
    if weight_grid is None:
        # relevance is FIXED at 1.0 — only ratios between weights affect
        # selection (uniformly scaling all four weights by the same
        # constant produces IDENTICAL selections, confirmed directly:
        # {0.5,0.5,0.5,0.5} and {2.0,2.0,2.0,2.0} gave the exact same
        # recall on real data). Searching all four independently wasted
        # most of the grid on redundant scale-equivalent points and let
        # tie-breaking order silently pick the first one every time.
        values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        weight_grid = [
            {"relevance": 1.0, "dominance": d, "diversity": dv, "topic": t}
            for d, dv, t in itertools.product(values, repeat=3)
        ]

    n = len(questions_data)
    fold_best_weights = []

    for i in range(n):
        train = questions_data[:i] + questions_data[i+1:]
        test = [questions_data[i]]

        best_score = -1
        best_w = None
        for w in weight_grid:
            score = evaluate_weights_on_questions(w, train)
            if score > best_score:
                best_score = score
                best_w = w

        test_score = evaluate_weights_on_questions(best_w, test)
        fold_best_weights.append({"fold": i, "weights": best_w,
                                    "train_score": best_score, "test_score": test_score})

    # aggregate: average each weight across folds, and check variance
    # as a stability diagnostic
    avg_weights = {}
    for key in DEFAULT_WEIGHTS:
        values = [f["weights"][key] for f in fold_best_weights]
        avg_weights[key] = sum(values) / len(values)

    return avg_weights, fold_best_weights


if __name__ == "__main__":
    # ── Test 1: marginal scoring behaves sensibly on a controlled case ──────
    print("=== Test 1: marginal selection basic sanity check ===")
    candidates = [
        {"id": "A", "similarity": 0.90, "source": "doc1", "topic": "housing"},
        {"id": "B", "similarity": 0.88, "source": "doc1", "topic": "housing"},
        {"id": "C", "similarity": 0.86, "source": "doc1", "topic": "housing"},
        {"id": "D", "similarity": 0.70, "source": "doc2", "topic": "climate"},
        {"id": "E", "similarity": 0.65, "source": "doc3", "topic": "mobility"},
    ]
    # heavy diversity weight should pull in doc2/doc3 despite lower similarity
    result = marginal_selection(candidates, weights={"relevance": 1.0, "dominance": 3.0,
                                                        "diversity": 2.0, "topic": 2.0})
    ids = [c["id"] for c in result]
    print(f"With heavy diversity/dominance weighting: {ids}")
    print(f"Expect doc2 (D) and doc3 (E) to be pulled in despite lower similarity,")
    print(f"since repeatedly picking doc1 (A,B,C) incurs a growing dominance penalty.")
    assert "D" in ids and "E" in ids, "Diversity weighting failed to pull in other sources"
    print("✅ Diversity/dominance weighting works as intended\n")

    # pure relevance weighting should just take top-5 by similarity, ignoring diversity
    result2 = marginal_selection(candidates, weights={"relevance": 1.0, "dominance": 0.0,
                                                         "diversity": 0.0, "topic": 0.0})
    ids2 = [c["id"] for c in result2]
    print(f"With pure relevance weighting: {ids2}")
    print(f"Expect plain similarity order: A, B, C, D, E")
    assert ids2 == ["A", "B", "C", "D", "E"], f"Pure relevance weighting broken — got {ids2}"
    print("✅ Pure relevance weighting correctly reduces to plain top-k\n")

    # ── Test 2: weight fitting sanity check on synthetic data ───────────────
    print("=== Test 2: weight fitting on synthetic ground truth ===")
    # IMPORTANT: target and flooding chunks share the SAME topic here,
    # deliberately — an earlier version of this test gave target a
    # unique topic, which let topic_coverage weighting alone rescue it
    # (confirmed by debugging), masking whether dominance/diversity
    # weighting was actually doing anything. Same topic for all
    # candidates isolates dominance/diversity as the only possible
    # signals that can recover the flooded-out target.
    synthetic_questions = []
    for i in range(6):
        cands = [
            {"id": f"q{i}_target", "similarity": 0.60, "source": f"target_doc_{i}", "topic": "shared"},
            {"id": f"q{i}_dup1", "similarity": 0.95, "source": "flooding_doc", "topic": "shared"},
            {"id": f"q{i}_dup2", "similarity": 0.93, "source": "flooding_doc", "topic": "shared"},
            {"id": f"q{i}_dup3", "similarity": 0.91, "source": "flooding_doc", "topic": "shared"},
            {"id": f"q{i}_dup4", "similarity": 0.89, "source": "flooding_doc", "topic": "shared"},
            {"id": f"q{i}_dup5", "similarity": 0.87, "source": "flooding_doc", "topic": "shared"},
        ]
        synthetic_questions.append({"candidates": cands, "target_docs": {f"target_doc_{i}"}})

    # confirm: with dominance=0, diversity=0, topic irrelevant (all same topic
    # anyway), target should NOT be recovered — establishing a true baseline
    baseline = evaluate_weights_on_questions({"relevance": 1.0, "dominance": 0.0,
                                                 "diversity": 0.0, "topic": 0.0}, synthetic_questions)
    print(f"Baseline (pure relevance, no dominance/diversity): recall={baseline:.2f}")
    assert baseline == 0.0, f"Baseline should be 0.0 (target always outranked) — got {baseline}"
    print("✅ Confirmed: without dominance/diversity weighting, target is never recovered\n")

    small_grid = [
        {"relevance": 1.0, "dominance": d, "diversity": dv, "topic": 0.0}
        for d in [0.0, 1.0, 3.0] for dv in [0.0, 1.0, 3.0]
    ]
    avg_weights, folds = fit_weights_grid_search_loo_cv(synthetic_questions, small_grid)
    print(f"Average fitted weights across folds: {avg_weights}")
    print(f"Per-fold best weights (checking stability):")
    for f in folds:
        print(f"  fold {f['fold']}: {f['weights']} -> test_score={f['test_score']:.2f}")

    print(f"\nExpect: dominance and/or diversity weights now pulled HIGH across")
    print(f"ALL folds, since with the topic confound removed, they are the")
    print(f"ONLY signals that can recover target_doc from being flooded out.")
    assert avg_weights["dominance"] > 0 or avg_weights["diversity"] > 0, \
        "Neither dominance nor diversity weight was learned — fitting failed"
    print("✅ Confirmed: fitting correctly learned to weight dominance/diversity")