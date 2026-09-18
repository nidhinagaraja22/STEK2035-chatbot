"""
STEK 2035 — Learn Alpha via Logistic Regression
==================================================
Earlier in this project, alpha (the similarity/authority blend weight in
authority_reranked) was left at an unvalidated default of 0.7, and a grid
sweep (stek_alpha_lambda_sweep.py) was used to test candidate values against
ground truth. This script goes one step further: instead of testing discrete
candidate alphas, it FITS a logistic regression classifier on the labeled
(chunk, is_relevant) pairs from the ground truth, then derives an
alpha-equivalent from the learned coefficients.

Method — pointwise learning-to-rank:
  For every question with verified relevant_chunks in the ground truth,
  retrieve its top-20 candidates. Each candidate becomes one training row:
      features = [similarity, authority_weight]
      label    = 1 if candidate.idx in question.relevant_chunks else 0
  Fit: P(relevant) = sigmoid(w1*similarity + w2*authority_weight + b)
  Derive: alpha_effective = w1 / (w1 + w2)   (normalised so it plugs back
          into the existing score = alpha*similarity + (1-alpha)*authority
          formula used throughout the project, unchanged)

Validation — Leave-One-Query-Out (LOQO):
  With only ~30 questions carrying verified ground truth, a single train/
  test split would be too small and noisy to trust. LOQO refits the model
  30 times, each time holding out one full question's candidates as the
  test set, and reports how well the learned weights generalise to unseen
  questions — this is the honest way to check for overfitting at this
  data scale, as flagged as a risk when this approach was first discussed.

Usage:
    python stek_fit_alpha_logistic_regression.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
    scikit-learn  (pip install scikit-learn --break-system-packages)

Outputs:
    learned_alpha_result.json   — fitted weights, effective alpha, LOQO
                                  validation scores, and an explicit
                                  recommendation on whether to trust it
"""

import json
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit(
        "scikit-learn not installed. Run:\n"
        "  pip install scikit-learn --break-system-packages"
    )

from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────
CORPUS_DIR   = Path("corpus/corpus_v2")
CHUNKS_PATH  = CORPUS_DIR / "corpus_v2_chunks.jsonl"
EMB_PATH     = CORPUS_DIR / "embeddings_v2_e5base.npy"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")
OUTPUT_PATH = Path("learned_alpha_result.json")

CANDIDATE_K = 20
AUTHORITY_WEIGHT = {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.30}

# minimum number of ground-truth questions required before trusting a
# fitted model at all — below this, the script will still run but will
# print a strong warning against using the result
MIN_QUESTIONS_FOR_TRUST = 40


def get_authority_weight(level) -> float:
    return AUTHORITY_WEIGHT.get(int(level), 0.70)


# ── loading ───────────────────────────────────────────────────────────────────

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
                "source": obj.get("document_id") or obj.get("origin") or "unknown",
                "authority_level": obj.get("authority_level", 3),
            })
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def load_ground_truth_questions():
    """Only questions with non-empty, verified relevant_chunks are usable."""
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        data = json.load(f)
    usable = [q for q in data["questions"] if q.get("relevant_chunks")]
    skipped = len(data["questions"]) - len(usable)
    return usable, skipped


# ── training data construction ────────────────────────────────────────────────

def build_training_rows(questions, model, chunk_embeddings, chunks):
    """
    Returns:
      X: (n_rows, 2) array of [similarity, authority_weight]
      y: (n_rows,) array of 0/1 labels
      groups: (n_rows,) array of question index — needed for LOQO splitting
    """
    X_rows, y_rows, group_rows = [], [], []

    for qi, q in enumerate(questions):
        relevant_idx = {r["idx"] for r in q["relevant_chunks"]}

        q_vec = model.encode(f"query: {q['question']}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        top_idx = np.argsort(-sims)[:CANDIDATE_K]

        for i in top_idx:
            sim = float(sims[i])
            auth_w = get_authority_weight(chunks[i]["authority_level"])
            label = 1 if int(i) in relevant_idx else 0
            X_rows.append([sim, auth_w])
            y_rows.append(label)
            group_rows.append(qi)

    return np.array(X_rows), np.array(y_rows), np.array(group_rows)


# ── leave-one-query-out cross-validation ──────────────────────────────────────

def loqo_cross_validate(X, y, groups, n_questions):
    """
    For each question, refit on all OTHER questions' rows, evaluate on
    the held-out question's rows. Returns list of per-fold AUC scores
    (skipping folds where the held-out question has only one class,
    since AUC is undefined there).
    """
    fold_aucs = []
    fold_effective_alphas = []

    for held_out_qi in range(n_questions):
        train_mask = groups != held_out_qi
        test_mask = groups == held_out_qi

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue  # can't fit or can't evaluate AUC with only one class

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        fold_aucs.append(auc)

        w1, w2 = clf.coef_[0]
        # only compute effective alpha if both weights are positive and
        # meaningful — a negative weight would mean "more similar/more
        # authoritative = LESS likely relevant", which would indicate the
        # model found a degenerate fit worth flagging separately
        if w1 > 0 and w2 > 0:
            fold_effective_alphas.append(w1 / (w1 + w2))

    return fold_aucs, fold_effective_alphas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEK 2035 — Learn Alpha via Logistic Regression")
    print("=" * 70)

    questions, skipped = load_ground_truth_questions()
    print(f"\nGround truth questions with verified relevant_chunks: {len(questions)}")
    print(f"Skipped (no verified chunks yet): {skipped}")

    if len(questions) < MIN_QUESTIONS_FOR_TRUST:
        print(f"\n⚠️  WARNING: only {len(questions)} usable questions — below the "
              f"{MIN_QUESTIONS_FOR_TRUST} recommended for a trustworthy fit. "
              f"Proceeding anyway, but treat the result as illustrative only, "
              f"not a production-ready weight.")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {CORPUS_DIR}/")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    print(f"\nBuilding training rows from {len(questions)} questions "
          f"(top-{CANDIDATE_K} candidates each)...")
    X, y, groups = build_training_rows(questions, model, chunk_embeddings, chunks)
    print(f"  {len(X)} total (chunk, label) rows, {int(y.sum())} positive, "
          f"{len(y) - int(y.sum())} negative")

    if y.sum() == 0:
        print("❌ No positive labels found at all — cannot fit. Check that "
              "relevant_chunks idx values actually appear within the top-20 "
              "candidates for their questions (if not, the embedding model "
              "used here may differ from the one used to build the ground "
              "truth, or top-20 is too narrow).")
        return

    # ── fit on ALL data (final model) ────────────────────────────────────────
    final_clf = LogisticRegression()
    final_clf.fit(X, y)
    w1_final, w2_final = final_clf.coef_[0]
    bias_final = final_clf.intercept_[0]

    print(f"\n{'='*70}")
    print("FITTED WEIGHTS (on full dataset)")
    print(f"{'='*70}")
    print(f"  w_similarity        = {w1_final:.4f}")
    print(f"  w_authority         = {w2_final:.4f}")
    print(f"  bias                = {bias_final:.4f}")

    if w1_final > 0 and w2_final > 0:
        alpha_effective = w1_final / (w1_final + w2_final)
        print(f"\n  alpha_effective = w_similarity / (w_similarity + w_authority)")
        print(f"                  = {alpha_effective:.4f}")
        print(f"  (compare to the unvalidated default of 0.7 used throughout the project)")
    else:
        alpha_effective = None
        print(f"\n  ⚠️  At least one weight is negative or zero — cannot derive a")
        print(f"      sensible alpha_effective. This suggests the current feature")
        print(f"      set or ground truth may not support this simple 2-feature model.")

    # ── leave-one-query-out validation ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("LEAVE-ONE-QUERY-OUT CROSS-VALIDATION")
    print(f"{'='*70}")
    fold_aucs, fold_alphas = loqo_cross_validate(X, y, groups, len(questions))

    if fold_aucs:
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        print(f"  Folds evaluated : {len(fold_aucs)} / {len(questions)} "
              f"(some skipped — held-out question had only one class present)")
        print(f"  Mean AUC        : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  (AUC = 0.5 is random; AUC = 1.0 is perfect separation)")

        if fold_alphas:
            print(f"\n  Per-fold effective alpha: mean={np.mean(fold_alphas):.4f}, "
                  f"std={np.std(fold_alphas):.4f}, "
                  f"range=[{min(fold_alphas):.4f}, {max(fold_alphas):.4f}]")
            if np.std(fold_alphas) > 0.15:
                print(f"  ⚠️  High variance across folds — the learned alpha is NOT")
                print(f"      stable across different held-out questions. This is a")
                print(f"      concrete sign of overfitting at this data size.")
    else:
        print("  ⚠️  No valid folds — every held-out question had only one class")
        print("      present, so AUC could not be computed for any fold. This "
              "      itself is a sign the dataset is too small/imbalanced for "
              "      reliable cross-validation.")
        mean_auc, std_auc = None, None

    # ── final recommendation ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    if len(questions) < MIN_QUESTIONS_FOR_TRUST:
        print(f"  ❌ DO NOT replace the grid-search alpha with this fitted value yet.")
        print(f"     Only {len(questions)} questions have verified ground truth —")
        print(f"     below the {MIN_QUESTIONS_FOR_TRUST}-question threshold set before")
        print(f"     this experiment. Treat alpha_effective as a data point to compare")
        print(f"     against the grid sweep result, not a replacement for it.")
    elif fold_aucs and np.std(fold_alphas) > 0.15:
        print(f"  ⚠️  Fitted alpha is UNSTABLE across folds (std={np.std(fold_alphas):.4f}).")
        print(f"     Prefer the grid-search alpha until more ground truth is added.")
    elif fold_aucs and mean_auc > 0.7:
        print(f"  ✅ Fitted model shows reasonable, stable generalisation")
        print(f"     (mean LOQO AUC = {mean_auc:.4f}). alpha_effective = "
              f"{alpha_effective:.4f} is a defensible candidate to test in")
        print(f"     production, alongside (not necessarily instead of) the grid-search value.")
    else:
        print(f"  ⚠️  Validation results are weak or inconclusive. Keep using the")
        print(f"     grid-search alpha for now.")

    # ── save ──────────────────────────────────────────────────────────────────
    result = {
        "n_questions_used": len(questions),
        "n_questions_skipped": skipped,
        "n_training_rows": len(X),
        "n_positive": int(y.sum()),
        "fitted_weights": {"w_similarity": float(w1_final), "w_authority": float(w2_final), "bias": float(bias_final)},
        "alpha_effective": alpha_effective,
        "loqo_mean_auc": mean_auc,
        "loqo_std_auc": std_auc,
        "loqo_n_folds": len(fold_aucs),
        "loqo_alpha_mean": float(np.mean(fold_alphas)) if fold_alphas else None,
        "loqo_alpha_std": float(np.std(fold_alphas)) if fold_alphas else None,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
