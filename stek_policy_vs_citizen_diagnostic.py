"""
STEK 2035 — Policy vs Citizen Question Diagnostic
====================================================
The pooled logistic regression (stek_fit_alpha_logistic_regression.py)
learned weights dominated by authority_weight (w_authority=1.96 vs
w_similarity=0.21), traced to a 74%/26% skew toward official documents
in the ground truth's relevant_chunks. That skew itself traces to a
34-vs-10 imbalance between policy-seeking and citizen-seeking questions
in the 45-question verified set.

Rather than fit a second small (n=10) logistic regression on the
citizen subset alone — which would carry its own, likely worse,
overfitting risk — this script checks the more fundamental question
descriptively: do relevant chunks for policy-seeking questions actually
look different (higher authority, possibly lower raw similarity) than
relevant chunks for citizen-seeking questions? If yes, this confirms
the pooled model's authority-dominance is a pooling artifact, not
evidence that authority never matters.

Usage:
    python stek_policy_vs_citizen_diagnostic.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR   = Path("corpus/corpus_v2")
CHUNKS_PATH  = CORPUS_DIR / "corpus_v2_chunks.jsonl"
EMB_PATH     = CORPUS_DIR / "embeddings_v2_e5base.npy"
EMBED_MODEL  = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

CANDIDATE_K = 20
AUTHORITY_WEIGHT = {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.30}


def get_authority_weight(level) -> float:
    return AUTHORITY_WEIGHT.get(int(level), 0.70)


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append({
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
    print("STEK 2035 — Policy vs Citizen Question Diagnostic")
    print("=" * 70)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")]

    policy_qs  = [q for q in questions if q.get("is_official_policy") is True]
    citizen_qs = [q for q in questions if q.get("is_official_policy") is False]
    print(f"\nPolicy-seeking questions : {len(policy_qs)}")
    print(f"Citizen-seeking questions: {len(citizen_qs)}")
    print(f"(ambiguous/null excluded : {len(questions) - len(policy_qs) - len(citizen_qs)})")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {CORPUS_DIR}/")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    def collect_stats(question_group, label):
        relevant_sims, relevant_auths = [], []
        nonrelevant_sims, nonrelevant_auths = [], []

        for q in question_group:
            relevant_idx = {r["idx"] for r in q["relevant_chunks"]}
            q_vec = model.encode(f"query: {q['question']}")
            q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
            sims = chunk_embeddings @ q_vec
            top_idx = np.argsort(-sims)[:CANDIDATE_K]

            for i in top_idx:
                sim = float(sims[i])
                auth_w = get_authority_weight(chunks[i]["authority_level"])
                if int(i) in relevant_idx:
                    relevant_sims.append(sim)
                    relevant_auths.append(auth_w)
                else:
                    nonrelevant_sims.append(sim)
                    nonrelevant_auths.append(auth_w)

        print(f"\n--- {label} (n={len(question_group)} questions) ---")
        if relevant_sims:
            print(f"  Relevant chunks    : n={len(relevant_sims):<4} "
                  f"mean_similarity={np.mean(relevant_sims):.4f}  "
                  f"mean_authority={np.mean(relevant_auths):.4f}")
        else:
            print("  Relevant chunks    : none found in candidate pool")
        if nonrelevant_sims:
            print(f"  Non-relevant chunks: n={len(nonrelevant_sims):<4} "
                  f"mean_similarity={np.mean(nonrelevant_sims):.4f}  "
                  f"mean_authority={np.mean(nonrelevant_auths):.4f}")

        return {
            "relevant_sim": float(np.mean(relevant_sims)) if relevant_sims else None,
            "relevant_auth": float(np.mean(relevant_auths)) if relevant_auths else None,
            "nonrelevant_sim": float(np.mean(nonrelevant_sims)) if nonrelevant_sims else None,
            "nonrelevant_auth": float(np.mean(nonrelevant_auths)) if nonrelevant_auths else None,
        }

    policy_stats = collect_stats(policy_qs, "POLICY-SEEKING questions")
    citizen_stats = collect_stats(citizen_qs, "CITIZEN-SEEKING questions")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if policy_stats["relevant_auth"] and citizen_stats["relevant_auth"]:
        auth_gap = policy_stats["relevant_auth"] - citizen_stats["relevant_auth"]
        print(f"\nAuthority of RELEVANT chunks — policy vs citizen questions:")
        print(f"  Policy questions : {policy_stats['relevant_auth']:.4f}")
        print(f"  Citizen questions: {citizen_stats['relevant_auth']:.4f}")
        print(f"  Gap              : {auth_gap:+.4f}")

        if auth_gap > 0.15:
            print(f"\n  ✅ CONFIRMS the hypothesis: relevant chunks for policy questions")
            print(f"     genuinely come from higher-authority sources than relevant")
            print(f"     chunks for citizen questions. The pooled model's authority-")
            print(f"     dominance reflects a REAL difference between question types,")
            print(f"     not just an artifact — but it should NOT be applied as one")
            print(f"     global weight. Use query-type classification (as originally")
            print(f"     designed in the Document Authority Taxonomy) to apply high")
            print(f"     authority weight ONLY for policy-seeking questions.")
        else:
            print(f"\n  ⚠️  Gap is smaller than expected — the pooled model's extreme")
            print(f"     authority-dominance may still be partly a pooling/sample-size")
            print(f"     artifact rather than a clean policy-vs-citizen split.")

    if citizen_stats["relevant_sim"] and citizen_stats["nonrelevant_sim"]:
        sim_discriminative_gap = citizen_stats["relevant_sim"] - citizen_stats["nonrelevant_sim"]
        print(f"\nFor citizen questions specifically — does similarity actually")
        print(f"discriminate relevant from non-relevant chunks?")
        print(f"  Relevant chunks similarity    : {citizen_stats['relevant_sim']:.4f}")
        print(f"  Non-relevant chunks similarity: {citizen_stats['nonrelevant_sim']:.4f}")
        print(f"  Gap: {sim_discriminative_gap:+.4f}  "
              f"({'similarity IS discriminative here' if sim_discriminative_gap > 0.02 else 'weak signal — small sample (n=10) caveat applies'})")

    print(f"\n⚠️  n=10 citizen questions is a small sample — treat these citizen-side")
    print(f"   numbers as suggestive, not conclusive. This diagnostic informs whether")
    print(f"   query-type-conditioned authority weighting is worth implementing, not")
    print(f"   a final validated weight for the citizen regime.")


if __name__ == "__main__":
    main()
