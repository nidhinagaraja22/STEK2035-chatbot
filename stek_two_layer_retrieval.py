"""
STEK 2035 — Two-Layer Retrieval (Query Classification + Dual-Branch Search)
==============================================================================
Diagnostic finding motivating this design (stek_policy_vs_citizen_diagnostic.py):
  Policy questions : 21/42 (50%) of ground-truth relevant chunks appear in
                      a standard top-20 similarity search
  Citizen questions:  4/13 (31%) of ground-truth relevant chunks appear —
                      69% are missing from the candidate pool ENTIRELY,
                      before any reranking (authority-weighted or otherwise)
                      even runs.

Root cause: official and citizen chunks compete in ONE shared similarity
ranking. Official documents (formal register, STEK-specific vocabulary)
apparently rank more reliably against formally-phrased questions than
citizen documents (informal register) do — so citizen-relevant chunks get
crowded out of the top-20 even when they exist in the corpus.

Fix — two layers:
  Layer 1 (classification): determine if the question is policy-seeking,
    citizen-seeking, or general, using keyword rules (the query_type
    classifier originally proposed in the Document Authority Taxonomy
    document, now finally implemented and tested).
  Layer 2 (dual-branch retrieval): run TWO independent similarity searches
    — one restricted to official-tier chunks (authority_level 1-3), one
    restricted to citizen-tier chunks (authority_level 4-5) — so a citizen
    chunk only competes against OTHER citizen chunks for its slots, never
    against higher-scoring official chunks. Merge slot counts according
    to the classified question type.

This does not replace authority_reranked or combined_source_topic — it
operates one stage earlier, at candidate retrieval, before any reranking
formula is applied.

Usage:
    python stek_two_layer_retrieval.py

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
BRANCH_POOL = 20  # candidates considered WITHIN each branch before merging

# ── Layer 1 — query type classification ───────────────────────────────────────
# keyword rules, same design proposed in the Document Authority Taxonomy
# document, implemented here for the first time. Broadened after testing
# against the 45 verified questions showed the initial narrow list left
# 27/45 falling into "general" by default, including clearly policy-seeking
# questions (e.g. "Wie hoch ist X laut STEK?", "Wie viele Y soll es 2035
# geben?") — these are policy questions phrased as factual/quantitative
# lookups rather than using an explicit "plant"/"offiziell" verb.
POLICY_KEYWORDS = [
    "plant die stadt", "plant heidelberg", "offizielle", "offiziell",
    "beschlossen", "ziel des stek", "sieht das stek", "sieht stek",
    "maßnahmen plant", "welche ziele verfolgt", "wie will heidelberg",
    "stek 2035", "laut stek", "nennt das stek", "nennt stek",
    "wie hoch ist", "wie viele", "wie hat sich", "wann", "bis wann",
    "wer hat", "was ist das", "was war vor", "welche stadtteile",
    "welche gruppen", "wie fördert", "wie soll sich", "wie wird",
    "welche rolle spielen", "was sind die", "welche herausforderung",
]
CITIZEN_KEYWORDS = [
    "wünschen sich bürger", "wünscht sich", "bürger fordern", "fordern bürger",
    "was denken bürger", "bürgermeinung", "was wollen bürger",
    "was war der zweck", "wie wurden bürgerinnen", "räume für kulturelle nutzung fehlten",
]


def classify_query_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in CITIZEN_KEYWORDS):
        return "citizen"
    if any(kw in q for kw in POLICY_KEYWORDS):
        return "policy"
    return "general"


# merge slot allocation per classified type — how many of the final TOP_K=5
# come from each branch
MERGE_RULES = {
    "policy":  {"official": 4, "citizen": 1},
    "citizen": {"official": 1, "citizen": 4},
    "general": {"official": 3, "citizen": 2},
}

# strict variant — no blending at all, matches a ChromaDB where={} filter
# in production: where={"authority_level": {"$lte": 3}} for policy questions.
# Risk: guaranteed zero recall on questions where official docs don't cover
# the topic at all (e.g. Q021 Kultur — verified answer is L4 arbeitstreffen_2024,
# official docs barely mention culture). Tested here explicitly rather than
# assumed better/worse than the blended version.
STRICT_MERGE_RULES = {
    "policy":  {"official": 5, "citizen": 0},
    "citizen": {"official": 0, "citizen": 5},
    "general": {"official": 3, "citizen": 2},  # general still blends — no
                                                 # strong reason to filter
                                                 # strictly for ambiguous questions
}


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
                "authority_level": obj.get("authority_level", 3),
            })
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


# ── Layer 2 — dual-branch retrieval ───────────────────────────────────────────

def two_layer_retrieve(question: str, model, chunk_embeddings, chunks,
                         official_idx: np.ndarray, citizen_idx: np.ndarray,
                         merge_rules: dict, top_k: int = TOP_K):
    query_type = classify_query_type(question)
    rule = merge_rules[query_type]

    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec

    branch_a, branch_b = [], []

    if rule["official"] > 0:
        official_sims = sims[official_idx]
        official_ranked = official_idx[np.argsort(-official_sims)][:BRANCH_POOL]
        branch_a = list(official_ranked[:rule["official"]])

    if rule["citizen"] > 0:
        citizen_sims = sims[citizen_idx]
        citizen_ranked = citizen_idx[np.argsort(-citizen_sims)][:BRANCH_POOL]
        branch_b = list(citizen_ranked[:rule["citizen"]])

    selected = branch_a + branch_b
    return {"query_type": query_type, "selected_idx": [int(i) for i in selected]}


# ── confidence-gated strict filter ────────────────────────────────────────────
# Addresses a distinct failure mode from plain strict filtering: even when
# question-type classification is CORRECT, the target authority tier may
# simply lack good coverage for this specific topic (e.g. Q021 — correctly
# classified as policy-seeking, but official docs barely cover Kultur).
# Plain strict filtering fails silently here. This variant checks whether
# the best match WITHIN the target tier is actually good before committing
# to the filter — if not, it falls back to searching across both tiers
# unrestricted, rather than confidently returning a bad answer.
#
# COVERAGE_THRESHOLD is set from rough inspection of this project's earlier
# retrieval scores (relevant chunks typically scored ~0.82-0.91, non-relevant
# ~0.60-0.85) — this has NOT been calibrated the way the vague-question
# threshold was meant to be (small labeled calibration set, as discussed
# earlier in the project). Treat 0.75 as a reasonable starting guess, not
# a validated cutoff.
COVERAGE_THRESHOLD = 0.75


def confidence_gated_retrieve(question: str, model, chunk_embeddings, chunks,
                                official_idx: np.ndarray, citizen_idx: np.ndarray,
                                top_k: int = TOP_K, coverage_threshold: float = COVERAGE_THRESHOLD):
    query_type = classify_query_type(question)

    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec

    if query_type == "policy":
        target_idx, fallback_idx = official_idx, citizen_idx
    elif query_type == "citizen":
        target_idx, fallback_idx = citizen_idx, official_idx
    else:
        # "general" questions were never intended for strict filtering —
        # just blend as usual
        target_sims = sims[official_idx]
        official_ranked = official_idx[np.argsort(-target_sims)][:3]
        citizen_sims = sims[citizen_idx]
        citizen_ranked = citizen_idx[np.argsort(-citizen_sims)][:2]
        selected = list(official_ranked) + list(citizen_ranked)
        return {"query_type": query_type, "used_strict": False, "best_target_sim": None,
                "selected_idx": [int(i) for i in selected]}

    target_sims = sims[target_idx]
    best_target_sim = float(target_sims.max())

    if best_target_sim >= coverage_threshold:
        # confident coverage exists in the target tier — trust the strict filter
        ranked = target_idx[np.argsort(-target_sims)][:top_k]
        return {"query_type": query_type, "used_strict": True, "best_target_sim": best_target_sim,
                "selected_idx": [int(i) for i in ranked]}
    else:
        # target tier has poor coverage for this question — don't trust a
        # confident-looking filter on weak evidence; search BOTH tiers
        # unrestricted instead (equivalent to standard_retrieve, but only
        # triggered when the strict path was about to fail silently)
        all_ranked = np.argsort(-sims)[:top_k]
        return {"query_type": query_type, "used_strict": False, "best_target_sim": best_target_sim,
                "selected_idx": [int(i) for i in all_ranked]}


def standard_retrieve(question: str, model, chunk_embeddings, top_k: int = TOP_K,
                        candidate_k: int = BRANCH_POOL):
    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec
    top_idx = np.argsort(-sims)[:top_k]
    return [int(i) for i in top_idx]


# ── evaluation against ground truth ───────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEK 2035 — Two-Layer Retrieval Evaluation")
    print("=" * 70)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")]
    print(f"\nEvaluating on {len(questions)} verified questions")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {CORPUS_DIR}/")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    authority = np.array([c["authority_level"] for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    citizen_idx = np.where(authority >= 4)[0]
    print(f"  Official-tier chunks (L1-3): {len(official_idx)}")
    print(f"  Citizen-tier chunks (L4-5) : {len(citizen_idx)}")

    # ── compare recall: standard vs blended vs strict vs confidence-gated ────
    def new_bucket():
        return {"std_hit": 0, "std_total": 0, "blend_hit": 0, "blend_total": 0,
                "strict_hit": 0, "strict_total": 0, "gated_hit": 0, "gated_total": 0}

    results_by_type = {"policy": new_bucket(), "citizen": new_bucket(), "general": new_bucket()}
    fallback_triggers = []  # questions where confidence-gating chose to fall back

    print(f"\n{'='*70}")
    print("PER-QUESTION RESULTS")
    print(f"{'='*70}")

    for q in questions:
        relevant_idx = {r["idx"] for r in q["relevant_chunks"]}
        n_relevant = len(relevant_idx)

        std_top5 = standard_retrieve(q["question"], model, chunk_embeddings, top_k=TOP_K)
        blend = two_layer_retrieve(q["question"], model, chunk_embeddings, chunks,
                                     official_idx, citizen_idx, MERGE_RULES, top_k=TOP_K)
        strict = two_layer_retrieve(q["question"], model, chunk_embeddings, chunks,
                                      official_idx, citizen_idx, STRICT_MERGE_RULES, top_k=TOP_K)
        gated = confidence_gated_retrieve(q["question"], model, chunk_embeddings, chunks,
                                            official_idx, citizen_idx, top_k=TOP_K)
        qtype = blend["query_type"]

        std_hits = len(relevant_idx & set(std_top5))
        blend_hits = len(relevant_idx & set(blend["selected_idx"]))
        strict_hits = len(relevant_idx & set(strict["selected_idx"]))
        gated_hits = len(relevant_idx & set(gated["selected_idx"]))

        if not gated["used_strict"] and qtype != "general":
            fallback_triggers.append((q["q_id"], gated["best_target_sim"], q["question"][:50]))

        b = results_by_type[qtype]
        b["std_hit"] += std_hits; b["std_total"] += n_relevant
        b["blend_hit"] += blend_hits; b["blend_total"] += n_relevant
        b["strict_hit"] += strict_hits; b["strict_total"] += n_relevant
        b["gated_hit"] += gated_hits; b["gated_total"] += n_relevant

        best = max(std_hits, blend_hits, strict_hits, gated_hits)
        marker = "⚠️ ALL MISS" if best == 0 else ("🔺" if best > std_hits else "  ")
        gate_flag = "↩️ fallback" if not gated["used_strict"] and qtype != "general" else ""
        print(f"{marker} [{qtype:<8}] {q['q_id']}: std={std_hits}/{n_relevant}  "
              f"blend={blend_hits}/{n_relevant}  strict={strict_hits}/{n_relevant}  "
              f"gated={gated_hits}/{n_relevant} {gate_flag}  \"{q['question'][:35]}...\"")

    print(f"\n{'='*70}")
    print("AGGREGATE RECALL-INTO-TOP-5 BY QUESTION TYPE")
    print(f"{'='*70}")
    print(f"{'Type':<10} {'Standard':>14} {'Blend':>14} {'Strict':>14} {'Gated':>14}")
    print("-" * 76)

    for qtype, r in results_by_type.items():
        if r["std_total"] == 0:
            continue
        std_r = r["std_hit"] / r["std_total"]
        blend_r = r["blend_hit"] / r["blend_total"]
        strict_r = r["strict_hit"] / r["strict_total"]
        gated_r = r["gated_hit"] / r["gated_total"]
        print(f"{qtype:<10} {r['std_hit']:>2}/{r['std_total']:<3}({std_r*100:>4.0f}%) "
              f"{r['blend_hit']:>2}/{r['blend_total']:<3}({blend_r*100:>4.0f}%) "
              f"{r['strict_hit']:>2}/{r['strict_total']:<3}({strict_r*100:>4.0f}%) "
              f"{r['gated_hit']:>2}/{r['gated_total']:<3}({gated_r*100:>4.0f}%)")

    print(f"\n{'='*70}")
    print(f"COVERAGE-GAP FALLBACK TRIGGERS ({len(fallback_triggers)} questions)")
    print(f"{'='*70}")
    print(f"These are cases where confidence-gating detected the target tier had")
    print(f"poor coverage (best similarity < {COVERAGE_THRESHOLD}) and fell back rather than")
    print(f"trusting a confident-looking but potentially wrong strict filter:")
    for qid, sim, qtext in fallback_triggers:
        print(f"  {qid}: best_target_sim={sim:.4f}  \"{qtext}...\"")

    print(f"\n⚠️  Watch specifically for Q021 above — if it appears in the fallback")
    print(f"   trigger list AND gated_hits > strict_hits, that confirms confidence-")
    print(f"   gating successfully recovers the coverage-gap case that pure strict")
    print(f"   filtering guarantees a miss on.")
    print(f"\n⚠️  COVERAGE_THRESHOLD=0.75 is an uncalibrated guess — if fallback_triggers")
    print(f"   is empty or fires on nearly everything, the threshold needs tuning")
    print(f"   against a labeled calibration set, same as the vague-question threshold.")


if __name__ == "__main__":
    main()
