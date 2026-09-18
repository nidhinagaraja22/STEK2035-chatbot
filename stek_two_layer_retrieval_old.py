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
                         top_k: int = TOP_K):
    query_type = classify_query_type(question)
    rule = MERGE_RULES[query_type]

    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec

    # Branch A — search ONLY within official-tier chunks (L1-3)
    official_sims = sims[official_idx]
    official_ranked = official_idx[np.argsort(-official_sims)][:BRANCH_POOL]
    branch_a = list(official_ranked[:rule["official"]])

    # Branch B — search ONLY within citizen-tier chunks (L4-5)
    citizen_sims = sims[citizen_idx]
    citizen_ranked = citizen_idx[np.argsort(-citizen_sims)][:BRANCH_POOL]
    branch_b = list(citizen_ranked[:rule["citizen"]])

    selected = branch_a + branch_b
    return {"query_type": query_type, "selected_idx": [int(i) for i in selected]}


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

    # ── compare recall: standard single-layer vs two-layer ──────────────────
    results_by_type = {"policy": {"std_hit": 0, "std_total": 0, "two_hit": 0, "two_total": 0},
                        "citizen": {"std_hit": 0, "std_total": 0, "two_hit": 0, "two_total": 0},
                        "general": {"std_hit": 0, "std_total": 0, "two_hit": 0, "two_total": 0}}

    print(f"\n{'='*70}")
    print("PER-QUESTION RESULTS")
    print(f"{'='*70}")

    for q in questions:
        relevant_idx = {r["idx"] for r in q["relevant_chunks"]}
        n_relevant = len(relevant_idx)

        std_top5 = standard_retrieve(q["question"], model, chunk_embeddings, top_k=TOP_K)
        two_layer = two_layer_retrieve(q["question"], model, chunk_embeddings, chunks,
                                         official_idx, citizen_idx, top_k=TOP_K)
        qtype = two_layer["query_type"]
        two_top5 = two_layer["selected_idx"]

        std_hits = len(relevant_idx & set(std_top5))
        two_hits = len(relevant_idx & set(two_top5))

        results_by_type[qtype]["std_hit"] += std_hits
        results_by_type[qtype]["std_total"] += n_relevant
        results_by_type[qtype]["two_hit"] += two_hits
        results_by_type[qtype]["two_total"] += n_relevant

        marker = "🔺" if two_hits > std_hits else ("🔻" if two_hits < std_hits else "  ")
        print(f"{marker} [{qtype:<8}] {q['q_id']}: standard={std_hits}/{n_relevant}  "
              f"two-layer={two_hits}/{n_relevant}  \"{q['question'][:45]}...\"")

    print(f"\n{'='*70}")
    print("AGGREGATE RECALL-INTO-TOP-5 BY QUESTION TYPE")
    print(f"{'='*70}")
    print(f"{'Type':<10} {'Standard':>18} {'Two-Layer':>18} {'Change':>10}")
    print("-" * 60)

    for qtype, r in results_by_type.items():
        if r["std_total"] == 0:
            continue
        std_recall = r["std_hit"] / r["std_total"]
        two_recall = r["two_hit"] / r["two_total"]
        change = two_recall - std_recall
        print(f"{qtype:<10} {r['std_hit']:>3}/{r['std_total']:<3} ({std_recall*100:>5.1f}%)   "
              f"{r['two_hit']:>3}/{r['two_total']:<3} ({two_recall*100:>5.1f}%)   "
              f"{change*100:>+7.1f}pp")

    print(f"\n⚠️  This evaluates recall@5 (relevant chunk in final top-5), stricter")
    print(f"   than the earlier diagnostic's recall-into-candidate-pool@20 — expect")
    print(f"   lower absolute numbers here, focus on the STANDARD vs TWO-LAYER change.")


if __name__ == "__main__":
    main()
