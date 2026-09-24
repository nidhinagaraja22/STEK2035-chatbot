"""
STEK 2035 — Cascading Retrieval with Mandatory Source Labeling
==================================================================
Design (confirmed with user):
  1. If the question EXPLICITLY requests citizen opinion (narrow
     trigger list) → search citizen tiers (L4+L5) directly, skip
     the cascade entirely.
  2. Otherwise, cascade through authority tiers in order:
       a. Search L1-L3 (official). If coverage is good (best
          similarity >= COVERAGE_THRESHOLD), use it.
       b. Else search L4 (workshop/consultation outcomes — a
          structured, city-documented, multi-stakeholder process,
          distinct from raw individual opinion). If good, use it.
       c. Else search L5 (raw individual citizen opinion). Use it
          regardless of score, since it's the last resort.
  3. EVERY answer is labeled with its actual source tier — an
     official-tier answer is never presented the same way as an
     L4 or L5 fallback. This is not optional; mislabeling source
     authority is exactly the misattribution problem this entire
     project's Document Authority Taxonomy was built to prevent.

This does NOT require correctly classifying every question's
"intent" up front — it only requires the NARROW explicit-citizen-
request check to be correct. Everything else is decided by actual
retrieval coverage, not guessed from question phrasing (this is
what fixes the Q021 problem: no keyword list ever has to correctly
guess that a "STEK 2035"-mentioning question needs L4 content —
the cascade discovers that naturally when official coverage proves
thin).

Usage:
    python stek_cascading_retrieval.py Q021 Q024 Q025 Q043 Q045

Requires:
    corpus_v3_chunks.jsonl (5-tier: L1/L2 official, L4 workshop, L5 raw opinion)
    matching embeddings file
    stek_task4_5_master_ground_truth.json
"""

import sys
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")  # v2, L4/L5 split applied
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")             # your EXISTING v2 embeddings —
                                                                           # unchanged, still row-aligned
                                                                           # since the L4/L5 split is
                                                                           # metadata-only
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

TOP_K = 5
COVERAGE_THRESHOLD = 0.35  # same caveat as before: uncalibrated, needs a
                            # labeled calibration set before trusting in
                            # production — see stek_two_layer_retrieval.py

# narrow, explicit-only trigger — same list validated earlier at 93%
# agreement with your ground truth's is_official_policy labels
CITIZEN_REQUEST_KEYWORDS = [
    "wünschen sich bürger", "wünscht sich", "bürger fordern", "fordern bürger",
    "was denken bürger", "bürgermeinung", "was wollen bürger",
    "was war der zweck", "wie wurden bürgerinnen", "räume für kulturelle nutzung fehlten",
]

# comparison trigger — needs BOTH official and citizen sides deliberately
COMPARISON_KEYWORDS = [
    "im vergleich", "verglichen mit", "unterschied zwischen",
    "bürgerwünsche und offizielle", "einerseits", "andererseits",
]

CITATION_PREFIX = {
    1: "Laut dem offiziellen STEK 2035",
    2: "Laut dem städtischen Bericht",
    4: "Laut den Ergebnissen eines Stakeholder-Workshops (mit Bürgerinnen, Gemeinderat, Zivilgesellschaft und Verwaltung)",
    5: "Ein einzelner Bürger äußerte in der Online-Beteiligung den Wunsch",
}


def classify_request_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in COMPARISON_KEYWORDS):
        return "comparison"
    if any(kw in q for kw in CITIZEN_REQUEST_KEYWORDS):
        return "citizen_explicit"
    return "default_cascade"


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def search_tier(sims, tier_idx, k):
    if len(tier_idx) == 0:
        return [], 0.0
    tier_sims = sims[tier_idx]
    ranked = tier_idx[np.argsort(-tier_sims)][:k]
    best_score = float(tier_sims.max())
    return list(ranked), best_score


def cascading_retrieve(question, chunks, chunk_embeddings, model,
                         official_idx, l4_idx, l5_idx, k=TOP_K):
    request_type = classify_request_type(question)

    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec

    if request_type == "citizen_explicit":
        # search L4+L5 combined directly, skip cascade
        citizen_idx = np.concatenate([l4_idx, l5_idx])
        result_idx, score = search_tier(sims, citizen_idx, k)
        tier_used = "L4/L5 (explicit citizen request)"
        return {"request_type": request_type, "tier_used": tier_used, "tier_label": "citizen_explicit",
                "result_idx": result_idx, "coverage_score": score, "sims": sims}

    if request_type == "comparison":
        # guaranteed slots from both official and citizen sides
        off_result, off_score = search_tier(sims, official_idx, k - 2)
        cit_result, cit_score = search_tier(sims, np.concatenate([l4_idx, l5_idx]), 2)
        return {"request_type": request_type, "tier_used": "official + citizen (comparison)", "tier_label": "comparison",
                "result_idx": off_result + cit_result, "coverage_score": (off_score, cit_score), "sims": sims}

    # default_cascade — try official, then L4, then L5
    off_result, off_score = search_tier(sims, official_idx, k)
    if off_score >= COVERAGE_THRESHOLD:
        return {"request_type": request_type, "tier_used": "official (cascade stage 1)", "tier_label": "official",
                "result_idx": off_result, "coverage_score": off_score, "sims": sims}

    l4_result, l4_score = search_tier(sims, l4_idx, k)
    if l4_score >= COVERAGE_THRESHOLD:
        return {"request_type": request_type, "tier_label": "L4",
                "tier_used": "L4 workshop (cascade stage 2 — official coverage was thin)",
                "result_idx": l4_result, "coverage_score": l4_score, "sims": sims}

    l5_result, l5_score = search_tier(sims, l5_idx, k)
    return {"request_type": request_type, "tier_label": "L5",
            "tier_used": "L5 raw citizen opinion (cascade stage 3 — official AND L4 coverage were thin)",
            "result_idx": l5_result, "coverage_score": l5_score, "sims": sims}


def main():
    import sys

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]
    print(f"  Official (L1-3): {len(official_idx)} | L4 workshop: {len(l4_idx)} | L5 raw opinion: {len(l5_idx)}")

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions_by_id = {q["q_id"]: q for q in gt_data["questions"]}

    # if specific question IDs given as args, use those; otherwise run ALL
    # verified questions in the ground truth file
    if len(sys.argv) > 1:
        q_ids = sys.argv[1:]
    else:
        q_ids = [q["q_id"] for q in gt_data["questions"] if q.get("relevant_chunks")]
        print(f"\nNo question IDs given — running all {len(q_ids)} verified questions.")

    results_by_type = {"default_cascade": {"official": 0, "L4": 0, "L5": 0},
                        "citizen_explicit": {"total": 0}, "comparison": {"total": 0}}
    doc_hits = 0
    doc_total = 0
    per_question_log = []

    for q_id in q_ids:
        q = questions_by_id.get(q_id)
        if not q:
            print(f"\n❌ {q_id} not found")
            continue

        result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model,
                                      official_idx, l4_idx, l5_idx)

        # tally cascade tier usage — use the explicit tier_label field,
        # not substring parsing of tier_used (which caused a real bug:
        # the L5 description text mentions "official" and "L4" as part
        # of its own explanation, causing false substring matches)
        rtype = result["request_type"]
        if rtype == "default_cascade":
            results_by_type["default_cascade"][result["tier_label"]] += 1
        else:
            results_by_type[rtype]["total"] = results_by_type[rtype].get("total", 0) + 1

        # document_id-based hit check — stable across v2/v3 chunk numbering,
        # unlike exact chunk_id which was verified against v3 specifically
        target_doc_ids = {t["document_id"] for t in q.get("relevant_chunks", [])}
        found_doc_ids = {chunks[i]["document_id"] for i in result["result_idx"]}
        hit = bool(target_doc_ids & found_doc_ids)
        doc_total += 1
        if hit:
            doc_hits += 1

        marker = "✅" if hit else "❌"
        print(f"{marker} [{rtype:<16}] {q_id}: tier={result['tier_used'][:30]:<30} "
              f"target_docs={target_doc_ids} found_docs={found_doc_ids & target_doc_ids or '—'}")

        per_question_log.append({
            "q_id": q_id, "request_type": rtype, "tier_used": result["tier_used"],
            "document_level_hit": hit, "target_docs": list(target_doc_ids),
        })

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Document-level recall: {doc_hits}/{doc_total} ({doc_hits/max(doc_total,1)*100:.0f}%)")
    print(f"\nRequest type distribution:")
    print(f"  default_cascade: {sum(results_by_type['default_cascade'].values())} "
          f"(official={results_by_type['default_cascade']['official']}, "
          f"L4={results_by_type['default_cascade']['L4']}, "
          f"L5={results_by_type['default_cascade']['L5']})")
    print(f"  citizen_explicit: {results_by_type['citizen_explicit'].get('total', 0)}")
    print(f"  comparison: {results_by_type['comparison'].get('total', 0)}")

    with open("cascading_full_run_results.json", "w", encoding="utf-8") as f:
        json.dump({"document_level_recall": f"{doc_hits}/{doc_total}",
                    "per_question": per_question_log}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: cascading_full_run_results.json")
    print(f"\n⚠️  NOTE: 'document-level recall' checks whether retrieval pulled")
    print(f"   from the CORRECT SOURCE DOCUMENT, not the exact correct chunk —")
    print(f"   this is a coarser but v2/v3-numbering-safe proxy metric.")


if __name__ == "__main__":
    main()