"""
STEK 2035 — Task 3: Retrieval Strategy Comparison
====================================================
Professor's Step 3 requirement: do not use a fixed "max 2 chunks
per source" rule without testing it. Compare 6 retrieval strategies
experimentally across 4 outcome dimensions.

Strategies tested:
  1. standard_topk        — baseline, plain cosine similarity top-5
  2. max_1_per_source      — diversity cap: 1 chunk per document max
  3. max_2_per_source      — diversity cap: 2 chunks per document max
  4. max_3_per_source      — diversity cap: 3 chunks per document max
  5. mmr                   — Maximal Marginal Relevance (relevance + diversity)
  6. authority_reranked    — reweight by document authority level (DAT)

Outcome metrics (per professor's Step 3):
  1. relevance    — mean cosine similarity of returned chunks (higher = better)
  2. diversity    — unique source documents in top-5 / 5 (higher = better)
  3. authority    — mean authority weight of returned chunks (higher = better,
                    weight scale: L1=1.0, L2=0.85, L3=0.70, L4=0.55, L5=0.30)
  4. dominance    — max fraction from a single source in top-5 (LOWER = better,
                    this is the flooding indicator — Online-Beteiligung problem)

Usage:
    python stek_retrieval_comparison.py

Requires:
    Existing numpy-based vector store (matches stek_pipeline_v2.py output):
      vector_store/chunks.jsonl      — one JSON object per line, chunk text + metadata
      vector_store/embeddings.npy    — (n_chunks, dim) float array, row i = chunks.jsonl line i
      vector_store/meta.json         — corpus-level metadata (optional, not required here)
      vector_store/lda_topics.json   — per-chunk topic assignments (optional, not required here)

Outputs:
    retrieval_comparison_results.json  — full per-query, per-strategy detail
    retrieval_comparison_summary.csv   — aggregated table for the paper
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = Path("vector_store")
CHUNKS_PATH      = VECTOR_STORE_DIR / "chunks.jsonl"
EMB_PATH         = VECTOR_STORE_DIR / "embeddings.npy"
META_PATH        = VECTOR_STORE_DIR / "meta.json"
LDA_TOPICS_PATH  = VECTOR_STORE_DIR / "lda_topics.json"

EMBED_MODEL     = "intfloat/multilingual-e5-base"   # 768-dim — MUST match the model
                                                      # used to build embeddings.npy.
                                                      # multilingual-e5-large = 1024-dim,
                                                      # multilingual-e5-base  = 768-dim.
                                                      # Your stored embeddings are 768-dim,
                                                      # so e5-base is what was actually used.
TOP_K           = 5                       # final number of chunks returned
CANDIDATE_K     = 20                      # candidates pulled before filtering/reranking
OUTPUT_JSON     = "retrieval_comparison_results.json"
OUTPUT_CSV      = "retrieval_comparison_summary.csv"


# ── authority ground truth (from DAT taxonomy) ────────────────────────────────
# same dict used throughout the project — filename keyword → (level, weight)
AUTHORITY_MAP = {
    "a3"                       : 1,
    "stadtentwicklungskonzept" : 1,
    "nachhaltigkeitsbericht"   : 2,
    "2015_stadtentwicklung"    : 2,
    "konzeptbericht"           : 3,
    "zukunftsreise"            : 4,
    "zukunft_gestalten"        : 4,
    "wege_zu_den_zielen"       : 4,
    "arbeitstreffen"           : 4,
    "ak_stek"                  : 4,
    "sitzung"                  : 4,
    "online-beteiligung"       : 5,
    "beteiligung"              : 5,
}

AUTHORITY_WEIGHT = {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.30}


def get_authority_level(source: str) -> int:
    src = source.lower().replace(" ", "_")
    for key, level in AUTHORITY_MAP.items():
        if key in src:
            return level
    return 3  # default fallback


def get_authority_weight(source: str) -> float:
    return AUTHORITY_WEIGHT[get_authority_level(source)]


# ── vector store loading ──────────────────────────────────────────────────────

def load_vector_store():
    """
    Load the numpy-based vector store matching stek_pipeline_v2.py output:
      chunks.jsonl    — one JSON object per line, e.g.
                        {"text": "...", "source": "STEK_A3.txt", "chunk_id": "..."}
                        (also tolerates "origin" instead of "source")
      embeddings.npy  — (n_chunks, dim) array, row i corresponds to
                        chunks.jsonl line i (same order — this alignment
                        is assumed and not re-verified here)

    Returns (chunks: list[dict], embeddings: np.ndarray L2-normalised).
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks file not found: {CHUNKS_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"embeddings file not found: {EMB_PATH}")

    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # normalise field names — accept "source" or "origin"
            source = obj.get("source") or obj.get("origin") or "unknown"
            text = obj.get("text") or obj.get("chunk") or ""
            chunks.append({"text": text, "source": source, "raw": obj})

    embeddings = np.load(EMB_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks in {CHUNKS_PATH.name} vs "
            f"{embeddings.shape[0]} rows in {EMB_PATH.name} — "
            f"chunks.jsonl and embeddings.npy must be row-aligned."
        )

    # L2-normalise once so cosine similarity is a plain dot product later
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms

    print(f"  Loaded {len(chunks)} chunks, embedding dim = {embeddings.shape[1]}")
    return chunks, embeddings


# ── test query set ────────────────────────────────────────────────────────────
# core 10 benchmark questions — extend with more from the 100-question set
TEST_QUERIES = [
    "Wie wird bezahlbarer Wohnraum in Heidelberg bis 2035 geschaffen?",
    "Welche Ziele verfolgt Heidelberg beim Klimaschutz?",
    "Wie sieht die Bürgerbeteiligung beim STEK 2035 aus?",
    "Welche Grünflächen sollen erhalten oder neu geschaffen werden?",
    "Wie soll sich der öffentliche Nahverkehr entwickeln?",
    "Welche Maßnahmen gibt es gegen Segregation?",
    "Wie plant die Stadt mit wachsender Bevölkerung?",
    "Welche Rolle spielt die Wirtschaft im STEK 2035?",
    "Was sind die zentralen Ergebnisse der Zukunftsreise 2035?",
    "Wie werden Kultur und Vielfalt im STEK 2035 berücksichtigt?",
]


# ── retrieval strategies ──────────────────────────────────────────────────────

def get_candidates(chunks: list, embeddings: np.ndarray, query_embedding: np.ndarray,
                    n: int = CANDIDATE_K) -> list:
    """
    Fetch top-n candidates by cosine similarity using brute-force numpy
    dot product (embeddings and query are both pre-normalised, so dot
    product == cosine similarity).
    """
    q = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
    sims = embeddings @ q  # (n_chunks,) cosine similarities

    top_idx = np.argsort(-sims)[:n]

    candidates = []
    for i in top_idx:
        candidates.append({
            "text"      : chunks[i]["text"],
            "source"    : chunks[i]["source"],
            "similarity": float(sims[i]),
        })
    return candidates


def strategy_standard_topk(candidates: list, k: int = TOP_K) -> list:
    """Baseline — plain top-k by similarity, no diversity/authority logic."""
    return sorted(candidates, key=lambda c: -c["similarity"])[:k]


def strategy_max_n_per_source(candidates: list, max_per_source: int, k: int = TOP_K) -> list:
    """Cap chunks per source document, preserving similarity order."""
    ranked = sorted(candidates, key=lambda c: -c["similarity"])
    selected = []
    source_count = defaultdict(int)
    for c in ranked:
        if source_count[c["source"]] < max_per_source:
            selected.append(c)
            source_count[c["source"]] += 1
        if len(selected) == k:
            break
    return selected


def strategy_mmr(candidates: list, embed_fn, lambda_param: float = 0.7, k: int = TOP_K) -> list:
    """
    Maximal Marginal Relevance — balances relevance against
    redundancy with already-selected chunks.
    Requires embeddings for pairwise similarity; recomputed here
    from candidate text for simplicity.
    """
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    vectors = embed_fn(texts)  # (n, dim)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    selected_idx = []
    remaining_idx = list(range(len(candidates)))

    # seed with most relevant
    first = max(remaining_idx, key=lambda i: candidates[i]["similarity"])
    selected_idx.append(first)
    remaining_idx.remove(first)

    while len(selected_idx) < k and remaining_idx:
        def mmr_score(i):
            relevance = candidates[i]["similarity"]
            redundancy = max(
                float(np.dot(vectors[i], vectors[j])) for j in selected_idx
            )
            return lambda_param * relevance - (1 - lambda_param) * redundancy

        best = max(remaining_idx, key=mmr_score)
        selected_idx.append(best)
        remaining_idx.remove(best)

    return [candidates[i] for i in selected_idx]


def strategy_authority_reranked(candidates: list, k: int = TOP_K,
                                  alpha: float = 0.7) -> list:
    """
    Rerank by combining similarity with authority weight.
    final_score = alpha * similarity + (1-alpha) * authority_weight
    """
    scored = []
    for c in candidates:
        auth_weight = get_authority_weight(c["source"])
        final_score = alpha * c["similarity"] + (1 - alpha) * auth_weight
        scored.append({**c, "final_score": final_score, "authority_weight": auth_weight})
    return sorted(scored, key=lambda c: -c["final_score"])[:k]


# ── outcome metrics ───────────────────────────────────────────────────────────

def compute_metrics(chunks: list) -> dict:
    """
    Compute the 4 outcome dimensions for a set of retrieved chunks.
    """
    if not chunks:
        return {"relevance": 0, "diversity": 0, "authority": 0, "dominance": 1}

    # 1. relevance — mean similarity
    relevance = float(np.mean([c["similarity"] for c in chunks]))

    # 2. diversity — unique sources / total returned
    sources = [c["source"] for c in chunks]
    diversity = len(set(sources)) / len(chunks)

    # 3. authority — mean authority weight
    authority = float(np.mean([get_authority_weight(c["source"]) for c in chunks]))

    # 4. dominance — max fraction from single source (lower = better)
    source_counts = defaultdict(int)
    for s in sources:
        source_counts[s] += 1
    dominance = max(source_counts.values()) / len(chunks)

    return {
        "relevance": round(relevance, 4),
        "diversity": round(diversity, 4),
        "authority": round(authority, 4),
        "dominance": round(dominance, 4),
    }


# ── main experiment runner ────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("STEK 2035 — Task 3: Retrieval Strategy Comparison")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {VECTOR_STORE_DIR}/")
    chunks, embeddings = load_vector_store()

    # fail fast with a clear message if the model doesn't match the
    # stored embedding dimension, instead of a confusing matmul error
    # deep inside the per-query loop
    model_dim = model.get_sentence_embedding_dimension()
    stored_dim = embeddings.shape[1]
    if model_dim != stored_dim:
        raise ValueError(
            f"Embedding dimension mismatch: EMBED_MODEL '{EMBED_MODEL}' "
            f"outputs {model_dim}-dim vectors, but embeddings.npy contains "
            f"{stored_dim}-dim vectors. Set EMBED_MODEL to whichever model "
            f"was actually used to build embeddings.npy — "
            f"multilingual-e5-base=768dim, multilingual-e5-large=1024dim, "
            f"bge-m3=1024dim."
        )
    print(f"  Dimension check OK: model={model_dim}-dim, stored={stored_dim}-dim")

    def embed_fn(texts):
        vecs = model.encode([f"passage: {t}" for t in texts], show_progress_bar=False)
        return np.asarray(vecs)

    strategies = {
        "standard_topk"     : lambda cands: strategy_standard_topk(cands),
        "max_1_per_source"  : lambda cands: strategy_max_n_per_source(cands, 1),
        "max_2_per_source"  : lambda cands: strategy_max_n_per_source(cands, 2),
        "max_3_per_source"  : lambda cands: strategy_max_n_per_source(cands, 3),
        "mmr"               : lambda cands: strategy_mmr(cands, embed_fn),
        "authority_reranked": lambda cands: strategy_authority_reranked(cands),
    }

    all_results = {name: [] for name in strategies}
    per_query_detail = []

    for qi, query in enumerate(TEST_QUERIES):
        print(f"\n[{qi+1}/{len(TEST_QUERIES)}] {query[:60]}...")

        q_embedding = model.encode(f"query: {query}")
        candidates = get_candidates(chunks, embeddings, q_embedding, n=CANDIDATE_K)

        query_result = {"query": query, "strategies": {}}

        for strat_name, strat_fn in strategies.items():
            chunks = strat_fn(candidates)
            metrics = compute_metrics(chunks)
            all_results[strat_name].append(metrics)

            query_result["strategies"][strat_name] = {
                "metrics": metrics,
                "sources_returned": [c["source"] for c in chunks],
            }

            print(f"    {strat_name:<20} "
                  f"rel={metrics['relevance']:.3f}  "
                  f"div={metrics['diversity']:.3f}  "
                  f"auth={metrics['authority']:.3f}  "
                  f"dom={metrics['dominance']:.3f}")

        per_query_detail.append(query_result)

    # ── aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (averaged across all queries)")
    print(f"{'='*70}")
    print(f"{'Strategy':<22} {'Relevance':>10} {'Diversity':>10} {'Authority':>10} {'Dominance':>10}")
    print("-" * 70)

    summary_rows = []
    for strat_name, results_list in all_results.items():
        avg = {
            k: round(float(np.mean([r[k] for r in results_list])), 4)
            for k in ["relevance", "diversity", "authority", "dominance"]
        }
        summary_rows.append({"strategy": strat_name, **avg})
        print(f"{strat_name:<22} {avg['relevance']:>10.4f} {avg['diversity']:>10.4f} "
              f"{avg['authority']:>10.4f} {avg['dominance']:>10.4f}")

    print("\nInterpretation guide:")
    print("  relevance  — higher is better (chunks match the query)")
    print("  diversity  — higher is better (chunks from different sources)")
    print("  authority  — higher is better (chunks from higher-authority docs)")
    print("  dominance  — LOWER is better (less flooding by one document)")

    # ── save outputs ──────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "test_queries": TEST_QUERIES,
            "per_query_results": per_query_detail,
            "aggregate_summary": summary_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Full results saved: {OUTPUT_JSON}")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "relevance", "diversity", "authority", "dominance"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"✅ Summary table saved: {OUTPUT_CSV}")

    return summary_rows


if __name__ == "__main__":
    run_experiment()