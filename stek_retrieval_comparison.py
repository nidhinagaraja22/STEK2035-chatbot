"""
STEK 2035 — Task 3: Retrieval Strategy Comparison
====================================================
Professor's Step 3 requirement: do not use a fixed "max 2 chunks
per source" rule without testing it. Compare 6 retrieval strategies
experimentally across 4 outcome dimensions.

Strategies tested:
  1. standard_topk         — baseline, plain cosine similarity top-5
  2. max_1_per_source       — diversity cap: 1 chunk per document max
  3. max_2_per_source       — diversity cap: 2 chunks per document max
  4. max_3_per_source       — diversity cap: 3 chunks per document max
  5. mmr                    — Maximal Marginal Relevance (relevance + diversity)
  6. authority_reranked     — reweight by document authority level (DAT)
  7. topic_diversified      — cap 2 chunks per LDA topic (thematic breadth,
                              independent of source document — a fully
                              source-diverse result can still be topic-narrow
                              if all documents discuss the same theme)
  8. combined_source_topic  — cap 2 per source AND 2 per topic simultaneously.
                              Neither cap alone guarantees the other (confirmed
                              empirically: topic-diverse top-5 was still 3/5
                              from one document) — this tests whether combining
                              them actually achieves both properties at once,
                              or whether the dual constraint costs relevance.

Outcome metrics (first 4 per professor's Step 3, defined relative to SOURCE
DOCUMENT; topic_coverage is an additional metric defined relative to LDA
TOPIC — the two are independent dimensions, not substitutes):
  1. relevance       — mean cosine similarity of returned chunks (higher = better)
  2. diversity        — unique source documents in top-5 / 5 (higher = better)
  3. authority         — mean authority weight of returned chunks (higher = better,
                         weight scale: L1=1.0, L2=0.85, L3=0.70, L4=0.55, L5=0.30)
  4. dominance         — max fraction from a single source in top-5 (LOWER = better,
                         this is the flooding indicator — Online-Beteiligung problem)
  5. topic_coverage    — unique LDA topics in top-5 / 5 (higher = better,
                         thematic breadth — orthogonal to source diversity)

Usage:
    python stek_retrieval_comparison.py

Requires:
    Corpus 2 vector store (deduped + metadata + authority already assigned):
      corpus/corpus_v2/corpus_v2_chunks.jsonl      — chunk text + metadata,
                                                       one JSON object per line
      corpus/corpus_v2/embeddings_v2_e5base.npy    — (n_chunks, 768) float array,
                                                       row i = corpus_v2_chunks.jsonl line i
      corpus/corpus_v2/meta_v2.json                 — corpus-level metadata (optional,
                                                       not required by this script)
      vector_store/lda_topics.json                  — per-chunk topic labels, k=8
                                                       (optional, not required here;
                                                       still valid for Corpus 2)

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
CORPUS_DIR       = Path("corpus/corpus_v2")               # Corpus 2: deduped + metadata + authority
CHUNKS_PATH      = CORPUS_DIR / "corpus_v2_chunks.jsonl"
EMB_PATH         = CORPUS_DIR / "embeddings_v2_e5base.npy"
META_PATH        = CORPUS_DIR / "meta_v2.json"
LDA_TOPICS_PATH  = VECTOR_STORE_DIR / "lda_topics.json"    # topic labels (k=8 ids still valid for Corpus 2)

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


# ── authority — read directly from chunk metadata, no guessing ───────────────
# corpus_v2_chunks.jsonl already stores authority_level per chunk (Task 2 /
# DAT work is already done upstream). Observed levels in the actual corpus:
# L1=116 (stek_a3_strategy), L2=481 (mro/nachhaltigkeitsbericht/2015 plan/
# web pdfs), L3=97 (website_html), L4=572 (online_beteiligung + event docs +
# arbeitstreffen). No L5 present — is_citizen_opinion (bool) is the separate
# flag that marks the 572 citizen-opinion chunks, overlapping with L4.
AUTHORITY_WEIGHT = {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.30}


def get_authority_weight(authority_level) -> float:
    return AUTHORITY_WEIGHT.get(int(authority_level), 0.70)


# ── vector store loading ──────────────────────────────────────────────────────

def load_vector_store():
    """
    Load the numpy-based vector store, split across two directories:
      CORPUS_DIR (corpus/corpus_v2/)  — Corpus 2: deduped + metadata + authority
        corpus_v2_chunks.jsonl        one JSON object per line. Confirmed fields:
                                       text, origin (raw filename/URL), document_id
                                       (clean per-document key, ~20 distinct values),
                                       authority_level (1-4, already assigned),
                                       is_citizen_opinion (bool), doc_type,
                                       topic / lda_topic / topics.
        embeddings_v2_e5base.npy      (n_chunks, 768) array, row i corresponds to
                                       corpus_v2_chunks.jsonl line i (same order —
                                       this alignment is assumed and not
                                       re-verified here).

    Note: "source" in the raw JSONL is a coarse type label (document /
    website_pdf / website_html) — NOT a per-document identifier. We use
    "document_id" instead, which correctly distinguishes all ~20 source
    documents.

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
            # document_id is the correct per-document identifier (~20
            # distinct values, confirmed via direct inspection). "source"
            # alone is a coarse type label (document/website_pdf/
            # website_html, only 3 values) and is NOT usable for
            # diversity/dominance grouping. "origin" is the raw filename/
            # URL, kept only for display.
            doc_id = obj.get("document_id") or obj.get("origin") or obj.get("source") or "unknown"
            text = obj.get("text") or obj.get("chunk") or ""
            authority_level = obj.get("authority_level", 3)  # fallback only if truly missing
            # dominant topic id — prefer lda_topic (explicit LDA output),
            # fall back to "topic" field, then -1 (unassigned) if neither present
            topic = obj.get("lda_topic")
            if topic is None:
                topic = obj.get("topic", -1)
            chunks.append({
                "text"            : text,
                "source"          : doc_id,               # used for diversity/dominance grouping
                "origin"          : obj.get("origin", ""), # raw filename, kept for display
                "authority_level" : authority_level,
                "is_citizen"      : obj.get("is_citizen_opinion", False),
                "topic"           : topic,                 # used for topic-diversity grouping
                "raw"             : obj,
            })

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
    n_docs = len(set(c["source"] for c in chunks))
    print(f"  Distinct source documents (via document_id): {n_docs}")
    return chunks, embeddings


def validate_metadata_health(chunks: list) -> None:
    """
    Sanity-check the loaded metadata BEFORE running any queries.

    This project has already hit three silent-failure bugs in this exact
    script, each of which produced plausible-looking but wrong CSV output
    with no error thrown:
      1. "source" field read as a coarse 3-value type category
         (document/website_pdf/website_html) instead of per-document id
         → diversity/dominance looked fine but meant almost nothing
      2. "authority_level" silently defaulting to 3 for every chunk
         because the field was missing from the file actually being read
         → authority_reranked was byte-identical to standard_topk
      3. CHUNKS_PATH pointing at a stale/wrong file entirely
         → both of the above at once

    This function checks for the same failure signatures on every run,
    so a future data/config change fails loudly here instead of producing
    a quietly-wrong aggregate table.
    """
    n = len(chunks)
    warnings = []

    # 1. source diversity — should reflect ~dozens of real documents,
    #    not a handful of coarse type categories
    n_sources = len(set(c["source"] for c in chunks))
    if n_sources <= 5:
        warnings.append(
            f"Only {n_sources} distinct 'source' values across {n} chunks. "
            f"This is the exact signature of reading a coarse type field "
            f"(e.g. 'document'/'website_pdf'/'website_html') instead of a "
            f"real per-document identifier — check that document_id (or "
            f"equivalent) is actually present in the loaded JSONL."
        )

    # 2. authority_level — should show real spread (e.g. L1-L4 in this
    #    corpus), not one value everywhere (the "silent fallback" bug)
    auth_values = [c["authority_level"] for c in chunks]
    n_auth_levels = len(set(auth_values))
    if n_auth_levels <= 1:
        only_value = auth_values[0] if auth_values else "?"
        warnings.append(
            f"All {n} chunks have the SAME authority_level ({only_value!r}). "
            f"This is the exact signature of the field being missing from "
            f"the source file and every chunk silently hitting the "
            f"fallback default — authority_reranked will be identical to "
            f"standard_topk if this is not fixed."
        )

    # 3. topic — should show multiple LDA topics, not all-missing (-1)
    topic_values = [c["topic"] for c in chunks]
    n_topics = len(set(t for t in topic_values if t != -1))
    n_missing_topic = sum(1 for t in topic_values if t == -1)
    if n_topics <= 1:
        warnings.append(
            f"Only {n_topics} distinct real topic value(s) found "
            f"({n_missing_topic}/{n} chunks have no topic at all). "
            f"topic_diversified and topic_coverage will be meaningless "
            f"until lda_topic/topic is actually present."
        )

    if warnings:
        print("\n" + "=" * 70)
        print("⚠️  METADATA HEALTH CHECK — POTENTIAL SILENT FAILURE DETECTED")
        print("=" * 70)
        for w in warnings:
            print(f"  • {w}")
        print("=" * 70)
        print("Results below may be misleading. Fix the data/config issue")
        print("above before trusting the aggregate table.\n")
    else:
        print(f"  Metadata health check OK: {n_sources} sources, "
              f"{n_auth_levels} authority levels, {n_topics} topics — "
              f"no known failure signatures detected.")


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
            "text"            : chunks[i]["text"],
            "source"          : chunks[i]["source"],
            "similarity"      : float(sims[i]),
            "authority_level" : chunks[i]["authority_level"],
            "is_citizen"      : chunks[i]["is_citizen"],
            "topic"           : chunks[i]["topic"],
        })
    return candidates


def strategy_standard_topk(candidates: list, k: int = TOP_K) -> list:
    """Baseline — plain top-k by similarity, no diversity/authority logic."""
    return sorted(candidates, key=lambda c: -c["similarity"])[:k]


def strategy_max_n_per_group(candidates: list, group_key_fn, max_per_group: int,
                               k: int = TOP_K) -> list:
    """
    Generic cap-per-group selector, preserving similarity order.
    group_key_fn extracts the grouping key from a candidate dict —
    e.g. c["source"] for document-level diversity, c["topic"] for
    topic-level diversity. This is the shared logic behind both
    strategy_max_n_per_source and strategy_max_n_per_topic below.
    """
    ranked = sorted(candidates, key=lambda c: -c["similarity"])
    selected = []
    group_count = defaultdict(int)
    for c in ranked:
        key = group_key_fn(c)
        if group_count[key] < max_per_group:
            selected.append(c)
            group_count[key] += 1
        if len(selected) == k:
            break
    return selected


def strategy_max_n_per_source(candidates: list, max_per_source: int, k: int = TOP_K) -> list:
    """Cap chunks per SOURCE DOCUMENT — fixes one document flooding results
    (e.g. Online-Beteiligung dominating a housing question)."""
    return strategy_max_n_per_group(candidates, lambda c: c["source"], max_per_source, k)


def strategy_max_n_per_topic(candidates: list, max_per_topic: int, k: int = TOP_K) -> list:
    """Cap chunks per LDA TOPIC — fixes the answer only covering one theme
    (e.g. all chunks about Wohnen when the question also touches Klimaschutz),
    regardless of which document they came from."""
    return strategy_max_n_per_group(candidates, lambda c: c["topic"], max_per_topic, k)


def strategy_combined_source_topic(candidates: list, max_per_source: int = 2,
                                     max_per_topic: int = 2, k: int = TOP_K) -> list:
    """
    Enforce BOTH source-document diversity AND topic diversity in a single
    selection pass — neither cap alone guarantees the other (confirmed
    empirically: topic-diverse results can still be 100% from one document,
    e.g. Online-Beteiligung spans all 8 topics on its own).

    Three-stage greedy fill, preserving similarity order at each stage:
      Stage 1 — strict: candidate must satisfy BOTH caps to be selected
      Stage 2 — relaxed: if Stage 1 couldn't fill k slots (both caps together
                can be too strict for a small candidate pool), fill remaining
                slots respecting ONLY the source cap — source diversity is
                the professor's explicit Step 3 requirement, so it takes
                priority over the topic extension when a tradeoff is forced
      Stage 3 — last resort: if still short, fill remaining slots by plain
                similarity ignoring both caps, so this strategy always
                returns the same number of chunks as the others (keeps
                metric comparisons fair — an under-filled result would
                distort diversity/dominance denominators)
    """
    ranked = sorted(candidates, key=lambda c: -c["similarity"])
    selected = []
    selected_ids = set()
    source_count = defaultdict(int)
    topic_count = defaultdict(int)

    # stage 1 — strict: both caps enforced
    for c in ranked:
        if len(selected) == k:
            break
        if (source_count[c["source"]] < max_per_source and
                topic_count[c["topic"]] < max_per_topic):
            selected.append(c)
            selected_ids.add(id(c))
            source_count[c["source"]] += 1
            topic_count[c["topic"]] += 1

    # stage 2 — relax topic cap, keep source cap
    if len(selected) < k:
        for c in ranked:
            if len(selected) == k:
                break
            if id(c) in selected_ids:
                continue
            if source_count[c["source"]] < max_per_source:
                selected.append(c)
                selected_ids.add(id(c))
                source_count[c["source"]] += 1

    # stage 3 — last resort, ignore both caps, just fill by similarity
    if len(selected) < k:
        for c in ranked:
            if len(selected) == k:
                break
            if id(c) in selected_ids:
                continue
            selected.append(c)
            selected_ids.add(id(c))

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
        auth_weight = get_authority_weight(c["authority_level"])
        final_score = alpha * c["similarity"] + (1 - alpha) * auth_weight
        scored.append({**c, "final_score": final_score, "authority_weight": auth_weight})
    return sorted(scored, key=lambda c: -c["final_score"])[:k]


# ── outcome metrics ───────────────────────────────────────────────────────────

def compute_metrics(chunks: list) -> dict:
    """
    Compute the 4 outcome dimensions required by the professor's Step 3
    (relevance, diversity, authority, dominance — all defined relative to
    SOURCE DOCUMENT), plus one additional metric (topic_coverage) measuring
    thematic breadth independently of which document each chunk came from.
    """
    if not chunks:
        return {"relevance": 0, "diversity": 0, "authority": 0, "dominance": 1, "topic_coverage": 0}

    # 1. relevance — mean similarity
    relevance = float(np.mean([c["similarity"] for c in chunks]))

    # 2. diversity — unique SOURCE DOCUMENTS / total returned (professor's Step 3 metric)
    sources = [c["source"] for c in chunks]
    diversity = len(set(sources)) / len(chunks)

    # 3. authority — mean authority weight
    authority = float(np.mean([get_authority_weight(c["authority_level"]) for c in chunks]))

    # 4. dominance — max fraction from single source (lower = better)
    source_counts = defaultdict(int)
    for s in sources:
        source_counts[s] += 1
    dominance = max(source_counts.values()) / len(chunks)

    # 5. topic_coverage — unique LDA TOPICS / total returned (additional metric,
    #    not part of the professor's original 4 — measures thematic breadth,
    #    independent of source document diversity)
    topics = [c["topic"] for c in chunks]
    topic_coverage = len(set(topics)) / len(chunks)

    return {
        "relevance": round(relevance, 4),
        "diversity": round(diversity, 4),
        "authority": round(authority, 4),
        "dominance": round(dominance, 4),
        "topic_coverage": round(topic_coverage, 4),
    }


# ── main experiment runner ────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("STEK 2035 — Task 3: Retrieval Strategy Comparison")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading vector store from: {CORPUS_DIR}/")
    chunks, embeddings = load_vector_store()
    validate_metadata_health(chunks)

    # fail fast with a clear message if the model doesn't match the
    # stored embedding dimension, instead of a confusing matmul error
    # deep inside the per-query loop
    model_dim = model.get_embedding_dimension()
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
        "standard_topk"        : lambda cands: strategy_standard_topk(cands),
        "max_1_per_source"     : lambda cands: strategy_max_n_per_source(cands, 1),
        "max_2_per_source"     : lambda cands: strategy_max_n_per_source(cands, 2),
        "max_3_per_source"     : lambda cands: strategy_max_n_per_source(cands, 3),
        "mmr"                  : lambda cands: strategy_mmr(cands, embed_fn),
        "authority_reranked"   : lambda cands: strategy_authority_reranked(cands),
        "topic_diversified"    : lambda cands: strategy_max_n_per_topic(cands, 2),
        "combined_source_topic": lambda cands: strategy_combined_source_topic(cands, 2, 2),
    }

    all_results = {name: [] for name in strategies}
    per_query_detail = []

    for qi, query in enumerate(TEST_QUERIES):
        print(f"\n[{qi+1}/{len(TEST_QUERIES)}] {query[:60]}...")

        q_embedding = model.encode(f"query: {query}")
        candidates = get_candidates(chunks, embeddings, q_embedding, n=CANDIDATE_K)

        query_result = {"query": query, "strategies": {}}

        for strat_name, strat_fn in strategies.items():
            retrieved = strat_fn(candidates)
            metrics = compute_metrics(retrieved)
            all_results[strat_name].append(metrics)

            query_result["strategies"][strat_name] = {
                "metrics": metrics,
                "sources_returned": [c["source"] for c in retrieved],
            }

            print(f"    {strat_name:<20} "
                  f"rel={metrics['relevance']:.3f}  "
                  f"div={metrics['diversity']:.3f}  "
                  f"auth={metrics['authority']:.3f}  "
                  f"dom={metrics['dominance']:.3f}  "
                  f"topic={metrics['topic_coverage']:.3f}")

        per_query_detail.append(query_result)

    # ── aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (averaged across all queries)")
    print(f"{'='*70}")
    print(f"{'Strategy':<22} {'Relevance':>10} {'Diversity':>10} {'Authority':>10} "
          f"{'Dominance':>10} {'TopicCov':>10}")
    print("-" * 80)

    summary_rows = []
    for strat_name, results_list in all_results.items():
        avg = {
            k: round(float(np.mean([r[k] for r in results_list])), 4)
            for k in ["relevance", "diversity", "authority", "dominance", "topic_coverage"]
        }
        summary_rows.append({"strategy": strat_name, **avg})
        print(f"{strat_name:<22} {avg['relevance']:>10.4f} {avg['diversity']:>10.4f} "
              f"{avg['authority']:>10.4f} {avg['dominance']:>10.4f} {avg['topic_coverage']:>10.4f}")

    print("\nInterpretation guide:")
    print("  relevance      — higher is better (chunks match the query)")
    print("  diversity      — higher is better (chunks from different SOURCE DOCUMENTS)")
    print("  authority      — higher is better (chunks from higher-authority docs)")
    print("  dominance      — LOWER is better (less flooding by one document)")
    print("  topic_coverage — higher is better (chunks span different LDA TOPICS —")
    print("                   independent of which document they came from;")
    print("                   e.g. a fully source-diverse top-5 can still be")
    print("                   topic-narrow if all 5 documents happen to discuss")
    print("                   the same theme)")

    # ── save outputs ──────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "test_queries": TEST_QUERIES,
            "per_query_results": per_query_detail,
            "aggregate_summary": summary_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Full results saved: {OUTPUT_JSON}")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "relevance", "diversity", "authority", "dominance", "topic_coverage"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"✅ Summary table saved: {OUTPUT_CSV}")

    return summary_rows


if __name__ == "__main__":
    run_experiment()