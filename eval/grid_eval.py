# -*- coding: utf-8 -*-
"""
STEK 2035 -- Embedding x Retrieval evaluation GRID
===================================================
Scores every (embedding model x retrieval method) "flow" on the SAME corpus
(Corpus 2) and the SAME gold set, producing a factual results table.

Metrics per flow, at k = 1, 3, 5, 10:
    Precision@k, Recall@k, MRR, nDCG@k, source diversity@k, authority correctness@k

Ground truth:
    PROPER  -> gold question has "relevant_chunk_ids" (from human annotation).
               Precision/Recall/nDCG are all well-defined.
    PROXY   -> gold question only has keyword rules (must_contain/any_of).
               Recall is left NaN (total-relevant unknown); use only to smoke-test.

Prerequisites:
    - corpus/corpus_v2/corpus_v2_chunks.jsonl
    - one embeddings_*.npy per embedding model (same row order as the corpus),
      generated with the CORRECT per-model prefix (see EMBED_MODELS).
    - pip install sentence-transformers rank-bm25

Run:
    python eval/grid_eval.py
Output:
    eval/grid_results.csv   (one row per flow x k)
"""
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE = Path.cwd()
for cand in [BASE, *BASE.parents]:
    if (cand / "corpus" / "corpus_v2" / "corpus_v2_chunks.jsonl").exists():
        BASE = cand
        break

CORPUS = BASE / "corpus/corpus_v2/corpus_v2_chunks.jsonl"
GOLD = BASE / "eval/gold_questions.json"
OUT = BASE / "eval/grid_results.csv"
K_VALUES = [1, 3, 5, 10]
DEPTH = 30                       # candidate depth for hybrid fusion / reranking
RERANKER = "BAAI/bge-reranker-v2-m3"

# Each embedding model: HF name, its query prefix (e5 uses "query: ", others ""),
# and the pre-generated embeddings file. Only models whose .npy exists are run.
EMBED_MODELS = {
    "e5-small":  {"hf": "intfloat/multilingual-e5-small",  "prefix": "query: ", "emb": "corpus/corpus_v2/embeddings_v2_e5small.npy"},
    "e5-base":   {"hf": "intfloat/multilingual-e5-base",   "prefix": "query: ", "emb": "corpus/corpus_v2/embeddings_v2_e5base.npy"},
    "e5-large":  {"hf": "intfloat/multilingual-e5-large",  "prefix": "query: ", "emb": "corpus/corpus_v2/embeddings_v2_e5large.npy"},
    "bge-m3":    {"hf": "BAAI/bge-m3",                      "prefix": "",        "emb": "corpus/corpus_v2/embeddings_v2_bge.npy"},
    "gte":       {"hf": "Alibaba-NLP/gte-multilingual-base","prefix": "",       "emb": "corpus/corpus_v2/embeddings_v2_gte.npy"},
    "mpnet":     {"hf": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "prefix": "", "emb": "corpus/corpus_v2/embeddings_v2_mpnet.npy"},
}
# retrieval methods that depend on the embedding model
DENSE_METHODS = ["dense", "hybrid", "rerank"]

# --------------------------------------------------------------------------- #
# Load corpus (chunk uid / text / document / authority, aligned by row)
# --------------------------------------------------------------------------- #
chunks = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines()]
uids = [c["chunk_uid"] for c in chunks]
texts = [c["text"] for c in chunks]
docids = [c.get("document_id") for c in chunks]
authlvl = [c.get("authority_level") for c in chunks]
gold = json.loads(GOLD.read_text(encoding="utf-8"))
print(f"Corpus: {len(chunks)} chunks | gold: {len(gold)} questions")


# --------------------------------------------------------------------------- #
# Relevance ground truth (proper vs proxy)
# --------------------------------------------------------------------------- #
def gold_relevant(g):
    """Return (relevant_uid_set or None). None => proxy keyword mode."""
    rc = g.get("relevant_chunk_ids")
    return set(rc) if rc else None


def is_rel_text(text, g):
    t = text.lower()
    if not all(w.lower() in t for w in g.get("must_contain", [])):
        return False
    ao = g.get("any_of", [])
    if ao and not any(w.lower() in t for w in ao):
        return False
    return True


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score_ranked(ranked_idx, g):
    """ranked_idx: retrieved chunk indices (best first). Returns metric dict."""
    R = gold_relevant(g)
    if R is not None:
        rel = [1 if uids[i] in R else 0 for i in ranked_idx]
        total_rel = len(R)
    else:                                   # proxy
        rel = [1 if is_rel_text(texts[i], g) else 0 for i in ranked_idx]
        total_rel = None
    exp_auth = g.get("expected_authority_level")

    out = {}
    first = next((n for n, r in enumerate(rel, 1) if r), None)
    out["mrr"] = 1.0 / first if first else 0.0
    for k in K_VALUES:
        relk = rel[:k]
        idxk = ranked_idx[:k]
        out[f"P@{k}"] = sum(relk) / k
        out[f"R@{k}"] = (sum(relk) / total_rel) if total_rel else float("nan")
        idc = _dcg(sorted(rel, reverse=True)[:k])
        out[f"nDCG@{k}"] = (_dcg(relk) / idc) if idc > 0 else 0.0
        out[f"div@{k}"] = len(set(docids[i] for i in idxk)) / k
        out[f"auth@{k}"] = (sum(1 for i in idxk if authlvl[i] == exp_auth) / k
                            if exp_auth is not None else float("nan"))
    return out


def mean_over_gold(rank_fn):
    """Average each metric across all gold questions for one flow."""
    acc, n = {}, 0
    for g in gold:
        ranked = rank_fn(g["question"], DEPTH)
        m = score_ranked(ranked, g)
        for key, val in m.items():
            if not (isinstance(val, float) and math.isnan(val)):
                acc.setdefault(key, []).append(val)
        n += 1
    return {key: (sum(v) / len(v) if v else float("nan")) for key, v in acc.items()}


# --------------------------------------------------------------------------- #
# Retrieval methods
# --------------------------------------------------------------------------- #
def build_bm25():
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", s.lower())
    return BM25Okapi([tok(t) for t in texts]), tok


def make_rankers(model_key):
    """Return {method: rank_fn} for one embedding model."""
    from sentence_transformers import SentenceTransformer
    cfg = EMBED_MODELS[model_key]
    emb = np.load(BASE / cfg["emb"])
    assert emb.shape[0] == len(chunks), f"{model_key}: emb rows != chunks"
    st = SentenceTransformer(cfg["hf"])
    prefix = cfg["prefix"]

    def dense_scores(q):
        v = st.encode(prefix + q, normalize_embeddings=True).astype("float32")
        return emb @ v

    def dense(q, depth):
        return list(np.argsort(-dense_scores(q))[:depth])

    def hybrid(q, depth, cap=DEPTH, rrf=60):
        d = list(np.argsort(-dense_scores(q))[:cap])
        b = list(np.argsort(-bm25.get_scores(_tok(q)))[:cap])
        fused = {}
        for r, i in enumerate(d):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (rrf + r)
        for r, i in enumerate(b):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (rrf + r)
        return sorted(fused, key=fused.get, reverse=True)[:depth]

    _rr = {}
    def rerank(q, depth):
        if "m" not in _rr:
            from sentence_transformers import CrossEncoder
            _rr["m"] = CrossEncoder(RERANKER)
        cand = dense(q, DEPTH)
        sc = _rr["m"].predict([[q, texts[i]] for i in cand])
        return [cand[j] for j in np.argsort(-np.asarray(sc))][:depth]

    return {"dense": dense, "hybrid": hybrid, "rerank": rerank}


# --------------------------------------------------------------------------- #
# Run the grid
# --------------------------------------------------------------------------- #
bm25, _tok = build_bm25()
rows = []

# BM25 is embedding-independent -> one flow
def bm25_rank(q, depth):
    return list(np.argsort(-bm25.get_scores(_tok(q)))[:depth])

rows.append({"flow": "bm25", "embedding": "-", "retrieval": "bm25", **mean_over_gold(bm25_rank)})
print("done  bm25")

for mkey, cfg in EMBED_MODELS.items():
    if not (BASE / cfg["emb"]).exists():
        print(f"skip  {mkey:9} (no {cfg['emb']})")
        continue
    rankers = make_rankers(mkey)
    for method in DENSE_METHODS:
        m = mean_over_gold(rankers[method])
        rows.append({"flow": f"{mkey}+{method}", "embedding": mkey, "retrieval": method, **m})
        print(f"done  {mkey:9} {method:7} nDCG@5={m.get('nDCG@5', float('nan')):.3f} MRR={m.get('mrr', float('nan')):.3f}")

# --------------------------------------------------------------------------- #
# Write factual results table
# --------------------------------------------------------------------------- #
cols = (["flow", "embedding", "retrieval", "mrr"]
        + [f"{m}@{k}" for k in K_VALUES for m in ["P", "R", "nDCG", "div", "auth"]])
with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
print(f"\nWrote {len(rows)} flows -> {OUT}")
