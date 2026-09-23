"""
STEK 2035 RAG Backend — Integrated with Hybrid Relevance Gate + Authority Cascade
====================================================================================
Adds two stages to the original pipeline, adapted for THIS server's numpy-based
storage (chunks.jsonl + embeddings.npy) — hybrid_gate.py was originally built
around ChromaDB, so the gate logic below is a direct port of its real, verified
formulas onto this server's actual data structures, not a different mechanism.

New pipeline:
  1. Hybrid Relevance Gate  (OUT_OF_DOMAIN / VAGUE / IN_DOMAIN)
  2. Semantic Retrieval + Authority-Aware Cascade  (replaces plain top-k)
  3. Context Construction + LLM Generation  (unchanged)

Run:
    uvicorn rag_server:app --reload --port 8000
"""

import json
import re
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = Path("vector_store")
CORPUS_DIR = Path("corpus/corpus_v2")
CHUNKS_PATH = CORPUS_DIR / "corpus_v2_chunks.jsonl"
EMB_PATH = CORPUS_DIR / "embeddings_v2_e5base.npy"
META_PATH = CORPUS_DIR / "meta_v2.json"
LDA_TOPICS_PATH = VECTOR_STORE_DIR / "lda_topics.json"

TOP_K = 5
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:32b"

# --- Hybrid gate config (ported from hybrid_gate.py's verified, fixed values) ---
ALPHA = 0.6
REJECT_THRESHOLD = 0.53          # fit to hybrid_gate.py's own 20-question test --
                                    # re-measure against YOUR real traffic before trusting in production
BM25_FIXED_CAP = 11.0
VAGUE_MIN_WORDS = 8
VAGUE_TOPN = 5
VAGUE_SPREAD_THRESHOLD = 0.03
VAGUE_SCORE_MIN = 0.25
VAGUE_SCORE_MAX = 0.60
MIN_MATCHED_TERMS = 2
MIN_QUERY_TERMS_FOR_GUARD = 2

# --- Request-type classification (ported from stek_cascading_retrieval.py --
# this was MISSING from the original integration; without it, every question,
# including explicit citizen-opinion requests, went through the same
# official-first threshold cascade) ---
CITIZEN_REQUEST_KEYWORDS = [
    "wünschen sich bürger", "wünscht sich", "bürger fordern", "fordern bürger",
    "was denken bürger", "bürgermeinung", "was wollen bürger",
    "was war der zweck", "wie wurden bürgerinnen", "räume für kulturelle nutzung fehlten",
]

COMPARISON_KEYWORDS = [
    "im vergleich", "verglichen mit", "unterschied zwischen",
    "bürgerwünsche und offizielle", "einerseits", "andererseits",
]


def classify_request_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in COMPARISON_KEYWORDS):
        return "comparison"
    if any(kw in q for kw in CITIZEN_REQUEST_KEYWORDS):
        return "citizen_explicit"
    return "default_cascade"


# --- Authority cascade config ---
AUTHORITY_THRESHOLD = 0.35   # ⚠️ see the note in cascading_retrieve() below —
                                # this is well below the 0.75-0.95 range real
                                # cosine similarity showed throughout this
                                # project's own testing on this corpus
CITATION_PREFIX = {
    1: "Laut dem offiziellen STEK 2035",
    2: "Laut dem städtischen Bericht",
    3: "Laut offiziellen städtischen Informationen",
    4: "Laut den Ergebnissen eines Stakeholder-Workshops",
    5: "Ein einzelner Bürger äußerte in der Online-Beteiligung den Wunsch",
}

# ---------------------------------------------------------------------------
# German stopwords for BM25 tokenization side of the gate (query-side only --
# chunk-side bm25_tokens are assumed pre-computed and stored per chunk)
# ---------------------------------------------------------------------------
GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einen",
    "und", "oder", "aber", "wie", "was", "wo", "wann", "wer", "welche",
    "ist", "sind", "war", "waren", "wird", "werden", "hat", "haben",
    "soll", "sollen", "will", "wollen", "kann", "können", "muss", "müssen",
    "für", "mit", "bei", "bis", "durch", "gegen", "ohne", "um", "auf", "aus",
    "in", "im", "an", "am", "zu", "zum", "zur", "von", "vom", "nach", "über",
    "sich", "es", "man", "ich", "du", "er", "sie", "wir", "ihr",
    "nicht", "kein", "keine", "auch", "noch", "schon", "nur",
}


def restore_umlauts(text: str) -> str:
    """ASCII-typed German (fuer->für) doesn't match correctly-spelled corpus
    tokens without this -- ported directly from hybrid_gate.py's verified fix."""
    for ascii_form, umlaut in [("ue", "ü"), ("oe", "ö"), ("ae", "ä")]:
        text = text.replace(ascii_form, umlaut)
        text = text.replace(ascii_form.capitalize(), umlaut.upper())
    text = re.sub(r"strasse", "straße", text, flags=re.IGNORECASE)
    return text


def simple_tokenize(text: str) -> list:
    """Query-side tokenization. NOTE: replace this with your actual
    lemmatize() function if you have one (e.g. spaCy-based, matching however
    your stored bm25_tokens were generated) -- using a different tokenizer
    for queries than for chunks will silently produce poor BM25 matches."""
    text = restore_umlauts(text)
    text = re.sub(r'[?!.,;:]', '', text.lower())
    return [w for w in text.split() if w not in GERMAN_STOPWORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# Load vector store once at startup
# ---------------------------------------------------------------------------
print("Loading vector store...")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
EMB_MODEL_NAME = meta["model"]

chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
embeddings = np.load(EMB_PATH)  # shape (N, 768), assumed L2-normalized already

print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")

# CRITICAL CHECK: authority_level must exist on every chunk for Stage 2's
# cascade to work at all. If missing, the cascade below falls back to
# treating everything as tier 3 (mid-authority) -- retrieval still runs, but
# WITHOUT any real authority protection. Fix your ingestion pipeline to add
# this field before relying on the cascade in production.
missing_authority = sum(1 for c in chunks if "authority_level" not in c)
if missing_authority > 0:
    print(f"WARNING: {missing_authority} of {len(chunks)} chunks have NO "
          f"authority_level field -- the authority cascade cannot protect these. "
          f"Add this field to your ingestion pipeline before trusting Stage 2 in production.")

authority_levels = np.array([c.get("authority_level", 3) for c in chunks])
official_idx = np.where(authority_levels <= 3)[0]
l4_idx = np.where(authority_levels == 4)[0]
l5_idx = np.where(authority_levels == 5)[0]

# CRITICAL CHECK: bm25_tokens must exist for the hybrid gate's BM25 side.
# You confirmed you have this field -- verify the KEY NAME matches exactly.
missing_bm25 = sum(1 for c in chunks if "bm25_tokens" not in c)
if missing_bm25 > 0:
    print(f"WARNING: {missing_bm25} of {len(chunks)} chunks have NO "
          f"bm25_tokens field -- check the actual key name in your chunks.jsonl "
          f"and update the loading code below if it differs.")

chunk_token_sets = [set(c.get("bm25_tokens", [])) for c in chunks]
tokenized_corpus = [c.get("bm25_tokens", []) for c in chunks]
bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus and any(tokenized_corpus) else None

lda_topics = json.loads(LDA_TOPICS_PATH.read_text(encoding="utf-8"))
topic_chunk_indices: dict = {t["id"]: [] for t in lda_topics}
for i, c in enumerate(chunks):
    topic_chunk_indices.setdefault(c.get("lda_topic", -1), []).append(i)
print(f"Loaded {len(lda_topics)} LDA topics")

# --- Anisotropy correction (Stage 1's mean-centering, ported from hybrid_gate.py) ---
mean_embedding = embeddings.mean(axis=0)
centered_embeddings = embeddings - mean_embedding
_norms = np.linalg.norm(centered_embeddings, axis=1, keepdims=True)
_norms[_norms == 0] = 1e-8
centered_embeddings = centered_embeddings / _norms

print(f"Loading embedding model: {EMB_MODEL_NAME} (first run downloads ~1GB)...")
embed_model = SentenceTransformer(EMB_MODEL_NAME)
print("Ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="STEK 2035 RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None
    topic: int | None = None


class Topic(BaseModel):
    id: int
    label: str
    top_words: list[str]
    chunk_count: int


class Source(BaseModel):
    origin: str
    chunk_id: int
    text: str
    score: float
    authority_level: int
    citation: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]
    verdict: str
    tier_used: str | None = None


# ---------------------------------------------------------------------------
# STAGE 1 -- Hybrid Relevance Gate
# ---------------------------------------------------------------------------
def hybrid_gate_check(question: str, debug_top_n: int = 5) -> dict:
    if bm25_index is None:
        bm25_raw = np.zeros(len(chunks))
        bm25_scaled = np.zeros(len(chunks))
    else:
        query_tokens = set(simple_tokenize(question))
        bm25_raw = np.array(bm25_index.get_scores(list(query_tokens)))
        bm25_scaled = np.clip(bm25_raw / BM25_FIXED_CAP, 0.0, 1.0)

        if len(query_tokens) >= MIN_QUERY_TERMS_FOR_GUARD:
            matched_counts = np.array([len(query_tokens & s) for s in chunk_token_sets])
            bm25_scaled = np.where(matched_counts >= MIN_MATCHED_TERMS, bm25_scaled, 0.0)

    query_emb = embed_model.encode(f"query: {question}", normalize_embeddings=True).astype("float32")
    q_centered = query_emb - mean_embedding
    q_centered = q_centered / (np.linalg.norm(q_centered) + 1e-8)
    cos_scaled = np.clip(centered_embeddings @ q_centered, 0.0, 1.0)

    hybrid = ALPHA * bm25_scaled + (1 - ALPHA) * cos_scaled
    top_score = float(hybrid.max())
    top_n = np.sort(hybrid)[-VAGUE_TOPN:]
    spread = float(top_n.max() - top_n.min())
    word_count = len(question.split())

    is_flat_and_short = word_count < VAGUE_MIN_WORDS and spread < VAGUE_SPREAD_THRESHOLD
    if is_flat_and_short and VAGUE_SCORE_MIN <= top_score < VAGUE_SCORE_MAX:
        verdict = "VAGUE"
    elif top_score < REJECT_THRESHOLD:
        verdict = "OUT_OF_DOMAIN"
    else:
        verdict = "IN_DOMAIN"

    # diagnostic breakdown: shows WHY the verdict happened, per top chunk --
    # is bm25 near-zero (no keyword match at all) or cos_scaled low too
    # (nothing topically similar either)?
    ranked_idx = np.argsort(hybrid)[::-1][:debug_top_n]
    top_chunks_debug = [{
        "rank": rank + 1,
        "hybrid_score": round(float(hybrid[idx]), 4),
        "bm25_raw": round(float(bm25_raw[idx]), 4),
        "bm25_scaled": round(float(bm25_scaled[idx]), 4),
        "cos_scaled": round(float(cos_scaled[idx]), 4),
        "origin": chunks[idx].get("origin", "?"),
        "authority_level": chunks[idx].get("authority_level", "?"),
        "text_preview": chunks[idx].get("text", "")[:120],
    } for rank, idx in enumerate(ranked_idx)]

    return {"verdict": verdict, "top_score": top_score, "spread": spread, "top_chunks": top_chunks_debug}


# ---------------------------------------------------------------------------
# STAGE 2 -- Semantic Retrieval + Authority-Aware Cascade (replaces plain top-k)
# ---------------------------------------------------------------------------
def search_tier(sims: np.ndarray, tier_idx: np.ndarray, k: int):
    if len(tier_idx) == 0:
        return np.array([], dtype=int), -1.0
    tier_sims = sims[tier_idx]
    top_k_in_tier = np.argsort(-tier_sims)[:k]
    result_idx = tier_idx[top_k_in_tier]
    best_score = float(tier_sims.max())
    return result_idx, best_score


def cascading_retrieve(query: str, k: int, topic: int | None = None):
    query_vec = embed_model.encode(f"query: {query}", normalize_embeddings=True).astype("float32")
    candidate_idx = np.array(topic_chunk_indices.get(topic, [])) if topic is not None else np.arange(len(chunks))
    sims_full = embeddings @ query_vec

    request_type = classify_request_type(query)

    if request_type == "citizen_explicit":
        # explicit citizen-opinion request — search L4+L5 directly,
        # skip official tier and the threshold check entirely
        citizen_idx = np.union1d(l4_idx, l5_idx)
        citizen_in_candidates = np.intersect1d(citizen_idx, candidate_idx)
        result_idx, _ = search_tier(sims_full, citizen_in_candidates, k)
        tier_used = "citizen_explicit"

    elif request_type == "comparison":
        # guaranteed slots from both official and citizen tiers
        off_in_candidates = np.intersect1d(official_idx, candidate_idx)
        citizen_idx = np.union1d(l4_idx, l5_idx)
        citizen_in_candidates = np.intersect1d(citizen_idx, candidate_idx)
        off_k = max(1, k - 2)
        off_result, _ = search_tier(sims_full, off_in_candidates, off_k)
        citizen_result, _ = search_tier(sims_full, citizen_in_candidates, k - off_k)
        result_idx = np.concatenate([off_result, citizen_result])
        tier_used = "comparison"

    else:  # default_cascade
        off_in_candidates = np.intersect1d(official_idx, candidate_idx)
        result_idx, best = search_tier(sims_full, off_in_candidates, k)
        tier_used = "official"
        if best < AUTHORITY_THRESHOLD:
            l4_in_candidates = np.intersect1d(l4_idx, candidate_idx)
            result_idx, best = search_tier(sims_full, l4_in_candidates, k)
            tier_used = "L4_workshop"
            if best < AUTHORITY_THRESHOLD:
                l5_in_candidates = np.intersect1d(l5_idx, candidate_idx)
                result_idx, best = search_tier(sims_full, l5_in_candidates, k)
                tier_used = "L5_citizen"

    results = []
    for i in result_idx:
        i = int(i)
        c = chunks[i]
        level = c.get("authority_level", 3)
        results.append({
            "origin": c.get("origin", "unknown"),
            "chunk_id": c.get("chunk_id", -1),
            "text": c.get("text", ""),
            "score": float(sims_full[i]),
            "authority_level": level,
            "citation": CITATION_PREFIX.get(level, "Quelle unklar"),
        })
    return results, tier_used


def build_prompt(question: str, contexts: list) -> str:
    context_block = "\n\n".join(
        f"[{c['citation']} -- {c['origin']}]\n{c['text']}" for c in contexts
    )
    return f"""Du bist ein hilfreicher Assistent für das Stadtentwicklungskonzept Heidelberg 2035 (STEK 2035).
Beantworte die folgende Frage NUR auf Basis der bereitgestellten Kontextauszüge.
Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich.
Antworte auf Deutsch, klar und präzise.

Kontext:
{context_block}

Frage: {question}

Antwort:"""


def ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks), "ollama_model": OLLAMA_MODEL,
            "bm25_available": bm25_index is not None,
            "chunks_missing_authority_level": int(missing_authority)}


@app.get("/debug/gate")
def debug_gate(question: str):
    """Diagnostic endpoint — shows the full score breakdown behind a
    verdict, without running retrieval or generation. Use this whenever
    a verdict looks surprising (e.g. OUT_OF_DOMAIN for a question that
    seems in-domain)."""
    return hybrid_gate_check(question)


@app.get("/topics", response_model=list[Topic])
def topics():
    return lda_topics


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    k = req.top_k or TOP_K

    gate_result = hybrid_gate_check(req.message)

    if gate_result["verdict"] == "OUT_OF_DOMAIN":
        return ChatResponse(
            reply="Diese Frage liegt außerhalb des STEK-2035-Korpus für Heidelberg. "
                  "Ich kann nur zu städtischen Planungsthemen für Heidelberg antworten.",
            sources=[], verdict="OUT_OF_DOMAIN", tier_used=None,
        )

    if gate_result["verdict"] == "VAGUE":
        return ChatResponse(
            reply="Ihre Frage ist sehr allgemein. Können Sie präzisieren, welches Thema "
                  "Sie interessiert -- z. B. Wohnen, Mobilität, Umwelt, Wirtschaft oder Soziales?",
            sources=[], verdict="VAGUE", tier_used=None,
        )

    contexts, tier_used = cascading_retrieve(req.message, k, topic=req.topic)
    prompt = build_prompt(req.message, contexts)
    reply = ask_ollama(prompt)
    return ChatResponse(
        reply=reply,
        sources=[Source(**c) for c in contexts],
        verdict="IN_DOMAIN",
        tier_used=tier_used,
    )