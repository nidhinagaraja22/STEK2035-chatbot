"""
STEK 2035 RAG Backend
----------------------
Loads the pre-built vector store (chunks.jsonl + embeddings.npy),
retrieves the most relevant chunks for a user question, and asks a
local Ollama model to answer using that context.

Run:
    uvicorn rag_server:app --reload --port 8000

Requires Ollama running locally with a model pulled, e.g.:
    ollama pull llama3.1
    ollama serve   (usually already running as a background service)
"""

import json
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config — adjust these paths / values as needed
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = Path("vector_store")
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.jsonl"
EMB_PATH = VECTOR_STORE_DIR / "embeddings.npy"
META_PATH = VECTOR_STORE_DIR / "meta.json"
LDA_TOPICS_PATH = VECTOR_STORE_DIR / "lda_topics.json"

TOP_K = 5
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:32b"  # change to whatever model you've pulled, e.g. "mistral", "qwen2.5"

# ---------------------------------------------------------------------------
# Load vector store once at startup
# ---------------------------------------------------------------------------
print("Loading vector store...")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
EMB_MODEL_NAME = meta["model"]  # "intfloat/multilingual-e5-base"

chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
embeddings = np.load(EMB_PATH)  # shape (N, 768), L2-normalized

print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")

# LDA topics (see assign_lda_topics.py) — each chunk carries a "lda_topic" id.
lda_topics = json.loads(LDA_TOPICS_PATH.read_text(encoding="utf-8"))
topic_chunk_indices: dict[int, list[int]] = {t["id"]: [] for t in lda_topics}
for i, c in enumerate(chunks):
    topic_chunk_indices.setdefault(c.get("lda_topic", -1), []).append(i)
print(f"Loaded {len(lda_topics)} LDA topics")

print(f"Loading embedding model: {EMB_MODEL_NAME} (first run downloads ~1GB)...")
embed_model = SentenceTransformer(EMB_MODEL_NAME)
print("Ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="STEK 2035 RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None
    topic: int | None = None  # restrict retrieval to this LDA topic id, if set


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


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]


def retrieve(query: str, k: int, topic: int | None = None) -> list[dict]:
    """Embed the query and return the top-k most similar chunks.

    If `topic` is given, only chunks whose dominant LDA topic matches are
    considered.
    """
    query_vec = embed_model.encode(f"query: {query}", normalize_embeddings=True).astype("float32")
    candidate_idx = np.array(topic_chunk_indices.get(topic, [])) if topic is not None else np.arange(len(chunks))
    # embeddings are L2-normalized -> cosine similarity is a plain dot product
    scores = embeddings[candidate_idx] @ query_vec
    ranked = np.argsort(-scores)[:k]
    results = []
    for r in ranked:
        i = int(candidate_idx[r])
        c = chunks[i]
        results.append({
            "origin": c.get("origin", "unknown"),
            "chunk_id": c.get("chunk_id", -1),
            "text": c.get("text", ""),
            "score": float(scores[r]),
        })
    return results


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Quelle: {c['origin']}]\n{c['text']}" for c in contexts
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
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks), "ollama_model": OLLAMA_MODEL}


@app.get("/topics", response_model=list[Topic])
def topics():
    return lda_topics


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    k = req.top_k or TOP_K
    contexts = retrieve(req.message, k, topic=req.topic)
    prompt = build_prompt(req.message, contexts)
    reply = ask_ollama(prompt)
    return ChatResponse(
        reply=reply,
        sources=[Source(**c) for c in contexts],
    )
