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
CORPUS_DIR = Path("corpus/corpus_v2")                     # Corpus 2: deduped + metadata + authority
CHUNKS_PATH = CORPUS_DIR / "corpus_v2_chunks.jsonl"
EMB_PATH = CORPUS_DIR / "embeddings_v2_e5base.npy"
META_PATH = CORPUS_DIR / "meta_v2.json"
LDA_TOPICS_PATH = VECTOR_STORE_DIR / "lda_topics.json"    # topic labels (k=8 ids still valid for Corpus 2)

TOP_K = 5
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:32b"  # change to whatever model you've pulled, e.g. "mistral", "qwen2.5","sadad"

# ---------------------------------------------------------------------------
# Authority taxonomy (L1-L5) — German source-type tags shown to the LLM.
# The corpus carries an English `authority_label` per chunk; for the German
# answer we tag each passage by its authority LEVEL so the model can tell
# approved city policy apart from individual citizen opinions.
# ---------------------------------------------------------------------------
AUTHORITY_DE = {
    1: "OFFIZIELLE STEK-STRATEGIE (beschlossene Politik der Stadt)",
    2: "OFFIZIELLER BERICHT / Fachplanung der Stadt",
    3: "ERGÄNZENDES FACHMATERIAL der Stadt",
    4: "BÜRGERBETEILIGUNG (zusammengefasste Bürgermeinungen)",
    5: "EINZELNE BÜRGERMEINUNG (Meinung einer einzelnen Person)",
}
# Levels 1-3 = the city speaking officially; 4-5 = residents speaking.
OFFICIAL_LEVELS = {1, 2, 3}

# ---------------------------------------------------------------------------
# Load vector store once at startup
# ---------------------------------------------------------------------------
print("Loading vector store...")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
EMB_MODEL_NAME = meta["model"]  # "intfloat/multilingual-e5-base"

chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
embeddings = np.load(EMB_PATH)  # shape (N, dim), L2-normalized

# Fail loudly if the corpus and embeddings are out of sync (e.g. chunks were added
# but the embeddings were not regenerated) rather than silently misaligning or
# raising a confusing IndexError deep in retrieval.
if embeddings.shape[0] != len(chunks):
    raise RuntimeError(
        f"Corpus/embeddings mismatch: {len(chunks)} chunks but {embeddings.shape[0]} "
        f"embedding rows. Re-embed the corpus (e.g. python stek_reembed_cleaned_corpus.py) "
        f"so {EMB_PATH.name} matches {CHUNKS_PATH.name}."
    )

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
    authority_level: int | None = None
    authority_label: str | None = None
    document_title: str | None = None
    source_url: str | None = None
    is_citizen_opinion: bool = False


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
            "authority_level": c.get("authority_level"),
            "authority_label": c.get("authority_label"),
            "document_title": c.get("document_title"),
            "source_url": c.get("source_url"),
            "is_citizen_opinion": bool(c.get("is_citizen_opinion", False)),
        })
    return results


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Authority-aware prompt.

    Every passage is tagged with its authority level (L1-L5) so the LLM can
    distinguish approved city policy from individual citizen opinions, and the
    instructions forbid presenting a citizen wish as official policy. This is
    the "Authority-Aware RAG" fix: standard RAG treats every retrieved chunk as
    equally authoritative, which lets a resident's suggestion be answered as if
    it were the city's decision.
    """
    context_block = "\n\n".join(
        f"[Passage {i}] Quellenart: {AUTHORITY_DE.get(c.get('authority_level'), 'Unbekannte Quelle')}\n"
        f"Dokument: {c.get('document_title') or c.get('origin', 'unbekannt')}\n"
        f"{c['text']}"
        for i, c in enumerate(contexts, 1)
    )
    has_official = any(c.get("authority_level") in OFFICIAL_LEVELS for c in contexts)
    has_citizen = any(c.get("authority_level") not in OFFICIAL_LEVELS
                      and c.get("authority_level") is not None for c in contexts)

    return f"""Du bist ein hilfreicher Assistent für das Stadtentwicklungskonzept Heidelberg 2035 (STEK 2035).
Beantworte die folgende Frage NUR auf Basis der bereitgestellten Kontextauszüge.
Antworte auf Deutsch, klar und präzise.

WICHTIG — Umgang mit der Quellenart (Autorität):
Jede Passage ist mit ihrer Quellenart gekennzeichnet. Es gibt zwei Kategorien:
- OFFIZIELL (STEK-Strategie, offizielle Berichte, Fachmaterial der Stadt): das ist die
  beschlossene bzw. geplante Position der Stadt Heidelberg.
- BÜRGERMEINUNG (Bürgerbeteiligung, einzelne Bürgermeinung): das sind Wünsche, Anregungen
  oder Meinungen von Bürgerinnen und Bürgern — NICHT die offizielle Position der Stadt.

Regeln:
1. Offizielle Inhalte formulierst du als Position der Stadt, z. B. "Die Stadt plant …",
   "Laut STEK 2035 …", "Im offiziellen Bericht heißt es …".
2. Bürgermeinungen schreibst du IMMER den Bürgerinnen und Bürgern zu, z. B.
   "Im Beteiligungsprozess wünschten sich einige Bürgerinnen und Bürger …",
   "Einzelne Teilnehmende schlugen vor …". Stelle sie NIEMALS als offizielle Politik dar.
3. Wenn zu einer Frage nach der offiziellen Position NUR Bürgermeinungen vorliegen,
   sage ausdrücklich, dass dazu keine offizielle Aussage im Kontext vorliegt, und gib
   die Bürgermeinung nur als solche wieder.
4. Wenn der Kontext keine ausreichenden Informationen zur Beantwortung enthält,
   ERFINDE NICHTS. Antworte in diesem Fall AUSSCHLIESSLICH mit genau diesem Satz:
   "Die vorliegenden STEK-Dokumente enthalten nicht genügend Informationen, um diese Frage zu beantworten."
   Das gilt auch für Fragen zu anderen Städten, für tagesaktuelle Daten, für exakte
   Zahlen/Budgets/Namen, die nicht im Kontext stehen, und für persönliche Empfehlungen.
5. Trenne bei Bedarf klar: was die Stadt offiziell plant vs. was sich Bürger wünschen.

Verfügbare Quellenarten in diesem Kontext: {'offiziell' if has_official else '—'}{' und ' if has_official and has_citizen else ''}{'Bürgermeinung' if has_citizen else ''}.

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
        timeout=120,
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
