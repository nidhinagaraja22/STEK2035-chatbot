"""
hybrid_gate.py — combined BM25 + embedding relevance gate for STEK 2035 RAG

Runs BEFORE the expensive top-5 chunk retrieval + qwen2.5:32b generation call.
Catches three cases from cheapest to most expensive:

  1. Out-of-domain query   (e.g. "permits in Mumbai")      -> reject, no LLM call
  2. Vague/generic query   (e.g. "Erzähl mir von Heidelberg") -> ask for clarification
  3. In-domain query, including synonym forms BM25 alone
     would miss (e.g. "Erlaubnis" instead of "Genehmigung") -> proceed to RAG

Uses the SAME lemmatize() function and SYNONYMS table as ingestion_pipeline.py —
this file imports them directly so the two can never drift out of sync.

Usage:
    python hybrid_gate.py
"""

import json
import re
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ingestion_pipeline import lemmatize, EMBEDDING_MODEL_NAME, CHROMA_PATH, COLLECTION_NAME

# ---------------------------------------------------------------------------
# Config — tune these against your own benchmark once you have real traffic
# ---------------------------------------------------------------------------

ALPHA = 0.6                # shifted toward BM25: diagnostic testing showed raw BM25
                            # tracks true relevance more reliably than centered cosine
                            # for this corpus (e.g. an exact "Gewinnung von Fachkräften"
                            # match scored cos=0.16 — lower than a vague small-talk query)
REJECT_THRESHOLD = 0.53    # NOTE: needs re-measurement after the ALPHA/cap change below —
                            # this value was fit to the OLD scoring scale.
BM25_FIXED_CAP = 11.0       # recalibrated from real observed raw scores: genuinely
                            # relevant matches ranged ~4-11.2 in the 20-question test;
                            # the old cap of 8.0 saturated most strong matches to 1.0,
                            # destroying ranking and falsely flattening the top-5 spread
                            # (which is what caused several false VAGUE verdicts).
VAGUE_MIN_WORDS = 8        # queries shorter than this, with a flat score spread, are vague
VAGUE_TOPN = 5              # how many top chunks to compare for vagueness
VAGUE_SPREAD_THRESHOLD = 0.03  # if top-N raw scores are within this range of each other -> vague
VAGUE_SCORE_MIN = 0.25         # below this + flat spread = genuinely nothing relevant found,
                                # not vagueness — e.g. OCR-garbled chunks scoring uniformly
                                # low aren't "vague", they're irrelevant noise
VAGUE_SCORE_MAX = 0.60         # above this + flat spread = many chunks STRONGLY agreeing on
                                # the same well-covered answer — that's a good sign, not
                                # vagueness (confirmed: "bezahlbaren Wohnraum" scored 0.70-0.73
                                # near-identically across 5 chunks because the topic is
                                # genuinely well documented, not because the query was vague)
MIN_MATCHED_TERMS = 2          # a chunk must share at least this many distinct lemmas with
                                # the query for BM25 to count it at all — prevents one rare,
                                # high-IDF word from single-handedly producing a high score
                                # in an otherwise irrelevant chunk (confirmed: a birds/lizards
                                # chunk scored high on "Fachkräfte" via one coincidental term)
MIN_QUERY_TERMS_FOR_GUARD = 2   # only apply the guard when the query itself has >=2 content
                                # words — a genuine single-word query can't be held to a
                                # 2-term-match bar

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Load the same chunk store built by ingestion_pipeline.py
# ---------------------------------------------------------------------------

def restore_umlauts(text: str) -> str:
    """
    Restores common ASCII-only German umlaut substitutions to proper Unicode
    form (the standard DIN 5007 convention: ue->ü, oe->ö, ae->ä), e.g.
    "fuer" -> "für", "gruenflaechen" -> "grünflächen".

    Applied ONLY to live user queries, never to the stored corpus — the
    corpus was already extracted with correct umlauts, and running this on
    already-correct text risks corrupting rare words that coincidentally
    contain these letter pairs (e.g. "Frequenz" contains "ue").

    Confirmed root cause: without this, spaCy's lemmatizer doesn't recognize
    ASCII-typed words like "gruenflaechen" as real German nouns and falls
    back to a crude truncated guess ("gruenflaech") that shares NO token
    with the corpus's correctly-spelled "grünfläche" — explaining why
    Grünflächen/Fachkräfte/Straßenbahnlinien-style queries kept failing
    through every scoring fix, independent of the actual scoring logic.
    """
    for ascii_form, umlaut in [("ue", "ü"), ("oe", "ö"), ("ae", "ä")]:
        text = text.replace(ascii_form, umlaut)
        text = text.replace(ascii_form.capitalize(), umlaut.upper())
    # "ss" -> "ß" is NOT applied globally: post-1996 spelling keeps "ss" in
    # many common words ("dass", "muss", "Prozess") and only "ß" after long
    # vowels/diphthongs in specific words. Blind substitution would break
    # more than it fixes. Targeted fix for the word family that actually
    # came up in testing:
    text = re.sub(r"strasse", "straße", text, flags=re.IGNORECASE)
    return text


def lemmatize_query(text: str) -> list[str]:
    """Query-side entry point: restores umlauts, then lemmatizes exactly as
    ingestion does. Use this (not the raw lemmatize import) for any live
    user query, so ASCII-typed input matches the corpus's correct spelling."""
    return lemmatize(restore_umlauts(text))


class RelevanceGate:
    def __init__(self, chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(collection_name)
        all_data = collection.get(include=["metadatas", "documents", "embeddings"])

        self.texts = all_data["documents"]
        self.metadatas = all_data["metadatas"]
        raw_embeddings = np.array(all_data["embeddings"])  # already unit-normalized at ingestion

        # Pre-parse every chunk's bm25_tokens into a set, so per-query we can
        # cheaply count how many DISTINCT query terms actually appear in each
        # chunk — needed for the single-rare-word-coincidence guard below.
        self.chunk_token_sets = [set(json.loads(m["bm25_tokens"])) for m in self.metadatas]

        # --- Anisotropy correction ---
        # Transformer sentence embeddings (e5-large included) cluster tightly in
        # one region of the vector space: cosine similarity between ANY two
        # fluent German sentences sits in a narrow high band (~0.75-0.95),
        # mostly reflecting "this is coherent German text" rather than topical
        # relevance. Subtracting the corpus mean vector removes that shared
        # direction, so the remaining similarity better reflects actual
        # topic overlap. Re-normalize afterward so cosine math stays valid.
        self.mean_embedding = raw_embeddings.mean(axis=0)
        centered = raw_embeddings - self.mean_embedding
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.chunk_embeddings = centered / norms

        # BM25 index built from tokens stored at ingestion time — never recomputed here
        tokenized_corpus = [json.loads(m["bm25_tokens"]) for m in self.metadatas]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Kept only for reference/debugging — NOT used in _hybrid_scores anymore.
        Per-query min-max normalization always maps the best-matching chunk to 1.0
        and the worst to 0.0, which erases the very signal an absolute reject
        threshold depends on. See _hybrid_scores() for the fix."""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def _hybrid_scores(self, question: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (hybrid, bm25_raw, bm25_scaled, cos_scaled) using RAW scales, not
        per-query min-max normalization. Min-max normalization always rescales
        the best-matching chunk to 1.0 and the worst to 0.0 regardless of
        whether the query is actually relevant to anything — which silently
        defeats any absolute reject threshold. Raw cosine similarity is already
        bounded and comparable across queries; BM25 is capped by a FIXED
        constant (not the per-query max) so its scale stays stable too.
        """
        # BM25 side — raw score, capped by a fixed constant (never per-query max)
        query_tokens = set(lemmatize_query(question))
        bm25_raw = np.array(self.bm25.get_scores(list(query_tokens)))
        bm25_scaled = np.clip(bm25_raw / BM25_FIXED_CAP, 0.0, 1.0)

        # Single-rare-word-coincidence guard: a chunk matching only ONE
        # distinct query term can still score high via BM25 if that term
        # happens to have high IDF (rare corpus-wide) — confirmed cases:
        # a birds/lizards chunk scored 8.4 raw on a "Fachkräfte" query via
        # one coincidental term, and a boilerplate chunk kept scoring high
        # on "Kartoffelsalat" via a single incidental word like "klassisch".
        # Require >=2 distinct matches when the query itself has >=2 content
        # words (a genuine single-word query can't be held to that bar).
        if len(query_tokens) >= MIN_QUERY_TERMS_FOR_GUARD:
            matched_counts = np.array([
                len(query_tokens & chunk_tokens) for chunk_tokens in self.chunk_token_sets
            ])
            bm25_scaled = np.where(matched_counts >= MIN_MATCHED_TERMS, bm25_scaled, 0.0)
        # Embedding side — center the query the same way the corpus was centered,
        # then re-normalize before the dot product (see __init__ for why).
        query_emb = embed_model.encode(f"query: {question}", normalize_embeddings=True)
        q_centered = query_emb - self.mean_embedding
        q_centered = q_centered / (np.linalg.norm(q_centered) + 1e-8)
        cos_raw = self.chunk_embeddings @ q_centered
        cos_scaled = np.clip(cos_raw, 0.0, 1.0)

        hybrid = ALPHA * bm25_scaled + (1 - ALPHA) * cos_scaled
        return hybrid, bm25_raw, bm25_scaled, cos_scaled

    def check(self, question: str, top_n_chunks: int = 5) -> dict:
        hybrid, bm25_raw, bm25_scaled, cos_scaled = self._hybrid_scores(question)
        top_idx = int(np.argmax(hybrid))
        top_score = float(hybrid[top_idx])

        # Vagueness: compare the top-N scores to each other (not top vs global
        # min, which is meaningless at corpus scale — almost everything has
        # SOME chunk near zero relevance).
        top_n = np.sort(hybrid)[-VAGUE_TOPN:]
        spread = float(top_n.max() - top_n.min())
        word_count = len(question.split())

        # Vagueness only applies in a MIDDLE score band, not just "flat +
        # short": too low + flat means genuinely nothing relevant (reject,
        # don't call it vague), too high + flat means many chunks strongly
        # agreeing on a real answer (accept, don't call it vague either).
        is_flat_and_short = word_count < VAGUE_MIN_WORDS and spread < VAGUE_SPREAD_THRESHOLD
        if is_flat_and_short and VAGUE_SCORE_MIN <= top_score < VAGUE_SCORE_MAX:
            verdict = "VAGUE"
        elif top_score < REJECT_THRESHOLD:
            verdict = "OUT_OF_DOMAIN"
        else:
            verdict = "IN_DOMAIN"

        # Diagnostic breakdown of the top-N chunks, so you can see whether a
        # low score means "nothing like this exists" (both bm25 and cos low
        # across the board) or "it exists but is phrased differently"
        # (cos moderate, bm25 near-zero — a synonym/terminology gap).
        ranked_idx = np.argsort(hybrid)[::-1][:top_n_chunks]
        top_chunks = [{
            "rank": rank + 1,
            "hybrid_score": round(float(hybrid[idx]), 4),
            "bm25_raw": round(float(bm25_raw[idx]), 4),
            "bm25_scaled": round(float(bm25_scaled[idx]), 4),
            "cos_scaled": round(float(cos_scaled[idx]), 4),
            "source": self.metadatas[idx]["source"],
            "doc_type": self.metadatas[idx].get("doc_type", "?"),
            "text_preview": self.texts[idx][:100],
            "tokens": sorted(self.chunk_token_sets[idx]),  # for synonym-candidate mining
        } for rank, idx in enumerate(ranked_idx)]

        return {
            "verdict": verdict,
            "top_score": round(top_score, 4),
            "spread": round(spread, 4),
            "best_chunk_preview": self.texts[top_idx][:80],
            "best_chunk_source": self.metadatas[top_idx]["source"],
            "top_chunks": top_chunks,
        }

    def diagnose(self, question: str, top_n_chunks: int = 5) -> None:
        """Prints a readable breakdown of the top-N chunks for a query —
        use this whenever a score looks surprisingly low or high."""
        result = self.check(question, top_n_chunks=top_n_chunks)
        print(f"\nQuery: {question}")
        print(f"Verdict: {result['verdict']}  (top_score={result['top_score']}, spread={result['spread']})")
        print(f"{'#':3} {'hybrid':7} {'bm25_raw':9} {'bm25_sc':8} {'cos':7} {'doc_type':16} source / text")
        print("-" * 110)
        for c in result["top_chunks"]:
            print(f"{c['rank']:3} {c['hybrid_score']:.4f}  {c['bm25_raw']:8.3f}  {c['bm25_scaled']:.4f}  {c['cos_scaled']:.4f}  "
                  f"{c['doc_type']:16} {c['source']}")
            print(f"      \u21b3 {c['text_preview']}...")


# ---------------------------------------------------------------------------
# How your FastAPI RAG endpoint should use this
# ---------------------------------------------------------------------------

def handle_query(gate: RelevanceGate, question: str) -> str:
    result = gate.check(question)

    if result["verdict"] == "OUT_OF_DOMAIN":
        return ("Diese Frage liegt außerhalb des STEK-2035-Korpus für Heidelberg. "
                "Ich kann nur zu städtischen Planungsthemen für Heidelberg antworten.")

    if result["verdict"] == "VAGUE":
        return ("Ihre Frage ist sehr allgemein. Können Sie präzisieren, welches Thema "
                "Sie interessiert — z. B. Wohnen, Mobilität, Umwelt, Wirtschaft oder Soziales?")

    # verdict == IN_DOMAIN -> proceed to normal RAG pipeline (top-5 retrieval + qwen2.5:32b)
    return f"[PROCEED TO RAG] closest chunk from '{result['best_chunk_source']}': {result['best_chunk_preview']}..."


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gate = RelevanceGate()

    # Queries that scored surprisingly low or high in the 20-question
    # calibration test — diagnose() shows the top-5 chunks and their
    # individual bm25/cos contributions so you can see WHY.
    diagnostic_queries = [
        "Wie will die Stadt die Grünflächen erhalten?",              # 0.3497 — lowest real in-domain score
        "Wie unterstützt die Stadt die Gewinnung von Fachkräften?",  # 0.4142
        "Gibt es Pläne für neue Straßenbahnlinien in Heidelberg?",   # 0.4958
        "Wie bereite ich einen klassischen Kartoffelsalat zu?",      # 0.4368 — off-topic, for comparison
        "Welche Genehmigungen brauche ich, um in München zu wohnen?", # 0.4382 — off-topic, for comparison
    ]

    for q in diagnostic_queries:
        gate.diagnose(q)