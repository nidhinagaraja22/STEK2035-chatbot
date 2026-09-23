"""
STEK 2035 — Top-5 Retrieval Comparison Across Different Cities
====================================================================
Demonstrates concretely what actually gets retrieved for the SAME
question structure ("Ich lebe in <CITY>. Welche Genehmigungen
werden benötigt?"), varying only the city — Heidelberg (in-scope),
Mannheim/Frankfurt (out-of-scope, some tangential mentions), and
Mumbai (out-of-scope, zero mentions).

This makes concrete what was established by direct keyword search
earlier: similarity search ALWAYS returns a top-5, regardless of
whether anything genuinely relevant exists — the question is
whether the SCORES and CONTENT make that fact visible, or whether
a plausible-looking wrong answer could be assembled from them.

Usage:
    python stek_city_retrieval_comparison.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
"""

from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
TOP_K = 5

TEST_CITIES = ["Heidelberg", "Mannheim", "Frankfurt", "Mumbai"]
QUESTION_TEMPLATE = "Ich lebe in {city}. Welche Genehmigungen werden benötigt?"


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


def main():
    print("=" * 70)
    print("Top-5 Retrieval Comparison Across Different Cities")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return
    print(f"  {len(chunks)} chunks loaded\n")

    all_results = {}

    for city in TEST_CITIES:
        question = QUESTION_TEMPLATE.format(city=city)
        print(f"\n{'='*70}")
        print(f"CITY: {city}")
        print(f"Question: {question}")
        print(f"{'='*70}")

        q_vec = model.encode(f"query: {question}")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        sims = chunk_embeddings @ q_vec
        top_idx = np.argsort(-sims)[:TOP_K]

        city_results = []
        for rank, i in enumerate(top_idx, 1):
            score = float(sims[i])
            doc_id = chunks[i]["document_id"]
            authority = chunks[i].get("authority_level", "?")
            text_preview = chunks[i]["text"][:180].replace("\n", " ")
            mentions_city = city.lower() in chunks[i]["text"].lower()

            print(f"\n  Rank {rank}: score={score:.4f}  doc={doc_id}  L{authority}  "
                  f"mentions '{city}': {mentions_city}")
            print(f"    {text_preview}...")

            city_results.append({"rank": rank, "score": score, "document_id": doc_id,
                                    "authority_level": authority, "mentions_city": mentions_city,
                                    "text_preview": text_preview})

        all_results[city] = city_results

    # summary comparison
    print(f"\n\n{'='*70}")
    print("SUMMARY — Top-1 Score by City")
    print(f"{'='*70}")
    for city in TEST_CITIES:
        top1 = all_results[city][0]
        print(f"  {city:<12}: top1_score={top1['score']:.4f}  mentions_city={top1['mentions_city']}")

    print(f"\n👉 Look specifically at:")
    print(f"   1. Does the top score DROP meaningfully for out-of-scope cities,")
    print(f"      or does it stay deceptively high regardless?")
    print(f"   2. For Mumbai specifically — since we confirmed 0 corpus chunks")
    print(f"      mention it — what content got retrieved instead, and would")
    print(f"      it plausibly mislead an LLM into fabricating a Mumbai-specific")
    print(f"      answer from unrelated Heidelberg content?")

    with open("city_retrieval_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: city_retrieval_comparison_results.json")


if __name__ == "__main__":
    main()
