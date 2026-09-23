"""
boilerplate_finder.py — identifies boilerplate/generic terms in the corpus
using document frequency (the DF half of TF-IDF).

Rationale: a word that appears in a huge fraction of chunks (e.g. "Stadt",
"Ziel", "Massnahme") carries almost no distinguishing information — it's
present whether the chunk is about Wohnen, Mobilitaet, or anything else.
BM25's IDF term already down-weights these, but if a query happens to
match SEVERAL such generic words in one chunk, the SUM can still add up to
a misleadingly high score (this is likely part of why "Kartoffelsalat"
scored higher than several genuine STEK questions in earlier testing).

Uses the SAME bm25_tokens already stored at ingestion time — no separate
tokenization step, so results reflect exactly what BM25 sees.

Usage:
    python boilerplate_finder.py
"""

import json
from collections import Counter
import chromadb
from ingestion_pipeline import CHROMA_PATH, COLLECTION_NAME


def find_boilerplate(min_doc_freq_ratio: float = 0.05, top_n: int = 40):
    """
    min_doc_freq_ratio: a term appearing in more than this fraction of ALL
    chunks is flagged as boilerplate. 0.05 = appears in 5%+ of chunks.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    all_data = collection.get(include=["metadatas"])

    total_chunks = len(all_data["metadatas"])
    doc_freq = Counter()  # term -> number of DIFFERENT chunks containing it

    for meta in all_data["metadatas"]:
        tokens = json.loads(meta["bm25_tokens"])
        unique_tokens_in_chunk = set(tokens)  # count each term once per chunk
        for t in unique_tokens_in_chunk:
            doc_freq[t] += 1

    print(f"Total chunks: {total_chunks}")
    print(f"Unique lemma tokens: {len(doc_freq)}\n")

    threshold_count = total_chunks * min_doc_freq_ratio
    boilerplate = [
        (term, count, count / total_chunks)
        for term, count in doc_freq.items()
        if count >= threshold_count
    ]
    boilerplate.sort(key=lambda x: -x[1])

    print(f"Terms appearing in >= {min_doc_freq_ratio*100:.0f}% of chunks "
          f"(>= {threshold_count:.0f} chunks) — BOILERPLATE CANDIDATES:\n")
    print(f"{'Term':25} {'Chunks':>8} {'% of corpus':>12}")
    print("-" * 50)
    for term, count, ratio in boilerplate[:top_n]:
        print(f"{term:25} {count:>8} {ratio*100:>11.1f}%")

    print(f"\nTotal boilerplate candidates found: {len(boilerplate)}")
    return [term for term, _, _ in boilerplate]


if __name__ == "__main__":
    boilerplate_terms = find_boilerplate(min_doc_freq_ratio=0.05, top_n=40)

    print("\nSuggested DOMAIN_STOPWORDS set (copy into ingestion_pipeline.py, "
          "review before using — some flagged terms may still carry real "
          "meaning, e.g. don't blindly exclude 'heidelberg' if the corpus "
          "has non-Heidelberg content mixed in):\n")
    print("DOMAIN_STOPWORDS = {")
    for term in boilerplate_terms:
        print(f'    "{term}",')
    print("}")
