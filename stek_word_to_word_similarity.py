"""
STEK 2035 — Word-to-Word Similarity: Is "Mumbai" Genuinely More Distant?
=============================================================================
Tests the hypothesis directly: does "Mumbai" score LOWER in embedding
similarity than "Mannheim"/"Frankfurt" do, when compared purely as
single words/short phrases — isolating the CITY NAME's own semantic
distance from "Heidelberg", separate from all the other shared words
("Genehmigungen", "leben", etc.) that diluted the signal in the full-
sentence retrieval test.

Two complementary checks:
  1. Direct word-to-word: cosine_sim(embed("Mumbai"), embed("Heidelberg"))
     vs the same for Mannheim/Frankfurt — does geographic/cultural
     distance show up as embedding distance?
  2. Word-to-corpus: average similarity of each city name against a
     sample of real corpus chunks — does "Mumbai" score lower against
     the corpus AS A WHOLE than the German cities do?

Usage:
    python stek_word_to_word_similarity.py

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

CITIES = ["Heidelberg", "Mannheim", "Frankfurt", "Mumbai"]
SAMPLE_SIZE = 100  # random sample of corpus chunks for the word-to-corpus check


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


def normalise_vec(v):
    return v / max(np.linalg.norm(v), 1e-8)


def main():
    print("=" * 70)
    print("Word-to-Word Similarity: Is 'Mumbai' Genuinely More Distant?")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # ── Check 1: direct word-to-word similarity against "Heidelberg" ────────
    print(f"\n{'='*70}")
    print("CHECK 1 — Direct word-to-word similarity vs 'Heidelberg'")
    print(f"{'='*70}")

    heidelberg_vec = normalise_vec(model.encode("Heidelberg"))
    word_to_word_results = {}

    for city in CITIES:
        city_vec = normalise_vec(model.encode(city))
        sim = float(np.dot(heidelberg_vec, city_vec))
        word_to_word_results[city] = sim
        print(f"  cosine_sim(Heidelberg, {city}): {sim:.4f}")

    # ── Check 2: word-to-corpus — average similarity against real chunks ────
    print(f"\n{'='*70}")
    print(f"CHECK 2 — Average similarity vs {SAMPLE_SIZE} random real corpus chunks")
    print(f"{'='*70}")

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    np.random.seed(42)
    sample_idx = np.random.choice(len(chunks), size=min(SAMPLE_SIZE, len(chunks)), replace=False)
    sample_embeddings = chunk_embeddings[sample_idx]

    word_to_corpus_results = {}
    for city in CITIES:
        city_vec = normalise_vec(model.encode(f"query: {city}"))
        sims = sample_embeddings @ city_vec
        mean_sim = float(np.mean(sims))
        word_to_corpus_results[city] = mean_sim
        print(f"  {city:<12}: mean_sim={mean_sim:.4f}  max_sim={float(np.max(sims)):.4f}")

    # ── Summary — does the hypothesis hold on BOTH checks? ──────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — Does 'Mumbai' Score Lower on Both Checks?")
    print(f"{'='*70}")

    w2w_sorted = sorted(word_to_word_results.items(), key=lambda x: -x[1])
    w2c_sorted = sorted(word_to_corpus_results.items(), key=lambda x: -x[1])

    print(f"\nCheck 1 ranking (word-to-word vs Heidelberg), highest to lowest:")
    for city, sim in w2w_sorted:
        print(f"  {city:<12}: {sim:.4f}")

    print(f"\nCheck 2 ranking (word-to-corpus mean), highest to lowest:")
    for city, sim in w2c_sorted:
        print(f"  {city:<12}: {sim:.4f}")

    mumbai_lowest_w2w = w2w_sorted[-1][0] == "Mumbai"
    mumbai_lowest_w2c = w2c_sorted[-1][0] == "Mumbai"
    print(f"\nMumbai scores LOWEST on word-to-word check: {mumbai_lowest_w2w}")
    print(f"Mumbai scores LOWEST on word-to-corpus check: {mumbai_lowest_w2c}")

    if mumbai_lowest_w2w and mumbai_lowest_w2c:
        print(f"\n✅ Hypothesis CONFIRMED on both checks — city-name-alone similarity")
        print(f"   could be a genuine, isolated out-of-scope signal, distinct from")
        print(f"   the diluted whole-sentence version tested earlier.")
    else:
        print(f"\n⚠️  Hypothesis only PARTIALLY confirmed or not confirmed —")
        print(f"   worth checking which specific city broke the expected pattern.")

    with open("word_to_word_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump({"word_to_word_vs_heidelberg": word_to_word_results,
                    "word_to_corpus_mean": word_to_corpus_results}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: word_to_word_similarity_results.json")


if __name__ == "__main__":
    main()
