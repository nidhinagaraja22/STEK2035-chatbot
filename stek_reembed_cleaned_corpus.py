"""
STEK 2035 — Re-embed Cleaned Corpus
======================================
Boilerplate cleaning removed chunks from corpus_v2_chunks_cleaned.jsonl,
changing row count/order — the old embeddings.npy no longer aligns and
must be regenerated.

This script auto-detects whether you were using e5-base (768-dim) or
e5-large (1024-dim) by checking your EXISTING embeddings file's shape
first — resolving the e5-base/e5-large naming ambiguity from earlier by
checking the actual data rather than the filename or a guess.

Usage:
    python stek_reembed_cleaned_corpus.py

Reads:
    corpus/corpus_v2/corpus_v2_chunks_cleaned.jsonl  (the cleaned corpus)
    corpus/corpus_v2/embeddings_v2_e5base.npy        (OLD file, used only
                                                        to detect dimension
                                                        — not reused directly)

Writes:
    corpus/corpus_v2/embeddings_v2_cleaned.npy       (new, row-aligned
                                                        with the cleaned
                                                        chunks file)
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
OLD_EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")  # used ONLY
                                                                    # to detect
                                                                    # dimension
OUTPUT_EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")

BATCH_SIZE = 64

MODEL_BY_DIM = {
    768: "intfloat/multilingual-e5-base",
    1024: "intfloat/multilingual-e5-large",
}


def detect_model_from_existing_embeddings() -> str:
    if not OLD_EMB_PATH.exists():
        print(f"⚠️  Old embeddings file not found at {OLD_EMB_PATH} — cannot")
        print(f"   auto-detect. Defaulting to multilingual-e5-base. If this is")
        print(f"   wrong, set MODEL_NAME manually below and rerun.")
        return "intfloat/multilingual-e5-base"

    old_emb = np.load(OLD_EMB_PATH)
    dim = old_emb.shape[1]
    model_name = MODEL_BY_DIM.get(dim)

    if model_name is None:
        print(f"⚠️  Unexpected dimension {dim} — neither e5-base (768) nor")
        print(f"   e5-large (1024). Defaulting to e5-base. Check OLD_EMB_PATH")
        print(f"   points at the right file.")
        return "intfloat/multilingual-e5-base"

    print(f"✅ Detected from existing file: {OLD_EMB_PATH}")
    print(f"   Shape: {old_emb.shape} → dimension {dim} → {model_name}")
    print(f"   (This resolves the earlier e5-base/e5-large naming ambiguity")
    print(f"   by reading the actual data, not trusting the filename.)")
    return model_name


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    print("=" * 70)
    print("STEK 2035 — Re-embed Cleaned Corpus")
    print("=" * 70)

    if not CHUNKS_PATH.exists():
        print(f"\n❌ Cleaned chunks file not found: {CHUNKS_PATH}")
        print(f"   Run stek_clean_boilerplate.py first.")
        return

    model_name = detect_model_from_existing_embeddings()

    print(f"\nLoading chunks from: {CHUNKS_PATH}")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks to embed")

    texts = [f"passage: {c.get('text', '')}" for c in chunks]

    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"\nEmbedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"\nEmbeddings shape: {embeddings.shape}")

    if embeddings.shape[0] != len(chunks):
        print(f"❌ Row count mismatch after embedding — {embeddings.shape[0]} "
              f"embeddings vs {len(chunks)} chunks. Something went wrong.")
        return

    np.save(OUTPUT_EMB_PATH, embeddings.astype(np.float32))
    print(f"\n✅ Saved: {OUTPUT_EMB_PATH}")
    print(f"   Shape: {embeddings.shape}")
    print(f"\nUpdate your scripts' EMB_PATH to point at:")
    print(f"   {OUTPUT_EMB_PATH}")
    print(f"and CHUNKS_PATH to:")
    print(f"   {CHUNKS_PATH}")
    print(f"before rerunning any retrieval evaluation.")


if __name__ == "__main__":
    main()
