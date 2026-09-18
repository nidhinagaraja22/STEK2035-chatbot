"""
STEK 2035 — Generate e5-large and BGE-M3 Embeddings
========================================================
Generates the two missing embedding sets needed for Task 6
(Precision/Recall/MRR/nDCG@k across embedding models) and Task 9
(speed verification), for your CURRENT corpus.

Important API difference between the two models:
  - multilingual-e5-large uses the "query: " / "passage: " prefix
    convention (same as e5-base, already in use throughout this
    project)
  - BGE-M3 does NOT use this prefix convention — passing it would
    degrade BGE-M3's embeddings, not just be redundant

Usage:
    python stek_generate_e5large_bgem3.py

Requires:
    Your current corpus chunks file (adjust CHUNKS_PATH below)
    sentence-transformers, torch

Outputs:
    embeddings_v2_e5large.npy   (1024-dim)
    embeddings_v2_bgem3.npy     (1024-dim)
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import json

# ── ADJUST THIS to your actual current corpus file ───────────────────────────
CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")

OUTPUT_DIR = Path("corpus/corpus_v2")
BATCH_SIZE = 32  # smaller than e5-base's batch size — both these models are
                  # larger and more memory-hungry per item

MODELS = {
    "e5large": {
        "hf_name": "intfloat/multilingual-e5-large",
        "output_file": OUTPUT_DIR / "embeddings_v2_e5large.npy",
        "prefix": "passage: ",   # e5 models need this prefix on CHUNK text
    },
    "bgem3": {
        "hf_name": "BAAI/bge-m3",
        "output_file": OUTPUT_DIR / "embeddings_v2_bgem3.npy",
        "prefix": "",             # BGE-M3 does NOT use e5-style prefixes —
                                   # passing one would degrade quality
    },
}


def load_chunk_texts():
    texts = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
    return texts


def main():
    print("=" * 70)
    print("STEK 2035 — Generate e5-large and BGE-M3 Embeddings")
    print("=" * 70)

    if not CHUNKS_PATH.exists():
        print(f"\n❌ {CHUNKS_PATH} not found — adjust CHUNKS_PATH at the top of this script.")
        return

    print(f"\nLoading chunks from: {CHUNKS_PATH}")
    texts = load_chunk_texts()
    print(f"  {len(texts)} chunks to embed")

    for model_key, config in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Generating: {model_key} ({config['hf_name']})")
        print(f"{'='*70}")

        if config["output_file"].exists():
            print(f"  ⚠️  {config['output_file']} already exists — skipping.")
            print(f"     Delete it first if you want to regenerate.")
            continue

        print(f"  Loading model (this may take a while on first run — downloads weights)...")
        model = SentenceTransformer(config["hf_name"])

        prefixed_texts = [f"{config['prefix']}{t}" for t in texts]

        print(f"  Embedding {len(prefixed_texts)} chunks in batches of {BATCH_SIZE}...")
        embeddings = model.encode(
            prefixed_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        print(f"  Shape: {embeddings.shape}")
        if embeddings.shape[0] != len(texts):
            print(f"  ❌ Row count mismatch! Expected {len(texts)}, got {embeddings.shape[0]} — NOT saving.")
            continue

        np.save(config["output_file"], embeddings.astype(np.float32))
        print(f"  ✅ Saved: {config['output_file']}")

        # free memory before loading the next model
        del model
        import gc
        gc.collect()

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"\nVerify both files before using them:")
    print(f"""
python3 -c "
import numpy as np
for f in ['{MODELS['e5large']['output_file']}', '{MODELS['bgem3']['output_file']}']:
    try:
        e = np.load(f)
        print(f, e.shape)
    except Exception as ex:
        print(f, 'MISSING or ERROR:', ex)
"
""")
    print(f"Both should show ({len(texts)}, 1024) — same row count as your")
    print(f"e5-base embeddings, but 1024 dimensions instead of 768.")


if __name__ == "__main__":
    main()
