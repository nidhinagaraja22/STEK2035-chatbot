# -*- coding: utf-8 -*-
"""
Generate embeddings for the STEK corpus with all three models.
==============================================================
Standalone script — upload to the cloud/GPU box and run. Reads a chunks JSONL
(each line a JSON object with a "text" field) and writes one L2-normalized
embedding file per model, row-aligned to the chunks.

    e5base  -> intfloat/multilingual-e5-base   (768)   prefix "passage: "
    e5large -> intfloat/multilingual-e5-large  (1024)  prefix "passage: "
    bgem3   -> BAAI/bge-m3                      (1024)  NO prefix

Correctness:
  * e5 models REQUIRE the "passage: " prefix on documents; bge-m3 must NOT get
    it (a prefix degrades bge-m3).
  * embeddings are L2-normalized on save, so cosine similarity == dot product
    (what rag_server retrieval and any dedup step expect).
  * every output has the SAME row order as the chunks file.

Requires:
    pip install sentence-transformers torch numpy

Run:
    python generate_embeddings.py
    python generate_embeddings.py path/to/chunks.jsonl path/to/out_dir
    python generate_embeddings.py --only e5base            # one model only
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Config — edit these two paths if your layout differs, or pass them as args.
# --------------------------------------------------------------------------- #
CHUNKS_PATH = Path("corpus/corpus_v2/chunks_v2_chunks.jsonl")
OUT_DIR = Path("corpus/corpus_v2")

MODELS = {
    "e5base":  {"hf": "intfloat/multilingual-e5-base",  "prefix": "passage: ", "batch": 64},
    "e5large": {"hf": "intfloat/multilingual-e5-large", "prefix": "passage: ", "batch": 32},
    "bgem3":   {"hf": "BAAI/bge-m3",                     "prefix": "",          "batch": 16},
}
# Output file per model: embeddings_<key>.npy
OUT_TEMPLATE = "embeddings_{key}.npy"


def load_chunks(path: Path):
    if not path.exists():
        raise SystemExit(f"Chunks file not found: {path}\n"
                         f"Pass the correct path: python generate_embeddings.py <chunks.jsonl> [out_dir]")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def embed_one(key: str, chunks: list, out_dir: Path):
    cfg = MODELS[key]
    texts = [cfg["prefix"] + c.get("text", "") for c in chunks]
    print(f"\n=== {key}  ({cfg['hf']}) ===")
    print(f"  prefix={cfg['prefix']!r}  batch={cfg['batch']}  chunks={len(texts)}")
    model = SentenceTransformer(cfg["hf"])
    emb = model.encode(
        texts,
        batch_size=cfg["batch"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,          # L2-normalize -> cosine == dot
    ).astype(np.float32)
    if emb.shape[0] != len(chunks):
        raise SystemExit(f"{key}: row mismatch — {emb.shape[0]} embeddings vs {len(chunks)} chunks.")
    out_path = out_dir / OUT_TEMPLATE.format(key=key)
    np.save(out_path, emb)
    print(f"  saved {out_path}  shape={emb.shape}")
    return emb.shape


def main():
    args = [a for a in sys.argv[1:]]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        del args[i:i + 2]
    chunks_path = Path(args[0]) if len(args) >= 1 else CHUNKS_PATH
    out_dir = Path(args[1]) if len(args) >= 2 else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [only] if only else list(MODELS)
    for k in keys:
        if k not in MODELS:
            raise SystemExit(f"Unknown model key {k!r}. Choose from: {', '.join(MODELS)}")

    chunks = load_chunks(chunks_path)
    print(f"Chunks: {len(chunks)} from {chunks_path}")
    print(f"Models: {', '.join(keys)}  ->  {out_dir}/")

    shapes = {}
    for k in keys:
        shapes[k] = embed_one(k, chunks, out_dir)

    print("\nDone. Written files:")
    for k, shp in shapes.items():
        print(f"  {OUT_TEMPLATE.format(key=k)}  {shp}")


if __name__ == "__main__":
    main()
