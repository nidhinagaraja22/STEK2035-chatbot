"""
STEK 2035 — Full Grid: All Questions x All Embedding Models x All LLMs
============================================================================
Generates real answers for every question, across every combination of
embedding model and LLM, in the exact requested output schema.

⚠️ SCALE WARNING: 3 embedding models x 5 LLMs x 60 questions = 900 total
generations. At even a conservative ~8s/generation average (based on this
project's own measured latencies), that's ~2 hours of real runtime minimum
— likely more for the slower models (command-r, gemma2:27b measured at
13-14s average earlier this session). Run this somewhere you can leave
running, not interactively.

⚠️ REQUIRES: embeddings for ALL THREE models (e5-base, e5-large, bge-m3)
must be freshly generated against your CURRENT (sentence-aware, overlapping)
chunks — reusing old embeddings from before the re-chunking would silently
invalidate every retrieval in this run. Check EMB_PATHS below.

Checkpointing: results are appended to a .jsonl file as they complete, one
line per generation. If interrupted, re-running SKIPS combinations already
present in that file — no lost work, no duplicate API calls.

Usage:
    python stek_full_grid_generation.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/embeddings_v2_e5large.npy
    corpus/corpus_v2/embeddings_v2_bgem3.npy
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py, stek_vague_detection.py (imported)
    Ollama running with all 5 LLMs pulled
"""

import json
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve, CITATION_PREFIX
from stek_vague_detection import is_vague_query_combined

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")
CHECKPOINT_PATH = Path("full_grid_results.jsonl")
FINAL_OUTPUT_PATH = Path("full_grid_results.json")

# Embedding models: name -> (sentence-transformers model id, embeddings file)
EMBEDDING_MODELS = {
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", Path("corpus/corpus_v2/embeddings_v2_e5base.npy")),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", Path("corpus/corpus_v2/embeddings_v2_e5large.npy")),
    "bge-m3": ("BAAI/bge-m3", Path("corpus/corpus_v2/embeddings_v2_bgem3.npy")),
}

LLM_MODELS = ["qwen2.5:32b", "mistral-small", "gemma2:27b", "command-r", "mistral"]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300
TOP_K = 5

SYSTEM_PROMPT = """Du bist ein Assistent für das STEK 2035 Heidelberg.
Beantworte die Frage NUR auf Basis der bereitgestellten Textauszüge.
Wenn die Antwort nicht enthalten ist, sage das klar."""


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


def check_ollama_model_available(model_name):
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any(model_name in m for m in models)
    except Exception:
        pass
    return False


def call_ollama(model_name, prompt):
    payload = {"model": model_name, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512}}
    start = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        elapsed_ms = int((time.time() - start) * 1000)
        if r.status_code == 200:
            return r.json().get("response", "").strip(), elapsed_ms
        return f"[Ollama error: status {r.status_code}]", elapsed_ms
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return f"[Ollama call failed: {e}]", elapsed_ms


def load_completed_keys():
    """Read the checkpoint file (if any) and return the set of
    (question_id, llm_model, embedding_model) combos already done."""
    completed = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed.add((rec["question_id"], rec["llm_model"], rec["embedding_model"]))
    return completed


def main():
    print("=" * 70)
    print("Full Grid Generation — All Questions x All Embeddings x All LLMs")
    print("=" * 70)

    missing_models = [m for m in LLM_MODELS if not check_ollama_model_available(m)]
    if missing_models:
        print(f"\n❌ Not available in Ollama: {missing_models}")
        print(f"   Pull them first before running.")
        return

    for name, (_, path) in EMBEDDING_MODELS.items():
        if not path.exists():
            print(f"\n❌ Missing embeddings for {name}: {path}")
            print(f"   Generate embeddings for ALL THREE models against your CURRENT")
            print(f"   (sentence-aware) chunks before running this — see the warning")
            print(f"   at the top of this script.")
            return

    print(f"\nLoading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    doc_ids = [c["document_id"] for c in chunks]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = gt_data["questions"]  # ALL 60 — including vague/unanswerable
    print(f"Loaded {len(questions)} questions (all categories)")

    completed = load_completed_keys()
    print(f"Already completed (from checkpoint): {len(completed)} combinations")

    total_combos = len(questions) * len(LLM_MODELS) * len(EMBEDDING_MODELS)
    print(f"Total combinations needed: {total_combos}")
    print(f"Remaining: {total_combos - len(completed)}\n")

    checkpoint_file = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    for emb_name, (emb_model_id, emb_path) in EMBEDDING_MODELS.items():
        print(f"\n{'='*70}\nEMBEDDING MODEL: {emb_name}\n{'='*70}")
        print(f"Loading embedding model: {emb_model_id}")
        embed_model = SentenceTransformer(emb_model_id)
        chunk_embeddings = load_normalised(emb_path)
        if chunk_embeddings.shape[0] != len(chunks):
            print(f"❌ Mismatch for {emb_name}: {chunk_embeddings.shape[0]} embeddings "
                  f"vs {len(chunks)} chunks — skipping this embedding model entirely.")
            continue

        authority = np.array([c.get("authority_level", 3) for c in chunks])
        official_idx = np.where(authority <= 3)[0]
        l4_idx = np.where(authority == 4)[0]
        l5_idx = np.where(authority == 5)[0]

        # precompute retrieval ONCE per question per embedding model — reused
        # across all 5 LLMs, since retrieval doesn't depend on the LLM
        print("Precomputing retrieval for all questions with this embedding model...")
        retrieval_cache = {}
        for q in questions:
            if is_vague_query_combined(q["question"]):
                retrieval_cache[q["q_id"]] = None  # vague — no retrieval needed
                continue
            result = cascading_retrieve(q["question"], chunks, chunk_embeddings, embed_model,
                                          official_idx, l4_idx, l5_idx)
            retrieval_cache[q["q_id"]] = result

        for llm in LLM_MODELS:
            print(f"\n--- LLM: {llm} (embedding: {emb_name}) ---")
            for q in questions:
                key = (q["q_id"], llm, emb_name)
                if key in completed:
                    continue

                retrieval = retrieval_cache[q["q_id"]]

                if retrieval is None:
                    # vague question — no retrieval, ask-for-clarification response
                    record = {
                        "question_id": q["q_id"], "question": q["question"],
                        "llm_model": llm, "embedding_model": emb_name,
                        "answer": "Ihre Frage ist sehr allgemein. Können Sie präzisieren, "
                                  "welches Thema Sie interessiert?",
                        "sources": [],
                        "metadata": {"response_time_ms": 0, "retrieved_chunks": 0, "verdict": "VAGUE"},
                    }
                else:
                    retrieved = [(chunks[i]["text"], chunks[i]["authority_level"], chunks[i]["document_id"],
                                    chunks[i].get("chunk_id"), float(retrieval["sims"][i]))
                                  for i in retrieval["result_idx"]]
                    context = "\n\n".join(text for text, _, _, _, _ in retrieved)
                    prompt = f"{SYSTEM_PROMPT}\n\n=== Kontext ===\n{context}\n\n=== Frage ===\n{q['question']}\n\n=== Antwort ==="

                    answer, elapsed_ms = call_ollama(llm, prompt)

                    sources = [{
                        "title": doc_id,
                        "page": chunk_id,  # no true page numbers in this corpus — chunk_id
                                             # is the closest real positional reference
                        "snippet": text[:200],
                        "url": None,  # internal documents have no real URLs
                    } for text, auth, doc_id, chunk_id, score in retrieved]

                    record = {
                        "question_id": q["q_id"], "question": q["question"],
                        "llm_model": llm, "embedding_model": emb_name,
                        "answer": answer, "sources": sources,
                        "metadata": {"response_time_ms": elapsed_ms,
                                      "retrieved_chunks": len(retrieved),
                                      "tier_used": retrieval["tier_used"],
                                      "verdict": "IN_DOMAIN"},
                    }

                checkpoint_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint_file.flush()
                completed.add(key)
                print(f"  {q['q_id']}: done ({len(completed)}/{total_combos})")

    checkpoint_file.close()

    print(f"\n{'='*70}")
    print("Converting checkpoint to final JSON array...")
    all_records = []
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(all_records)} records to {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
