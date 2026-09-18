"""
STEK 2035 — Task 9: Verify the Speed Claim With Repeated Trials + SD
========================================================================
The original finding ("qwen2.5:32b + e5-large is ~4.5x faster than
qwen2.5:32b + e5-base") was based on a SINGLE run per configuration —
27.9s vs 6.2s average, with e5-base spiking to 216s on some queries.
A single run cannot distinguish a real, reproducible effect from
noise (server load, one unusually slow/fast generation, etc.).

This script re-runs the SAME comparison with N repeated trials per
question per embedding model, and reports mean ± standard deviation
for both — the correct basis for a "Xx faster" claim.

Usage:
    python stek_verify_speed_claim.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/embeddings_v2_e5large.npy   (adjust filename if different)
    Ollama running with qwen2.5:32b loaded
    sentence-transformers, requests, numpy

Outputs:
    speed_verification_results.json   — every individual trial, raw
    speed_verification_summary.txt    — mean ± SD comparison table
"""

import json
import time
import statistics
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────
CORPUS_DIR  = Path("corpus/corpus_v2")
CHUNKS_PATH = CORPUS_DIR / "corpus_v2_chunks_l4l5split.jsonl"  # matches what
                                                                  # e5-large embeddings
                                                                  # were generated from

EMBEDDING_MODELS = {
    "e5-base":  {"emb_path": CORPUS_DIR / "embeddings_v2_e5base.npy",
                 "model_name": "intfloat/multilingual-e5-base", "prefix": "query: "},
    "e5-large": {"emb_path": CORPUS_DIR / "embeddings_v2_e5large.npy",
                 "model_name": "intfloat/multilingual-e5-large", "prefix": "query: "},
    "bge-m3":   {"emb_path": CORPUS_DIR / "embeddings_v2_bgem3.npy",
                 "model_name": "BAAI/bge-m3", "prefix": ""},  # BGE-M3 does NOT use
                                                                 # e5-style prefixes —
                                                                 # confirmed when
                                                                 # generating its
                                                                 # embeddings earlier
}

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:32b"
OLLAMA_TIMEOUT = 300

N_TRIALS = 5  # repeated trials per question per embedding model — this is
              # the core fix: the original claim used N_TRIALS=1

TOP_K = 5

# same 10 benchmark questions used in the original single-run comparison,
# for direct comparability
TEST_QUESTIONS = [
    "Wie wird bezahlbarer Wohnraum in Heidelberg bis 2035 geschaffen?",
    "Welche Ziele verfolgt Heidelberg beim Klimaschutz?",
    "Wie sieht die Bürgerbeteiligung beim STEK 2035 aus?",
    "Welche Grünflächen sollen erhalten oder neu geschaffen werden?",
    "Wie soll sich der öffentliche Nahverkehr entwickeln?",
    "Welche Maßnahmen gibt es gegen Segregation?",
    "Wie plant die Stadt mit wachsender Bevölkerung?",
    "Welche Rolle spielt die Wirtschaft im STEK 2035?",
    "Was sind die zentralen Ergebnisse der Zukunftsreise 2035?",
    "Wie werden Kultur und Vielfalt im STEK 2035 berücksichtigt?",
]

SYSTEM_PROMPT = """Du bist ein Assistent für das STEK 2035 Heidelberg.
Beantworte die Frage NUR auf Basis der bereitgestellten Textauszüge.
Wenn die Antwort nicht enthalten ist, sage das klar."""


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                chunks.append(obj.get("text", ""))
    return chunks


def load_normalised(path: Path) -> np.ndarray:
    arr = np.load(path)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms


def retrieve_top_k(question, model, chunk_embeddings, chunks, prefix, k=TOP_K):
    q_vec = model.encode(f"{prefix}{question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec
    top_idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in top_idx]


def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"{SYSTEM_PROMPT}\n\n=== Kontext ===\n{context}\n\n=== Frage ===\n{question}\n\n=== Antwort ==="


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512},
    }
    start = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        elapsed = time.time() - start
        if r.status_code == 200:
            return {"success": True, "seconds": elapsed, "tokens": r.json().get("eval_count", 0)}
        return {"success": False, "seconds": elapsed, "tokens": 0}
    except Exception as e:
        return {"success": False, "seconds": time.time() - start, "tokens": 0, "error": str(e)}


def check_ollama_running():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        pass
    return False


def main():
    print("=" * 70)
    print("STEK 2035 — Task 9: Speed Claim Verification (Repeated Trials)")
    print("=" * 70)

    if not check_ollama_running():
        print(f"\n❌ Ollama not running or {OLLAMA_MODEL} not loaded — start it first.")
        return

    print(f"\nLoading chunks from: {CHUNKS_PATH}")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks")

    all_results = {}

    for model_key, model_info in EMBEDDING_MODELS.items():
        emb_path = model_info["emb_path"]
        if not emb_path.exists():
            print(f"\n⚠️  SKIPPED {model_key} — embeddings file not found: {emb_path}")
            print(f"    Adjust EMBEDDING_MODELS['{model_key}']['emb_path'] to your actual filename.")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {model_key}")
        print(f"{'='*70}")

        embed_model = SentenceTransformer(model_info["model_name"])
        chunk_embeddings = load_normalised(emb_path)
        if chunk_embeddings.shape[0] != len(chunks):
            print(f"❌ Row mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — skipping {model_key}.")
            continue

        model_times = []  # every individual trial's generation time, flat list
        per_question_results = []

        for qi, question in enumerate(TEST_QUESTIONS):
            print(f"\n  [{qi+1}/{len(TEST_QUESTIONS)}] {question[:50]}...")
            retrieved = retrieve_top_k(question, embed_model, chunk_embeddings, chunks, model_info["prefix"])
            prompt = build_prompt(question, retrieved)

            trial_times = []
            for trial in range(N_TRIALS):
                result = call_ollama(prompt)
                if result["success"]:
                    trial_times.append(result["seconds"])
                    model_times.append(result["seconds"])
                    print(f"      trial {trial+1}/{N_TRIALS}: {result['seconds']:.2f}s")
                else:
                    print(f"      trial {trial+1}/{N_TRIALS}: FAILED ({result.get('error', 'unknown')})")

            if trial_times:
                per_question_results.append({
                    "question": question,
                    "trial_times": trial_times,
                    "mean": statistics.mean(trial_times),
                    "stdev": statistics.stdev(trial_times) if len(trial_times) > 1 else 0.0,
                    "min": min(trial_times), "max": max(trial_times),
                })

            # save incrementally
            all_results[model_key] = {
                "per_question": per_question_results,
                "all_trial_times": model_times,
            }
            with open("speed_verification_results.json", "w", encoding="utf-8") as f:
                json.dump({"n_trials": N_TRIALS, "results": all_results}, f, ensure_ascii=False, indent=2)

    # ── final comparison ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON — Mean ± SD Across All Trials")
    print(f"{'='*70}")

    summary_lines = []
    summary_lines.append(f"Task 9 — Speed Claim Verification ({N_TRIALS} repeated trials per question)\n")
    summary_lines.append("=" * 70)

    model_means = {}
    for model_key, data in all_results.items():
        times = data["all_trial_times"]
        if not times:
            continue
        mean_t = statistics.mean(times)
        stdev_t = statistics.stdev(times) if len(times) > 1 else 0.0
        max_t = max(times)
        model_means[model_key] = mean_t

        line = (f"{model_key:<12} mean={mean_t:.2f}s  stdev={stdev_t:.2f}s  "
                f"max={max_t:.2f}s  n_trials={len(times)}")
        print(f"  {line}")
        summary_lines.append(line)

    if len(model_means) >= 2:
        print(f"\nPairwise speed ratios:")
        summary_lines.append("\nPairwise speed ratios:")
        model_keys = list(model_means.keys())
        for i in range(len(model_keys)):
            for j in range(i + 1, len(model_keys)):
                a, b = model_keys[i], model_keys[j]
                ratio = model_means[a] / model_means[b]
                line = f"  {a} / {b} = {ratio:.2f}x"
                print(line)
                summary_lines.append(line)

        # specifically flag the original e5-base/e5-large claim if both are present
        if "e5-base" in model_means and "e5-large" in model_means:
            speedup = model_means["e5-base"] / model_means["e5-large"]
            line = f"\nOriginal claim check (e5-base / e5-large): measured {speedup:.2f}x vs. original single-run claim of 4.5x — {'CONFIRMED within reasonable range' if 3.0 <= speedup <= 6.0 else 'DOES NOT MATCH — original claim may have been noise-driven'}"
            print(line)
            summary_lines.append(line)

        fastest = min(model_means, key=model_means.get)
        slowest = max(model_means, key=model_means.get)
        line = f"\nFastest: {fastest} ({model_means[fastest]:.2f}s avg)  |  Slowest: {slowest} ({model_means[slowest]:.2f}s avg)"
        print(line)
        summary_lines.append(line)

    with open("speed_verification_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n✅ Full results: speed_verification_results.json")
    print(f"✅ Summary: speed_verification_summary.txt")


if __name__ == "__main__":
    main()