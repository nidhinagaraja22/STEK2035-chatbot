# -*- coding: utf-8 -*-
"""
STEK 2035 Chatbot -- Embedding model x LLM model benchmark grid
==================================================================
Runs every combination of EMBEDDING_MODELS x LLM_MODELS over a fixed set
of representative German questions, and logs retrieval + generated
answers + timing for side-by-side comparison.

RUN THIS ON YOUR CLOUD INSTANCE (needs the real corpus + GPU + Ollama).

Prerequisites on the cloud instance, before running:
    pip install sentence-transformers
    ollama pull qwen2.5:32b
    ollama pull mistral
    ollama pull mistral-small
    ollama pull command-r
    ollama pull gemma2:27b

Run from the repo root (where vector_store/chunks.jsonl already exists):
    python benchmark_grid.py

Output:
    benchmark_results.json   -- full raw results (every combo x query)
    benchmark_summary.txt    -- readable summary tables
Both are also printed to stdout as the run progresses, so you can watch
it live and Ctrl+C early if needed (partial results are saved after each
embedding-model stage).
"""
import json
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
CHUNKS_PATH = Path("vector_store/chunks.jsonl")
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 5
OLLAMA_TIMEOUT = 600  # seconds per generation - generous margin in case this instance
                       # is running these 30b+ models on CPU (unconfirmed GPU access)

# Each embedding model gets its own query/passage prefixing rule, since not
# every model uses the e5-style "query: "/"passage: " convention.
EMBEDDING_MODELS = {
    "multilingual-e5-base": {
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "query_prefix": "",       # bge-m3 does not use the e5 prefix convention
        "passage_prefix": "",
    },
}

LLM_MODELS = ["qwen2.5:32b", "mistral", "mistral-small", "command-r", "gemma2:27b"]

QUESTIONS = [
    "Wie wird bezahlbarer Wohnraum in Heidelberg bis 2035 geschaffen?",
    "Welche Ziele verfolgt Heidelberg beim Klimaschutz und bei der Waermeplanung?",
    "Wie sieht die Buergerbeteiligung beim STEK 2035 aus?",
    "Welche Gruenflaechen und Freiraeume sollen erhalten oder neu geschaffen werden?",
    "Wie soll sich der oeffentliche Nahverkehr in Heidelberg entwickeln?",
    "Welche Massnahmen gibt es gegen Segregation und fuer soziale Durchmischung in den Stadtteilen?",
    "Wie plant die Stadt den Umgang mit wachsender Bevoelkerung und Flaechenverbrauch?",
    "Welche Rolle spielt die Wirtschaft und Unternehmensansiedlung im STEK 2035?",
    "Was sind die zentralen Ergebnisse der Zukunftsreise 2035?",
    "Wie werden Kultur und gesellschaftliche Vielfalt im STEK 2035 beruecksichtigt?",
]

# ---------------------------------------------------------------------------

def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(f"[Quelle: {c['origin']}]\n{c['text']}" for c in contexts)
    return f"""Du bist ein hilfreicher Assistent fuer das Stadtentwicklungskonzept Heidelberg 2035 (STEK 2035).
Beantworte die folgende Frage NUR auf Basis der bereitgestellten Kontextauszuege.
Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich.
Antworte auf Deutsch, klar und praezise.

Kontext:
{context_block}

Frage: {question}

Antwort:"""


def ask_ollama(model: str, prompt: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip(), time.time() - t0
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0


def main():
    print(f"Loading corpus from {CHUNKS_PATH} ...")
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(chunks)} chunks.\n")

    all_results = {}  # emb_label -> {"retrieval": {...}, "generation": {...}}

    for emb_label, cfg in EMBEDDING_MODELS.items():
        print(f"{'='*70}\nEmbedding model: {emb_label} ({cfg['name']})\n{'='*70}")
        model = SentenceTransformer(cfg["name"])

        def embed_passages(texts):
            inputs = [cfg["passage_prefix"] + t for t in texts]
            return model.encode(inputs, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=True).astype("float32")

        def embed_query(q):
            return model.encode(cfg["query_prefix"] + q, normalize_embeddings=True).astype("float32")

        print("Embedding full corpus ...")
        t0 = time.time()
        corpus_emb = embed_passages([c["text"] for c in chunks])
        print(f"Done in {time.time()-t0:.1f}s. Shape: {corpus_emb.shape}\n")

        retrieval_by_query = {}
        for q in QUESTIONS:
            qv = embed_query(q)
            scores = corpus_emb @ qv
            top_idx = np.argsort(-scores)[:TOP_K]
            contexts = [
                {"origin": chunks[i].get("origin", "unknown"),
                 "chunk_id": chunks[i].get("chunk_id", -1),
                 "text": chunks[i].get("text", ""),
                 "score": float(scores[i])}
                for i in top_idx
            ]
            retrieval_by_query[q] = contexts
            top_score = contexts[0]["score"] if contexts else 0.0
            print(f"  [{emb_label}] retrieved top-{TOP_K} for: {q[:60]}...  (top score={top_score:.3f})")

        del model  # free memory before next embedding model loads

        generation_by_llm = {}
        for llm in LLM_MODELS:
            print(f"\n  --- Generating with LLM: {llm} ---")
            per_query = {}
            for q in QUESTIONS:
                prompt = build_prompt(q, retrieval_by_query[q])
                reply, elapsed = ask_ollama(llm, prompt)
                per_query[q] = {"reply": reply, "seconds": round(elapsed, 1)}
                preview = reply[:100].replace("\n", " ")
                print(f"    [{llm}] {elapsed:5.1f}s | {q[:45]:45} -> {preview}...")
            generation_by_llm[llm] = per_query

        all_results[emb_label] = {
            "retrieval": retrieval_by_query,
            "generation": generation_by_llm,
        }

        # save partial results after every embedding-model stage
        Path("benchmark_results.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[Partial results saved to benchmark_results.json after '{emb_label}' stage]\n")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)

    lines.append("\n--- Avg top-1 retrieval score per embedding model ---")
    for emb_label, r in all_results.items():
        scores = [r["retrieval"][q][0]["score"] for q in QUESTIONS if r["retrieval"][q]]
        lines.append(f"  {emb_label:28} avg_top1_score={np.mean(scores):.4f}")

    lines.append("\n--- Avg generation time per LLM model (across all embedding stages) ---")
    for llm in LLM_MODELS:
        times = []
        for emb_label, r in all_results.items():
            times.extend(v["seconds"] for v in r["generation"][llm].values())
        lines.append(f"  {llm:20} avg_seconds={np.mean(times):.1f}  (n={len(times)})")

    lines.append("\nFull answers are in benchmark_results.json - open it and read")
    lines.append("the 'reply' text for each (embedding_model, llm_model, question)")
    lines.append("combination to judge actual answer quality yourself.")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    Path("benchmark_summary.txt").write_text(summary_text, encoding="utf-8")
    print("\nSaved: benchmark_results.json (full) + benchmark_summary.txt (summary)")


if __name__ == "__main__":
    main()
