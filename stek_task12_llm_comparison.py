"""
STEK 2035 — Task 12: Final Model Benchmark (Actual Requirement)
=====================================================================
Compares all 5 LLMs on the SAME 8 required metrics, using the
SAME stabilized retrieval pipeline (threshold-gated cascade) for
every model, so the comparison isolates GENERATION quality/speed,
not retrieval differences:

  Qwen2.5:32B, Mistral Small, Gemma2:27B, Command-R, Mistral 7B

Required metrics, per model:
  Answer correctness    — LLM-judge (factual_accuracy, 1-5 scale)
  Faithfulness           — LLM-judge (groundedness, 1-5 scale)
  Authority correctness  — rule-based (citation label matches real
                            authority tier)
  Hallucination rate     — defined here as: factual_accuracy <= 2
                            AND groundedness <= 2 (both wrong AND
                            unsupported by context — a clear,
                            conservative hallucination signal)
  Abstention accuracy    — rule-based (correct decline/non-decline,
                            using the corrected, substantive-content-
                            aware detector)
  Average latency        — real, measured per-call generation time
  Latency SD             — standard deviation of the same
  Failure rate           — Ollama call failed, OR empty response,
                            OR judge parsing failed

Only after all 5 are compared does this script suggest a
"recommended production configuration" — never before seeing the
full comparison, per the requirement.

Usage:
    python stek_task12_llm_comparison.py [N]
    (N = number of questions per model; default 15 — this runs
    5 models x N questions x 2 calls each, so keep N modest unless
    you have real time to spend; 48 questions x 5 models = 480
    Ollama calls minimum)

Requires:
    corpus/corpus_v2/corpus_v2_chunks.jsonl (confirmed L4/L5-split)
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py, stek_task7_rule_based_metrics.py,
    stek_task7_llm_judge.py, stek_vague_detection.py,
    stek_task7_full_evaluation.py (imported)
    Ollama running with ALL 5 models pulled
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve, CITATION_PREFIX as CASCADE_CITATION_PREFIX
from stek_task7_rule_based_metrics import score_source_attribution, score_abstention
from stek_task7_llm_judge import judge_answer
from stek_task7_full_evaluation import is_genuine_decline

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

MODELS = ["qwen2.5:32b", "mistral-small:latest", "gemma2:27b", "command-r:latest", "mistral:latest"]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300

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


def check_model_available(model_name):
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any(model_name in m for m in models)
    except Exception:
        pass
    return False


def call_ollama(model_name, prompt, timeout=OLLAMA_TIMEOUT):
    payload = {"model": model_name, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512}}
    try:
        start = time.time()
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        elapsed = time.time() - start
        if r.status_code == 200:
            return r.json().get("response", "").strip(), elapsed, False
        return "", elapsed, True
    except Exception:
        return "", None, True


def main():
    n_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 15

    print("=" * 70)
    print(f"Task 12 — Final Model Benchmark ({len(MODELS)} models x {n_questions} questions)")
    print("=" * 70)

    missing_models = [m for m in MODELS if not check_model_available(m)]
    if missing_models:
        print(f"\n❌ Not available in Ollama: {missing_models}")
        print(f"   Pull them first (e.g. 'ollama pull mistral-small') before running.")
        return

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    online_bet_levels = {c.get("authority_level") for c in chunks if c["document_id"] == "online_beteiligung_2024"}
    if online_bet_levels != {5}:
        print(f"\n⚠️  WARNING: L4/L5 split looks wrong (online_beteiligung_2024 shows "
              f"{online_bet_levels}, expected {{5}}) — verify CHUNKS_PATH before trusting results.\n")

    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q for q in gt_data["questions"] if q.get("relevant_chunks")][:n_questions]
    print(f"\nEvaluating {len(questions)} questions x {len(MODELS)} models "
          f"({len(questions) * len(MODELS)} generations + judgments)\n")

    print("Retrieving context (same for all models — isolates generation differences)...")
    retrieved_context = {}
    for q in questions:
        result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model,
                                      official_idx, l4_idx, l5_idx)
        retrieved = [(chunks[i]["text"], chunks[i]["authority_level"]) for i in result["result_idx"]]
        context = "\n\n".join(text for text, _ in retrieved)
        cited_chunks = [{"authority_level": lvl, "citation_used": CASCADE_CITATION_PREFIX.get(lvl, "Quelle unklar")}
                         for _, lvl in retrieved]
        retrieved_context[q["q_id"]] = {"context": context, "cited_chunks": cited_chunks}

    all_model_results = {}

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        per_question = []
        for q in questions:
            ctx = retrieved_context[q["q_id"]]
            prompt = f"{SYSTEM_PROMPT}\n\n=== Kontext ===\n{ctx['context']}\n\n=== Frage ===\n{q['question']}\n\n=== Antwort ==="

            generated_answer, latency, gen_failed = call_ollama(model_name, prompt)

            attribution = score_source_attribution(ctx["cited_chunks"])
            system_declined = False
            if not gen_failed:
                system_declined = is_genuine_decline(generated_answer)
            abstention = score_abstention(q["question"], is_genuinely_answerable=True, system_declined=system_declined)

            judge_failed = False
            judge_scores = None
            if not gen_failed:
                judge_result = judge_answer(q["question"], ctx["context"], q.get("expected_answer", ""), generated_answer)
                judge_failed = not judge_result["success"]
                judge_scores = judge_result.get("scores")

            hallucination = False
            if judge_scores:
                hallucination = judge_scores.get("factual_accuracy", 5) <= 2 and judge_scores.get("groundedness", 5) <= 2

            failed = gen_failed or judge_failed

            latency_str = f"{latency:.1f}s" if latency else "FAILED"
            print(f"  {q['q_id']}: latency={latency_str}  attribution={attribution['score']}  failed={failed}")

            per_question.append({
                "q_id": q["q_id"], "latency": latency, "failed": failed,
                "attribution_score": attribution["score"], "abstention_score": abstention["score"],
                "judge_scores": judge_scores, "hallucination": hallucination,
            })

        valid = [p for p in per_question if not p["failed"]]
        latencies = [p["latency"] for p in valid if p["latency"] is not None]
        judge_valid = [p["judge_scores"] for p in valid if p["judge_scores"]]

        summary = {
            "answer_correctness": statistics.mean([j["factual_accuracy"] for j in judge_valid]) if judge_valid else None,
            "faithfulness": statistics.mean([j["groundedness"] for j in judge_valid]) if judge_valid else None,
            "authority_correctness": statistics.mean([p["attribution_score"] for p in valid]) if valid else None,
            "hallucination_rate": sum(1 for p in valid if p["hallucination"]) / len(valid) if valid else None,
            "abstention_accuracy": statistics.mean([p["abstention_score"] for p in valid]) if valid else None,
            "average_latency": statistics.mean(latencies) if latencies else None,
            "latency_sd": statistics.stdev(latencies) if len(latencies) > 1 else None,
            "failure_rate": sum(1 for p in per_question if p["failed"]) / len(per_question),
        }
        all_model_results[model_name] = {"summary": summary, "per_question": per_question}

        print(f"\n  Summary for {model_name}:")
        for k, v in summary.items():
            print(f"    {k}: {v:.3f}" if v is not None else f"    {k}: n/a")

    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON — ALL 5 MODELS")
    print(f"{'='*70}")
    metrics = ["answer_correctness", "faithfulness", "authority_correctness", "hallucination_rate",
                "abstention_accuracy", "average_latency", "latency_sd", "failure_rate"]
    header = f"{'Metric':<24}" + "".join(f"{m:<16}" for m in MODELS)
    print(header)
    for metric in metrics:
        row = f"{metric:<24}"
        for m in MODELS:
            v = all_model_results[m]["summary"][metric]
            row += f"{v:<16.3f}" if v is not None else f"{'n/a':<16}"
        print(row)

    with open("task12_llm_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: task12_llm_comparison_results.json")

    print(f"\n⚠️  No model is labelled 'recommended production configuration' by this")
    print(f"   script automatically — per the requirement, that judgment should be")
    print(f"   made only after reviewing this full comparison, weighing which metrics")
    print(f"   matter most for your deployment (e.g. is a small hallucination-rate")
    print(f"   difference worth a large latency cost?).")


if __name__ == "__main__":
    main()