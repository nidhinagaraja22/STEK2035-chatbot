"""
STEK 2035 — Task 7: Full 8-Dimension Answer Evaluation, End-to-End
========================================================================
Generates real answers (retrieve + qwen2.5:32b) for a sample of
questions, then scores all 8 dimensions:

  Rule-based (from stek_task7_rule_based_metrics.py):
    3. Source Attribution Correctness
    6. Appropriate Abstention
    8. Conciseness

  LLM-as-judge (from stek_task7_llm_judge.py):
    1. Factual Accuracy
    2. Groundedness
    4. Completeness
    5. Relevance
    7. Language Quality

Uses the threshold-gated cascade for retrieval — the strategy
actually selected for deployment in this project (chosen for
misattribution-prevention reasons, despite plain top-k scoring
higher on pure retrieval accuracy — see the fair strategy
comparison report for that full discussion).

Usage:
    python stek_task7_full_evaluation.py [N]
    (N = number of questions to evaluate; default 10, since each
    question needs 2 real LLM calls — generation + judging — and
    this will take real time)

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
    stek_cascading_retrieval.py, stek_vague_detection.py,
    stek_task7_rule_based_metrics.py, stek_task7_llm_judge.py (imported)
    Ollama running with qwen2.5:32b loaded
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from stek_cascading_retrieval import cascading_retrieve, CITATION_PREFIX as CASCADE_CITATION_PREFIX
from stek_task7_rule_based_metrics import score_source_attribution, score_abstention, score_conciseness
from stek_task7_llm_judge import judge_answer, check_ollama_running

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")  # confirmed correct:
                                                                  # authority dist
                                                                  # {1:116,2:476,3:75,
                                                                  # 4:159,5:432}
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:32b"
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


def call_ollama_generate(prompt):
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return f"[Ollama error: status {r.status_code}]"
    except Exception as e:
        return f"[Ollama call failed: {e}]"


DECLINE_PHRASES = ["nicht enthalten", "keine information", "nicht beantwort",
                     "nicht angegeben", "nicht bekannt", "kann ich nicht", "liegen keine"]


def is_genuine_decline(generated_answer: str) -> bool:
    """A genuine decline means the answer is SUBSTANTIVELY empty —
    not merely that a decline-sounding phrase appears anywhere in an
    otherwise-substantive response. A model correctly answering part
    of a multi-part question while honestly flagging a gap for
    another part (e.g. "solar: 22 hectares; wind: not specified in
    the source") is NOT a decline — it's the honest, granular
    behavior this system should exhibit, and scoring it as a false
    refusal would penalize exactly the right behavior.

    Found via a real case: Q061 correctly answered the solar-energy
    half of a two-part question with a specific figure, while
    honestly noting the wind-energy figure wasn't in the source —
    the naive "contains a decline phrase" check wrongly flagged this
    entire, mostly-correct answer as a full refusal.

    Heuristic: only treat it as a decline if a decline phrase is
    present AND the answer lacks real numeric content — a proxy for
    "this is substantively empty" rather than "this is a full
    answer with one honestly-flagged gap.\""""
    text_lower = generated_answer.lower()
    has_decline_phrase = any(p in text_lower for p in DECLINE_PHRASES)
    if not has_decline_phrase:
        return False

    digit_groups = re.findall(r'\d+', generated_answer)
    word_count = len(generated_answer.split())

    if len(digit_groups) >= 1 and word_count > 40:
        return False  # substantive, specific answer — not a genuine decline

    return True


def main():
    n_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("=" * 70)
    print(f"Task 7 — Full 8-Dimension Evaluation ({n_questions} questions)")
    print("=" * 70)

    if not check_ollama_running():
        print(f"\n❌ Ollama not running or {OLLAMA_MODEL} not loaded.")
        return

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    authority = np.array([c.get("authority_level", 3) for c in chunks])
    official_idx = np.where(authority <= 3)[0]
    l4_idx = np.where(authority == 4)[0]
    l5_idx = np.where(authority == 5)[0]

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    answerable = [q for q in gt_data["questions"] if q.get("relevant_chunks")][:n_questions]
    print(f"\nEvaluating {len(answerable)} questions end-to-end (generation + full 8-dimension scoring)\n")

    all_results = []
    for q in answerable:
        print(f"\n{'='*70}")
        print(f"{q['q_id']}: {q['question']}")
        print(f"{'='*70}")

        result = cascading_retrieve(q["question"], chunks, chunk_embeddings, model,
                                      official_idx, l4_idx, l5_idx)
        retrieved = [(chunks[i]["text"], chunks[i]["authority_level"]) for i in result["result_idx"]]
        context = "\n\n".join(text for text, _ in retrieved)

        prompt = f"{SYSTEM_PROMPT}\n\n=== Kontext ===\n{context}\n\n=== Frage ===\n{q['question']}\n\n=== Antwort ==="
        generated_answer = call_ollama_generate(prompt)
        print(f"Generated: {generated_answer[:150]}...")

        system_declined = is_genuine_decline(generated_answer)

        # ── Rule-based dimensions ────────────────────────────────────────────
        cited_chunks = [{"authority_level": lvl, "citation_used": CASCADE_CITATION_PREFIX.get(lvl, "Quelle unklar")}
                         for _, lvl in retrieved]
        attribution = score_source_attribution(cited_chunks)
        abstention = score_abstention(q["question"], is_genuinely_answerable=True, system_declined=system_declined)
        conciseness = score_conciseness(generated_answer, q.get("expected_answer", ""))

        # ── LLM-judge dimensions ─────────────────────────────────────────────
        judge_result = judge_answer(q["question"], context, q.get("expected_answer", ""), generated_answer)

        print(f"Rule-based: attribution={attribution['score']}  abstention={abstention['score']}  "
              f"conciseness={conciseness['score']}")
        if judge_result["success"]:
            print(f"LLM-judge: {judge_result['scores']}")
        else:
            print(f"⚠️  LLM-judge FAILED to parse: {judge_result.get('validation_issues', judge_result.get('raw_response'))}")

        all_results.append({
            "q_id": q["q_id"], "question": q["question"], "generated_answer": generated_answer,
            "rule_based": {"source_attribution": attribution, "abstention": abstention, "conciseness": conciseness},
            "llm_judge": judge_result,
        })

    with open("task7_full_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: task7_full_evaluation_results.json")

    # aggregate summary
    print(f"\n{'='*70}")
    print("AGGREGATE (rule-based dimensions)")
    print(f"{'='*70}")
    attr_scores = [r["rule_based"]["source_attribution"]["score"] for r in all_results
                    if r["rule_based"]["source_attribution"]["score"] is not None]
    absten_scores = [r["rule_based"]["abstention"]["score"] for r in all_results]
    concise_scores = [r["rule_based"]["conciseness"]["score"] for r in all_results
                        if r["rule_based"]["conciseness"]["score"] is not None]
    print(f"Source Attribution: mean={np.mean(attr_scores):.3f}" if attr_scores else "Source Attribution: no data")
    print(f"Abstention:         mean={np.mean(absten_scores):.3f}")
    print(f"Conciseness:        mean={np.mean(concise_scores):.3f}" if concise_scores else "Conciseness: no data")

    valid_judgments = [r["llm_judge"]["scores"] for r in all_results if r["llm_judge"]["success"]]
    print(f"\n{'='*70}")
    print(f"AGGREGATE (LLM-judge dimensions) — {len(valid_judgments)}/{len(all_results)} valid judgments")
    print(f"{'='*70}")
    if valid_judgments:
        for dim in ["factual_accuracy", "groundedness", "completeness", "relevance", "language_quality"]:
            vals = [j[dim] for j in valid_judgments if dim in j]
            print(f"{dim}: mean={np.mean(vals):.2f}/5" if vals else f"{dim}: no data")


if __name__ == "__main__":
    main()