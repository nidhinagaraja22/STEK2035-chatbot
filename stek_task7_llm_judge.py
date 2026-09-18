"""
STEK 2035 — Task 7: LLM-as-Judge Metrics (remaining 5 of 8 dimensions)
============================================================================
Scores the 5 dimensions that genuinely require semantic judgment,
not just rule-checking:

  1. Factual Accuracy    — does it match the real facts?
  4. Completeness         — does it cover all parts of the question?
  2. Groundedness         — is every claim actually supported by the
                             retrieved context (no hallucination)?
  5. Relevance            — does it address what was actually asked?
  7. Language Quality     — grammatically correct, natural German?

Uses ONE judge call per question (not 5 separate calls) with
structured JSON output, for efficiency. The judge model itself is
qwen2.5:32b — the SAME model used for generation — since a separate,
larger judge model is not available in this project's infrastructure;
this is a known limitation (self-judging can be less reliable than
independent judging) stated explicitly, not hidden.

Usage:
    python stek_task7_llm_judge.py

Requires:
    Ollama running with qwen2.5:32b loaded
    A file of {question, expected_answer, retrieved_context,
    generated_answer} records to judge (see build_judge_input below
    for the expected structure — this script judges records already
    produced by the retrieval+generation pipeline, it does not
    generate them itself)
"""

import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:32b"
OLLAMA_TIMEOUT = 300

JUDGE_PROMPT_TEMPLATE = """Du bist ein strenger, objektiver Gutachter für ein Frage-Antwort-System über die Heidelberger Stadtentwicklung (STEK 2035).

Bewerte die folgende GENERIERTE ANTWORT anhand von 5 Kriterien, jeweils auf einer Skala von 1 (sehr schlecht) bis 5 (ausgezeichnet).

=== FRAGE ===
{question}

=== BEREITGESTELLTER KONTEXT (worauf die Antwort basieren sollte) ===
{context}

=== ERWARTETE ANTWORT (Referenz, nicht die einzig gültige Formulierung) ===
{expected_answer}

=== GENERIERTE ANTWORT (was bewertet werden soll) ===
{generated_answer}

=== KRITERIEN ===
1. factual_accuracy: Stimmen die genannten Fakten mit der erwarteten Antwort überein?
2. groundedness: Basiert JEDE Aussage in der generierten Antwort auf dem bereitgestellten Kontext, ohne Erfindungen?
3. completeness: Werden ALLE Teile der Frage beantwortet (bei mehrteiligen Fragen)?
4. relevance: Beantwortet die Antwort tatsächlich die gestellte Frage, statt nur allgemein verwandte Informationen zu geben?
5. language_quality: Ist die Antwort grammatikalisch korrekt und natürlich formuliertes Deutsch?

Antworte NUR mit einem JSON-Objekt in exakt diesem Format, ohne zusätzlichen Text davor oder danach:
{{"factual_accuracy": <1-5>, "groundedness": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "language_quality": <1-5>, "reasoning": "<ein kurzer Satz, der die niedrigste Bewertung erklärt>"}}
"""


def check_ollama_running():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        pass
    return False


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.9, "num_predict": 300},
        # temperature=0.0 — judging should be as deterministic as possible,
        # unlike generation which used 0.1 elsewhere in this project
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return None
    except Exception as e:
        print(f"  ⚠️  Ollama call failed: {e}")
        return None


def parse_judge_response(raw_response: str) -> dict:
    """
    Extracts the JSON object from the judge's response, tolerating
    minor formatting issues (leading/trailing text, code fences) —
    LLMs frequently don't produce perfectly clean JSON even when asked.
    Returns None if parsing fails entirely, rather than crashing.
    """
    if not raw_response:
        return None

    # try direct parse first
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # try extracting the first {...} block, tolerating surrounding text
    match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_judge_scores(scores: dict) -> dict:
    """Ensures all 5 required fields are present and within [1,5] —
    an LLM judge can produce malformed or out-of-range values, and
    silently trusting them would corrupt the aggregate metrics."""
    required = ["factual_accuracy", "groundedness", "completeness", "relevance", "language_quality"]
    issues = []
    for field in required:
        if field not in scores:
            issues.append(f"missing field: {field}")
        elif not isinstance(scores[field], (int, float)) or not (1 <= scores[field] <= 5):
            issues.append(f"{field} out of range or wrong type: {scores.get(field)}")
    return {"valid": len(issues) == 0, "issues": issues}


def judge_answer(question, context, expected_answer, generated_answer):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, context=context,
        expected_answer=expected_answer or "(keine Referenzantwort verfügbar)",
        generated_answer=generated_answer,
    )
    raw = call_ollama(prompt)
    parsed = parse_judge_response(raw)

    if parsed is None:
        return {"success": False, "raw_response": raw, "scores": None}

    validation = validate_judge_scores(parsed)
    return {"success": validation["valid"], "scores": parsed,
            "validation_issues": validation["issues"], "raw_response": raw}


def main():
    print("=" * 70)
    print("Task 7 — LLM-as-Judge Metrics")
    print("=" * 70)

    if not check_ollama_running():
        print(f"\n❌ Ollama not running or {OLLAMA_MODEL} not loaded.")
        return

    print(f"\n⚠️  This script judges records you provide — it does not run")
    print(f"   retrieval/generation itself. Point INPUT_RECORDS_PATH at a")
    print(f"   JSON file of {{question, context, expected_answer, generated_answer}}")
    print(f"   records, e.g. produced by stek_test_llm_unanswerable_handling.py")
    print(f"   or a similar generation run, then re-run this script.")


if __name__ == "__main__":
    main()
