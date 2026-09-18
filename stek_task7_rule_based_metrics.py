"""
STEK 2035 — Task 7: Rule-Based Answer Evaluation Metrics (3 of 8 dimensions)
================================================================================
Implements the 3 dimensions that can be scored WITHOUT an LLM judge —
cheap, deterministic, and reusing infrastructure already built and
validated this session:

  3. Source Attribution Correctness — does the citation prefix shown
     to the user match the ACTUAL authority tier of the retrieved
     chunk it's attached to?
  6. Appropriate Abstention — does the system correctly abstain on
     genuinely vague/unanswerable questions, and correctly NOT
     abstain on genuinely answerable ones? (reuses the finalized
     Task 10 detector)
  8. Conciseness — is the generated answer's length reasonable
     relative to the expected answer's length, rather than padded
     or truncated?

The remaining 5 dimensions (Factual Accuracy, Groundedness,
Completeness, Relevance, Language Quality) require semantic judgment
and are scored separately via LLM-as-judge (stek_task7_llm_judge.py).

Usage:
    from stek_task7_rule_based_metrics import score_source_attribution, \
        score_abstention, score_conciseness
"""

from stek_vague_detection import is_vague_query_combined

CITATION_PREFIX = {
    1: "Laut dem offiziellen STEK 2035",
    2: "Laut dem städtischen Bericht",
    4: "Laut den Ergebnissen eines Stakeholder-Workshops (mit Bürgerinnen, Gemeinderat, Zivilgesellschaft und Verwaltung)",
    5: "Ein einzelner Bürger äußerte in der Online-Beteiligung den Wunsch",
}


def score_source_attribution(retrieved_chunks: list) -> dict:
    """
    retrieved_chunks: list of {"authority_level": int, "citation_used": str}
    for each chunk actually cited in the generated answer.

    Returns a score in [0, 1]: fraction of citations whose label
    correctly matches the chunk's REAL authority level — directly
    testing whether the misattribution problem (citizen opinion
    presented as official policy) is actually prevented in practice,
    not just in the citation-labeling CODE.
    """
    if not retrieved_chunks:
        return {"score": None, "correct": 0, "total": 0, "detail": "no citations to check"}

    correct = 0
    mismatches = []
    for c in retrieved_chunks:
        expected_prefix = CITATION_PREFIX.get(c["authority_level"], "Quelle unklar")
        if c["citation_used"].strip() == expected_prefix:
            correct += 1
        else:
            mismatches.append({"authority_level": c["authority_level"],
                                 "expected": expected_prefix, "actual": c["citation_used"]})

    return {"score": correct / len(retrieved_chunks), "correct": correct,
            "total": len(retrieved_chunks), "mismatches": mismatches}


def score_abstention(question: str, is_genuinely_answerable: bool, system_declined: bool) -> dict:
    """
    Checks TWO things against ground truth:
      1. Did the calibrated Task 10 detector correctly flag/not-flag
         this question as vague?
      2. Did the SYSTEM'S ACTUAL RESPONSE correctly decline or not
         decline, matching real answerability?

    These can disagree — e.g. the detector might correctly say
    "not vague" while the LLM still incorrectly declines to answer
    a real question (or vice versa) — both are worth tracking
    separately since they test different parts of the pipeline.
    """
    detector_flagged_vague = is_vague_query_combined(question)
    should_decline = not is_genuinely_answerable

    detector_correct = (detector_flagged_vague == should_decline) if should_decline or detector_flagged_vague else True
    # note: detector only targets VAGUE questions specifically — an
    # unanswerable-but-not-vague question correctly not being flagged
    # by the vague detector is NOT a detector error (see Task 10 scope)

    system_correct = (system_declined == should_decline)

    return {"should_decline": should_decline, "detector_flagged_vague": detector_flagged_vague,
            "system_declined": system_declined, "system_correct": system_correct,
            "score": 1.0 if system_correct else 0.0}


def score_conciseness(generated_answer: str, expected_answer: str,
                        min_ratio: float = 0.3, max_ratio: float = 3.0) -> dict:
    """
    Compares generated answer length to expected answer length.
    Neither too short (missing content) nor too long (padded/rambling)
    relative to what a complete answer should look like.

    Score: 1.0 if within [min_ratio, max_ratio] of expected length,
    degrading linearly outside that band, floored at 0.0.
    """
    if not expected_answer:
        return {"score": None, "detail": "no expected_answer to compare against"}

    gen_len = len(generated_answer.split())
    exp_len = len(expected_answer.split())
    if exp_len == 0:
        return {"score": None, "detail": "expected_answer has zero words"}

    ratio = gen_len / exp_len

    if min_ratio <= ratio <= max_ratio:
        score = 1.0
    elif ratio < min_ratio:
        score = max(0.0, ratio / min_ratio)
    else:
        score = max(0.0, max_ratio / ratio)

    return {"score": score, "generated_words": gen_len, "expected_words": exp_len, "ratio": round(ratio, 2)}


if __name__ == "__main__":
    print("=== Test 1: Source attribution — all correct ===")
    chunks_correct = [
        {"authority_level": 1, "citation_used": "Laut dem offiziellen STEK 2035"},
        {"authority_level": 5, "citation_used": "Ein einzelner Bürger äußerte in der Online-Beteiligung den Wunsch"},
    ]
    result = score_source_attribution(chunks_correct)
    print(result)
    assert result["score"] == 1.0
    print("✅ Correct\n")

    print("=== Test 2: Source attribution — one mismatch (the actual misattribution risk) ===")
    chunks_wrong = [
        {"authority_level": 5, "citation_used": "Laut dem offiziellen STEK 2035"},  # WRONG — citizen labeled as official
    ]
    result2 = score_source_attribution(chunks_wrong)
    print(result2)
    assert result2["score"] == 0.0
    print("✅ Correctly caught the misattribution\n")

    print("=== Test 3: Abstention — correct decline on vague question ===")
    result3 = score_abstention("Was plant Heidelberg?", is_genuinely_answerable=False, system_declined=True)
    print(result3)
    assert result3["score"] == 1.0
    print("✅ Correct\n")

    print("=== Test 4: Abstention — system WRONGLY declined a real question ===")
    result4 = score_abstention("Bis wann will Heidelberg klimaneutral sein?",
                                  is_genuinely_answerable=True, system_declined=True)
    print(result4)
    assert result4["score"] == 0.0
    print("✅ Correctly flagged as wrong\n")

    print("=== Test 5: Conciseness — reasonable length ===")
    result5 = score_conciseness("Die Stadt plant bis 2040 klimaneutral zu werden durch verschiedene Maßnahmen.",
                                  "Heidelberg will bis 2040 klimaneutral werden.")
    print(result5)
    assert result5["score"] == 1.0
    print("✅ Correct\n")

    print("=== Test 6: Conciseness — way too long (padded) ===")
    long_answer = " ".join(["Wort"] * 50)
    result6 = score_conciseness(long_answer, "Heidelberg will bis 2040 klimaneutral werden.")
    print(result6)
    assert result6["score"] < 1.0
    print("✅ Correctly penalized padding")