"""
STEK 2035 — Identify the Abstention Misses from the Full Task 7 Run
========================================================================
The full 48-question Task 7 run showed abstention mean=0.917, down
from 1.000 on the 10-question sample. This pulls out exactly which
questions caused that drop, and why — either the system wrongly
declined a real question, or wrongly did NOT decline one it should have.

Usage:
    python stek_identify_abstention_misses.py

Requires:
    task7_full_evaluation_results.json (produced by
    stek_task7_full_evaluation.py's 48-question run)
"""

import json
from pathlib import Path

RESULTS_PATH = Path("task7_full_evaluation_results.json")


def main():
    if not RESULTS_PATH.exists():
        print(f"❌ {RESULTS_PATH} not found — run stek_task7_full_evaluation.py 48 first.")
        return

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    print("=" * 70)
    print("Abstention Misses — Full 48-Question Task 7 Run")
    print("=" * 70)

    misses = [r for r in results if r["rule_based"]["abstention"]["score"] < 1.0]
    print(f"\n{len(misses)} question(s) with incorrect abstention behavior:\n")

    for r in misses:
        ab = r["rule_based"]["abstention"]
        print(f"  {r['q_id']}: {r['question']}")
        print(f"    should_decline={ab['should_decline']}  system_declined={ab['system_declined']}")
        if ab["should_decline"] and not ab["system_declined"]:
            print(f"    ⚠️  Should have DECLINED (unanswerable/vague) but ANSWERED instead")
            print(f"       — hallucination risk: check the generated answer below")
        elif not ab["should_decline"] and ab["system_declined"]:
            print(f"    ⚠️  Should have ANSWERED (genuinely answerable) but DECLINED instead")
            print(f"       — false refusal: denies a real user a real answer")
        print(f"    Generated answer: {r['generated_answer'][:200]}...")
        print()

    if not misses:
        print("  None found — abstention score of <1.0 may come from a different")
        print("  cause; check the raw results file directly.")


if __name__ == "__main__":
    main()
