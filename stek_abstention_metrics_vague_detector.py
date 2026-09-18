"""
STEK 2035 — Abstention Metrics Using the FINALIZED Vague Detector
======================================================================
Recomputes abstention accuracy using is_vague_query_combined() —
the calibrated, two-signal detector from Task 10 (6/6 vague caught,
0 false positives on 48 real answerable questions) — replacing the
earlier naive similarity-threshold approach, which was tested and
found to completely fail (F1=0.000: it never once triggered
abstention across 59 real questions).

IMPORTANT — honest scope: is_vague_query_combined() only detects
VAGUE questions (underspecified, no clear retrievable intent). It
does NOT detect UNANSWERABLE questions (specific, clear questions
whose answer genuinely doesn't exist in the corpus) — that remains
an open problem; no working general detector was found for it this
session (see the topic-concentration and top_score/spread negative
results). This script reports BOTH the overall abstention metrics
AND a breakdown by true category, so the vague-detection success and
the unanswerable-detection gap are both visible separately, not
averaged into one misleading number.

This needs NO embeddings or retrieval at all — the detector operates
purely on question text.

Usage:
    python stek_abstention_metrics_vague_detector.py

Requires:
    stek_task4_5_master_ground_truth.json
    stek_vague_detection.py (imported)
"""

import json
from pathlib import Path

from stek_vague_detection import is_vague_query_combined

GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")


def compute_abstention_metrics(results):
    tp = sum(1 for r in results if r["should_abstain"] and r["did_abstain"])
    tn = sum(1 for r in results if not r["should_abstain"] and not r["did_abstain"])
    fp = sum(1 for r in results if not r["should_abstain"] and r["did_abstain"])
    fn = sum(1 for r in results if r["should_abstain"] and not r["did_abstain"])

    accuracy = (tp + tn) / len(results) if results else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    print("=" * 70)
    print("Abstention Metrics — Using the Finalized Vague Detector")
    print("=" * 70)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    results = []
    for q in gt_data["questions"]:
        # should_abstain = ground truth: is this question genuinely
        # not answerable as-is (vague, unanswerable, OR semi_answerable)?
        should_abstain = q["category"] in ("vague", "unanswerable", "semi_answerable") \
                          or not q.get("answerable", True)
        if not should_abstain and not q.get("relevant_chunks"):
            continue  # skip anything with neither a clear answerable nor abstain label

        did_abstain = is_vague_query_combined(q["question"])

        marker = "✅" if should_abstain == did_abstain else "❌"
        print(f"{marker} {q['q_id']:<6} [{q['category']:<16}] "
              f"should_abstain={should_abstain}  did_abstain={did_abstain}")

        results.append({"q_id": q["q_id"], "category": q["category"],
                          "should_abstain": should_abstain, "did_abstain": did_abstain})

    overall = compute_abstention_metrics(results)

    print(f"\n{'='*70}")
    print("OVERALL (all categories combined)")
    print(f"{'='*70}")
    print(f"Confusion matrix: TP={overall['tp']} TN={overall['tn']} FP={overall['fp']} FN={overall['fn']}")
    print(f"Accuracy:  {overall['accuracy']:.4f}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1:        {overall['f1']:.4f}")

    # honest breakdown by TRUE category — separates the solved (vague)
    # problem from the unsolved (unanswerable) one, rather than averaging
    # them into one number that would misrepresent both
    print(f"\n{'='*70}")
    print("BREAKDOWN BY TRUE CATEGORY (why the overall number looks the way it does)")
    print(f"{'='*70}")
    for cat in ["vague", "unanswerable", "semi_answerable"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        caught = sum(1 for r in cat_results if r["did_abstain"])
        print(f"  {cat:<16}: {caught}/{len(cat_results)} correctly flagged for abstention")

    n_answerable = sum(1 for r in results if not r["should_abstain"])
    n_fp = sum(1 for r in results if not r["should_abstain"] and r["did_abstain"])
    print(f"  {'answerable':<16}: {n_answerable - n_fp}/{n_answerable} correctly NOT flagged "
          f"({n_fp} false positive{'s' if n_fp != 1 else ''})")

    print(f"\n⚠️  As expected: vague questions should show ~6/6 caught (Task 10's")
    print(f"   calibrated result). Unanswerable/semi_answerable questions are")
    print(f"   expected to show 0 caught — this detector was never designed to")
    print(f"   catch them, and no working detector for that category exists yet.")
    print(f"   This is why overall RECALL will look modest despite vague")
    print(f"   detection being fully solved — the two categories should be")
    print(f"   read separately, not blended into one verdict.")

    with open("abstention_metrics_vague_detector_results.json", "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_question": results}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: abstention_metrics_vague_detector_results.json")


if __name__ == "__main__":
    main()
