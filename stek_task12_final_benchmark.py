"""
STEK 2035 — Task 12: Final Comprehensive Benchmark (Runnable)
====================================================================
Synthesizes every real, already-computed result from this project
into one final benchmark — Task 12 is a SYNTHESIS task, not a new
measurement, so this script loads and aggregates the real JSON
outputs already produced by earlier scripts, rather than re-running
hours of LLM calls to recompute numbers you already have.

Usage:
    python stek_task12_final_benchmark.py

Reads (whichever of these exist on your vault — missing ones are
reported, not fatal):
    fair_strategy_comparison_policy.json
    fair_strategy_comparison_citizen_opinion.json
    policy_contamination_results.json
    task7_full_evaluation_results.json
    task7_analysis_summary.json
"""

import json
import statistics
from pathlib import Path

FILES = {
    "policy_strategy": Path("fair_strategy_comparison_policy.json"),
    "citizen_strategy": Path("fair_strategy_comparison_citizen_opinion.json"),
    "contamination": Path("policy_contamination_results.json"),
    "task7_results": Path("task7_full_evaluation_results.json"),
}


def load_if_exists(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    print("=" * 70)
    print("TASK 12 — FINAL COMPREHENSIVE BENCHMARK")
    print("=" * 70)

    data = {name: load_if_exists(path) for name, path in FILES.items()}
    missing = [name for name, d in data.items() if d is None]
    if missing:
        print(f"\n⚠️  Missing (will be skipped in synthesis): {missing}")
        print(f"   Run the corresponding earlier scripts first if you want these included.")

    # ── Section 1: Retrieval strategy comparison ─────────────────────────────
    print(f"\n{'='*70}")
    print("1. RETRIEVAL STRATEGY COMPARISON")
    print(f"{'='*70}")
    if data["policy_strategy"]:
        d = data["policy_strategy"]
        print(f"\nPolicy questions (n={d['n_questions']}):")
        for strat in ["plain_topk", "cascade", "authority_source_topic"]:
            if strat in d:
                print(f"  {strat:<25} MRR={d[strat]['mrr']:.4f}  "
                      f"R@5={d[strat]['metrics_by_k']['5']['recall'] if '5' in d[strat]['metrics_by_k'] else d[strat]['metrics_by_k'][5]['recall']:.4f}")
    else:
        print("  (not available — run stek_fair_strategy_comparison_segmented.py)")

    # ── Section 2: Misattribution safety ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("2. MISATTRIBUTION SAFETY (the deciding metric)")
    print(f"{'='*70}")
    if data["contamination"]:
        d = data["contamination"]
        print(f"  Plain top-k mean citizen-content fraction:  {d['plain_mean_citizen_fraction']:.4f}")
        print(f"  Cascade mean citizen-content fraction:      {d['cascade_mean_citizen_fraction']:.4f}")
        reduction = d['plain_mean_citizen_fraction'] - d['cascade_mean_citizen_fraction']
        print(f"  Absolute risk reduction:                    {reduction:.4f} "
              f"({reduction/d['plain_mean_citizen_fraction']*100:.1f}% relative reduction)")
    else:
        print("  (not available — run stek_test_policy_question_contamination.py)")

    # ── Section 3: Answer quality (Task 7) ───────────────────────────────────
    print(f"\n{'='*70}")
    print("3. ANSWER QUALITY — 8 DIMENSIONS")
    print(f"{'='*70}")
    if data["task7_results"]:
        results = data["task7_results"]
        n = len(results)
        print(f"  n={n} questions evaluated")

        attr = [r["rule_based"]["source_attribution"]["score"] for r in results
                if r["rule_based"]["source_attribution"]["score"] is not None]
        absten = [r["rule_based"]["abstention"]["score"] for r in results]
        concise = [r["rule_based"]["conciseness"]["score"] for r in results
                    if r["rule_based"]["conciseness"]["score"] is not None]
        print(f"\n  Rule-based:")
        print(f"    Source Attribution: {statistics.mean(attr):.3f}" if attr else "    Source Attribution: n/a")
        print(f"    Abstention:         {statistics.mean(absten):.3f}")
        print(f"    Conciseness:        {statistics.mean(concise):.3f}" if concise else "    Conciseness: n/a")

        valid = [r["llm_judge"]["scores"] for r in results if r["llm_judge"]["success"]]
        print(f"\n  LLM-judge ({len(valid)}/{n} valid):")
        if valid:
            for dim in ["factual_accuracy", "groundedness", "completeness", "relevance", "language_quality"]:
                vals = [j[dim] for j in valid if dim in j]
                if vals:
                    print(f"    {dim:<18}: {statistics.mean(vals):.2f}/5")

        # abstention misses, inline
        misses = [r for r in results if r["rule_based"]["abstention"]["score"] < 1.0]
        if misses:
            print(f"\n  Abstention misses ({len(misses)}): {[r['q_id'] for r in misses]}")
    else:
        print("  (not available — run stek_task7_full_evaluation.py 48)")

    # ── Final synthesis ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SYNTHESIS")
    print(f"{'='*70}")

    have_strategy = data["policy_strategy"] is not None
    have_safety = data["contamination"] is not None
    have_quality = data["task7_results"] is not None

    if have_strategy and have_safety:
        p = data["policy_strategy"]
        cascade_mrr = p["cascade"]["mrr"]
        plain_mrr = p["plain_topk"]["mrr"]
        gap = (plain_mrr - cascade_mrr) / plain_mrr * 100
        c = data["contamination"]
        print(f"\nOn your primary use case (policy questions):")
        print(f"  Cascade costs {gap:.1f}% MRR relative to unconstrained top-k...")
        print(f"  ...in exchange for reducing citizen-content contamination from "
              f"{c['plain_mean_citizen_fraction']*100:.1f}% to {c['cascade_mean_citizen_fraction']*100:.1f}%.")

    if have_quality:
        results = data["task7_results"]
        valid = [r["llm_judge"]["scores"] for r in results if r["llm_judge"]["success"]]
        if valid:
            dims = ["factual_accuracy", "groundedness", "completeness", "relevance", "language_quality"]
            means = {d: statistics.mean([j[d] for j in valid if d in j]) for d in dims}
            weakest = min(means, key=means.get)
            print(f"\nWeakest answer-quality dimension: {weakest} ({means[weakest]:.2f}/5)")

    print(f"\n{'✅ Full synthesis complete' if (have_strategy and have_safety and have_quality) else '⚠️  Partial synthesis — some components missing, see above'}")

    with open("task12_final_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump({"sections_available": {k: v is not None for k, v in data.items()}}, f, indent=2)
    print(f"\n✅ Saved: task12_final_benchmark_summary.json")


if __name__ == "__main__":
    main()
