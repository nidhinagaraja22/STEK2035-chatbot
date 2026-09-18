"""
STEK 2035 — Second-Wave Boilerplate Cleaning
================================================
The first cleaning pass (stek_clean_boilerplate.py) removed pure cover
pages and agendas. Diagnosing 19 remaining "ALL MISS" questions found a
SECOND, structurally different contamination pattern: generic STEK/MRO
explainer content that repeats "STEK 2035"/"Stadtentwicklungskonzept"
heavily, winning retrieval on keyword overlap alone.

IMPORTANT — a first draft of this script used "count of 'Mehr dazu'
(read more) links >= 3" as a navigation-menu detector. This was WRONG:
"Mehr dazu" is a ubiquitous heidelberg.de page-design pattern appearing
on genuinely substantive pages too (e.g. idx=1113, the confirmed-correct
Q003 answer with real housing percentages, has 4 "Mehr dazu" links
alongside real content). That draft would have deleted ~20 legitimate
chunks. Caught via spot-checking before running — do NOT reuse a
"Mehr dazu" count-based signal.

After manual verification of every candidate, the SAFE removal list is:
    idx=970  — website NAVIGATION SIDEBAR (matched via an exact,
               distinctive substring unique to this specific chunk's
               structure — NOT a generalizable link-count rule)
    idx=988, 989 — "STEK in einfacher Sprache" (Simple/Easy-Language
               accessibility version) — confirmed duplicate of content
               available in full detail elsewhere, using distinctive
               syllable-break punctuation ('·') specific to Leichte
               Sprache German (verified: only 2 matches in the entire
               corpus, both parts of this same document)
    idx=1041 — PRESS RELEASE about the STEK/MRO launch event (photo
               captions, official names, press logistics — ceremonial)
    idx=1046 — DISCRETIONARY: an off-topic city news article (youth
               climate grant program) mixed with pagination UI chrome —
               not STEK planning content, but has real (off-topic) facts

KEPT (confirmed genuine content, verified by full-text reading):
    idx=854, 990, 992, 1042, 1113 (and by extension, any other
    "Mehr dazu"-containing chunk not otherwise flagged)

Usage:
    python stek_clean_second_wave.py

Reads:  corpus/corpus_v2/corpus_v2_chunks.jsonl (your current corpus)
Writes: corpus/corpus_v2/corpus_v2_chunks.jsonl
        second_wave_cleaning_report.json (audit trail)
"""

import json
from pathlib import Path

INPUT_PATH  = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
OUTPUT_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")
REPORT_PATH = Path("second_wave_cleaning_report.json")

# exact, distinctive substring unique to idx=970's navigation sidebar —
# chosen because it's a specific consecutive sequence of short menu
# items that would not plausibly appear in genuine prose content
NAV_SIDEBAR_SIGNATURE = "Was ist das STEK? Mehr dazu Inhalte nach Clustern Mehr dazu"


def should_remove(text: str) -> tuple[bool, str]:
    """Returns (should_remove, reason) based on verified-safe signals only."""
    if NAV_SIDEBAR_SIGNATURE in text:
        return True, "navigation_sidebar"
    if "Pressetermin" in text and "(Foto:" in text:
        return True, "press_release_or_offtopic_news"
    if "·" in text:
        return True, "easy_language_duplicate"
    return False, ""


def main():
    print("=" * 70)
    print("STEK 2035 — Second-Wave Boilerplate Cleaning")
    print("=" * 70)

    if not INPUT_PATH.exists():
        print(f"\n❌ Input file not found: {INPUT_PATH}")
        return

    chunks = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"\nLoaded {len(chunks)} chunks from {INPUT_PATH}")

    removed = []
    output_chunks = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        remove, reason = should_remove(text)
        if remove:
            removed.append({
                "idx": i, "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"), "reason": reason,
                "text_preview": text[:200],
            })
            continue
        output_chunks.append(chunk)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in output_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "input_chunks": len(chunks),
            "output_chunks": len(output_chunks),
            "removed_count": len(removed),
            "removed_chunks": removed,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Removed: {len(removed)}")
    for r in removed:
        print(f"    idx={r['idx']} [{r['reason']}] ({r['document_id']}): {r['text_preview'][:80]}")
    print(f"  Output chunks: {len(output_chunks)}")

    print(f"\n✅ Cleaned corpus saved: {OUTPUT_PATH}")
    print(f"✅ Audit report saved  : {REPORT_PATH}")
    print(f"\n⚠️  {len(removed)} chunks removed — row count/order changed.")
    print(f"   You MUST re-embed this file (stek_reembed_cleaned_corpus.py,")
    print(f"   pointed at this new filename) AND remap ground truth idx")
    print(f"   values (stek_remap_ground_truth.py) before any further testing.")


if __name__ == "__main__":
    main()
