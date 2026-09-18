"""
STEK 2035 — Corpus Boilerplate Cleaning (Exact-Match Only)
=============================================================
IMPORTANT — this script was rewritten after a scoring-based heuristic
for detecting "pure boilerplate" chunks produced false positives TWICE:
  1. A "Mehr dazu" link-count signal flagged ~20 genuinely substantive
     pages as navigation menus.
  2. Re-testing the same style of heuristic against this corpus flagged
     idx=640, 648, 655, 830 for removal — every one of which, on full-text
     reading, contains real substantive content (AK STEK composition,
     session summaries, vulnerable-groups discussion) despite opening
     with a title/date/DOKUMENTATION line.

Given this corpus repeatedly mixes real content with boilerplate-looking
openings, a generalizable score is not trustworthy here. This script
removes ONLY the 3 chunks individually verified by full-text reading to
be PURE boilerplate with zero narrative content — matched by exact,
distinctive substring:
    idx=795 (arbeitstreffen_2024)     — pure event cover page
    idx=432 (zukunft_gestalten_2024)  — pure event cover page
    idx=618 (zukunftsreise_2035)      — pure event cover page

Every other candidate checked (idx=0, 640, 645, 648, 655, 830) was found
to contain genuine content and is intentionally NOT removed.

Separately, this script also strips the recurring inline "ERSTELLT
DURCH...STADT HEIDELBERG" agency-credit phrase from chunks that
otherwise contain real content (verified: this phrase appears scattered
throughout ~527 chunks, mostly in online_beteiligung_2024, as a
PDF-extraction page-footer artifact) — these chunks are KEPT, just
cleaned of the recurring phrase.

Usage:
    python stek_clean_boilerplate.py

Reads:  corpus/corpus_v2/corpus_v2_chunks.jsonl (your CURRENT corpus —
        already has the zukunftsreise authority fix and second-wave
        cleaning applied)
Writes: corpus/corpus_v2/corpus_v2_chunks_fullyclean.jsonl
        boilerplate_cleaning_report_v2.json  (full audit trail)
"""

import json
import re
from pathlib import Path
from collections import Counter

INPUT_PATH  = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")             # your CURRENT corpus
OUTPUT_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")   # DISTINCT name — never
                                                                             # the same as INPUT_PATH
                                                                             # (this exact mistake
                                                                             # broke the ground truth
                                                                             # remap last time)
REPORT_PATH = Path("boilerplate_cleaning_report_v2.json")

# exact, distinctive substrings — one per hand-verified pure-boilerplate
# chunk. Each was confirmed by reading the FULL chunk text (not a
# preview) to contain zero narrative content beyond title/date/time/
# location/DOKUMENTATION/agency-credit.
PURE_BOILERPLATE_SIGNATURES = [
    "ARBEITSTREFFEN ZUM STADTENTWICKLUNGSKONZEPT (STEK) 2035 17. JUNI 2024",         # idx=795
    "ÖFFENTLICHE VERANSTALTUNG STADTENTWICKLUNGSKONZEPT (STEK) 2035 25. JUNI 2024",  # idx=432
    "Öffentliche Veranstaltung „Zukunftsreise Heidelberg – Start der Bürgerbeteiligung",  # idx=618
]

ERSTELLT_DURCH_PATTERN = re.compile(
    r"ERSTELLT\s+DURCH.*?STADT\s+HEIDELBERG", re.IGNORECASE | re.DOTALL
)


def should_remove(text: str) -> tuple[bool, str]:
    """Exact-match only — returns (should_remove, reason)."""
    for sig in PURE_BOILERPLATE_SIGNATURES:
        if sig in text:
            return True, "pure_boilerplate_verified"
    return False, ""


def clean_chunk_text(text: str) -> str:
    """Strip the inline ERSTELLT DURCH boilerplate phrase, keep the rest."""
    return ERSTELLT_DURCH_PATTERN.sub("", text).strip()


def main():
    print("=" * 70)
    print("STEK 2035 — Corpus Boilerplate Cleaning (Exact-Match Only)")
    print("=" * 70)

    if not INPUT_PATH.exists():
        print(f"\n❌ Input file not found: {INPUT_PATH}")
        print(f"   Adjust INPUT_PATH to point at your current corpus file.")
        return

    chunks = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"\nLoaded {len(chunks)} chunks from {INPUT_PATH}")

    removed = []
    cleaned_inline = []
    kept_unchanged = 0
    output_chunks = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        remove, reason = should_remove(text)

        if remove:
            removed.append({
                "idx": i,
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "reason": reason,
                "text_preview": text[:200],
            })
            continue  # do not include in output — fully removed

        has_erstellt_durch = bool(ERSTELLT_DURCH_PATTERN.search(text))
        if has_erstellt_durch:
            new_text = clean_chunk_text(text)
            cleaned_inline.append({
                "idx": i,
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "before_preview": text[:150],
                "after_preview": new_text[:150],
            })
            chunk = {**chunk, "text": new_text}
        else:
            kept_unchanged += 1

        output_chunks.append(chunk)

    # ── write cleaned corpus ──────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in output_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── write audit report ────────────────────────────────────────────────
    doc_removal_counts = Counter(r["document_id"] for r in removed)
    report = {
        "input_chunks": len(chunks),
        "output_chunks": len(output_chunks),
        "removed_count": len(removed),
        "inline_cleaned_count": len(cleaned_inline),
        "kept_unchanged_count": kept_unchanged,
        "removed_by_document": dict(doc_removal_counts),
        "removed_chunks": removed,
        "inline_cleaned_chunks": cleaned_inline,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Input chunks               : {len(chunks)}")
    print(f"  Removed (verified boilerplate) : {len(removed)}")
    print(f"  Inline-cleaned (kept)      : {len(cleaned_inline)}")
    print(f"  Unchanged                  : {kept_unchanged}")
    print(f"  Output chunks              : {len(output_chunks)}")

    print(f"\n  Removed chunks (should be exactly 3):")
    for r in removed:
        print(f"    idx={r['idx']} ({r['document_id']}): {r['text_preview'][:100]}")

    print(f"\n✅ Cleaned corpus saved: {OUTPUT_PATH}")
    print(f"✅ Audit report saved  : {REPORT_PATH}")
    print(f"\n⚠️  IMPORTANT: {len(removed)} chunks were REMOVED — this changes row")
    print(f"   count/order. You MUST re-embed this cleaned file to get a new")
    print(f"   embeddings.npy — the old one will no longer be row-aligned.")
    print(f"\n⚠️  You must also remap ground truth idx values before testing —")
    print(f"   use stek_remap_ground_truth.py with OLD_CHUNKS_PATH pointing at")
    print(f"   your CURRENT corpus_v2_chunks.jsonl (before this pass) and")
    print(f"   NEW_CHUNKS_PATH pointing at corpus_v2_chunks_fullyclean.jsonl.")


if __name__ == "__main__":
    main()
