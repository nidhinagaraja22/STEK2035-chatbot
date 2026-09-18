"""
STEK 2035 — Remap Ground Truth After Corpus Cleaning
========================================================
Boilerplate cleaning (stek_clean_boilerplate.py) REMOVED chunks from the
corpus, which shifts every subsequent chunk's absolute row position
(idx). The ground truth file (stek_task4_5_master_ground_truth.json)
stores idx values computed against the PRE-cleaning corpus — these are
now silently wrong for any chunk located after a removed one, pointing
at whatever chunk randomly ended up in that position after the shift.

Confirmed concretely: Q021's ground truth idx=811 pointed at Kultur
content before cleaning, but at unrelated Entsiegelung/climate content
after cleaning — same number, shifted meaning.

Fix: chunk_uid (e.g. "website_html::95567::0") is a STABLE, FULLY UNIQUE
identifier per chunk that does not change when OTHER chunks are removed —
confirmed unique across all 1,266 chunks in the corpus. This script
remaps every ground truth idx by looking up its chunk_uid in the OLD
corpus, then finding that same chunk_uid's NEW position in the cleaned
corpus.

NOTE: an earlier version of this script used (document_id, chunk_id) as
the key, but this is AMBIGUOUS for website_html specifically — that
document_id bundles ~90 different scraped web pages, each with its own
internal chunk_id numbering restarting from 0, so (website_html, 0)
matches 88 different actual chunks. chunk_uid does not have this
problem since it embeds the source URL as well.

Usage:
    python stek_remap_ground_truth.py

Reads:
    corpus/corpus_v2/corpus_v2_chunks.jsonl           (OLD corpus — your
                                              current state, pre-this-
                                              cleaning-pass, matches
                                              existing ground truth idx)
    corpus/corpus_v2/corpus_v2_chunks_v2cleaned.jsonl  (NEW corpus —
                                              output of
                                              stek_clean_second_wave.py)
    stek_task4_5_master_ground_truth.json  (ground truth to fix)

Writes:
    stek_task4_5_master_ground_truth_remapped.json  (corrected idx values)
    remap_report.json                                (audit trail —
                                              what changed, and any
                                              chunks that could not be
                                              remapped because they were
                                              REMOVED during cleaning)
"""

import json
from pathlib import Path

OLD_CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl")           # your current corpus,
                                                                              # BEFORE second-wave cleaning
NEW_CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks.jsonl") # output of
                                                                              # stek_clean_second_wave.py
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

OUTPUT_GT_PATH = Path("stek_task4_5_master_ground_truth.json")
REPORT_PATH = Path("remap_report.json")


def load_chunks_with_idx(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_idx"] = i
            chunks.append(obj)
    return chunks


def main():
    print("=" * 70)
    print("STEK 2035 — Remap Ground Truth After Corpus Cleaning")
    print("=" * 70)

    for p in [OLD_CHUNKS_PATH, NEW_CHUNKS_PATH, GROUND_TRUTH_PATH]:
        if not p.exists():
            print(f"\n❌ Required file not found: {p}")
            return

    print(f"\nLoading OLD corpus (pre-cleaning): {OLD_CHUNKS_PATH}")
    old_chunks = load_chunks_with_idx(OLD_CHUNKS_PATH)
    print(f"  {len(old_chunks)} chunks")

    print(f"Loading NEW corpus (post-cleaning): {NEW_CHUNKS_PATH}")
    new_chunks = load_chunks_with_idx(NEW_CHUNKS_PATH)
    print(f"  {len(new_chunks)} chunks  ({len(old_chunks) - len(new_chunks)} removed)")

    # build old_idx -> chunk_uid lookup
    old_idx_to_key = {}
    for c in old_chunks:
        old_idx_to_key[c["_idx"]] = c.get("chunk_uid")

    # build chunk_uid -> new_idx lookup
    new_key_to_idx = {}
    for c in new_chunks:
        new_key_to_idx[c.get("chunk_uid")] = c["_idx"]

    # sanity check: chunk_uid must be unique in both files, or this
    # remap is unsafe (confirmed unique for the real STEK corpus, but
    # verify again here in case this script is reused on different data)
    from collections import Counter
    key_counts = Counter(c.get("chunk_uid") for c in new_chunks)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    if duplicates:
        print(f"\n⚠️  WARNING: {len(duplicates)} chunk_uid values are NOT unique in")
        print(f"   the new corpus — remapping may be ambiguous for these. chunk_uid")
        print(f"   was confirmed unique for the original STEK corpus; if this fires,")
        print(f"   something about the cleaned file's chunk_uid field is unexpected.")

    print(f"\nLoading ground truth: {GROUND_TRUTH_PATH}")
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    remapped_count = 0
    unchanged_count = 0
    removed_count = 0
    remap_log = []

    for q in gt_data["questions"]:
        for chunk_ref in q.get("relevant_chunks", []):
            old_idx = chunk_ref["idx"]
            old_key = old_idx_to_key.get(old_idx)  # this is now a chunk_uid string

            if old_key is None:
                remap_log.append({
                    "q_id": q["q_id"], "old_idx": old_idx, "status": "OLD_IDX_NOT_FOUND",
                })
                continue

            new_idx = new_key_to_idx.get(old_key)

            if new_idx is None:
                # this chunk was ITSELF removed during boilerplate cleaning —
                # a real problem: the cited relevant chunk no longer exists
                removed_count += 1
                remap_log.append({
                    "q_id": q["q_id"], "old_idx": old_idx, "chunk_uid": old_key,
                    "document_id": chunk_ref.get("document_id"),
                    "status": "REMOVED_DURING_CLEANING — GROUND TRUTH CHUNK NO LONGER EXISTS",
                })
                continue

            if new_idx != old_idx:
                remap_log.append({
                    "q_id": q["q_id"], "old_idx": old_idx, "new_idx": new_idx,
                    "chunk_uid": old_key, "document_id": chunk_ref.get("document_id"),
                    "status": "REMAPPED",
                })
                chunk_ref["idx"] = new_idx
                remapped_count += 1
            else:
                unchanged_count += 1

    with open(OUTPUT_GT_PATH, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_relevant_chunk_refs": remapped_count + unchanged_count + removed_count,
            "remapped": remapped_count,
            "unchanged": unchanged_count,
            "ground_truth_chunks_removed_during_cleaning": removed_count,
            "log": remap_log,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Remapped (idx changed)     : {remapped_count}")
    print(f"  Unchanged (idx same)       : {unchanged_count}")
    print(f"  ⚠️  REMOVED during cleaning : {removed_count}")

    if removed_count > 0:
        print(f"\n  ⚠️  CRITICAL: {removed_count} ground-truth-cited chunks no longer")
        print(f"     exist — they were removed by boilerplate cleaning. These")
        print(f"     questions now have NO valid answer in the corpus at all.")
        print(f"     Affected questions:")
        for entry in remap_log:
            if entry["status"].startswith("REMOVED"):
                print(f"       {entry['q_id']}: was idx={entry['old_idx']} "
                      f"({entry.get('document_id')}, chunk_uid={entry.get('chunk_uid')})")
        print(f"\n     These need MANUAL re-verification — find a new relevant")
        print(f"     chunk for each in the cleaned corpus before trusting any")
        print(f"     retrieval test on these questions.")

    print(f"\n✅ Remapped ground truth saved: {OUTPUT_GT_PATH}")
    print(f"✅ Full audit log saved       : {REPORT_PATH}")
    print(f"\n⚠️  Use {OUTPUT_GT_PATH} (not the original file) for all further")
    print(f"   retrieval testing against the cleaned corpus.")


if __name__ == "__main__":
    main()
