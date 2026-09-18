"""
STEK 2035 — Testing Candidate-Pool Statistics for Vague/Unanswerable Detection
===================================================================================
Tests whether two PRE-SELECTION candidate-pool statistics can
distinguish vague questions from unanswerable questions from normal
answerable questions — using the same real ground-truth categories
(6 vague, 6 unanswerable, 48 answerable) already used to calibrate
Task 10's word-count threshold.

Two DIFFERENT hypothesized signals, since vague and unanswerable are
different failure modes:

  top_score    = highest similarity across ALL corpus chunks
                 Hypothesis: UNANSWERABLE questions show a distinctly
                 LOW top_score — nothing in the corpus is genuinely
                 relevant, so even the best match is weak.

  score_spread = standard deviation of similarity across the top-20
                 candidates
                 Hypothesis: VAGUE questions show a distinctly LOW
                 spread — nothing discriminates itself as clearly
                 more relevant than anything else. This matches a
                 real, already-observed finding: Q046 ("Ich möchte
                 etwas über Heidelberg erfahren") showed retrieval
                 scores clustered in a 0.863-0.868 range, spread of
                 only 0.005.

Usage:
    python stek_test_vague_unanswerable_signals.py

Requires:
    corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    stek_task4_5_master_ground_truth.json
"""

import json
import statistics
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("corpus/corpus_v2/corpus_v2_chunks_l4l5split.jsonl")
EMB_PATH = Path("corpus/corpus_v2/embeddings_v2_e5base.npy")
EMBED_MODEL = "intfloat/multilingual-e5-base"
GROUND_TRUTH_PATH = Path("stek_task4_5_master_ground_truth.json")

TOP_N_FOR_SPREAD = 20

# Generic, ubiquitous terms that appear in nearly every corpus chunk
# (confirmed throughout this project: boilerplate/overview chunks
# repeat these heavily — "STADTENTWICKLUNGSKONZEPT 2035", "Liebe
# Bürgerinnen und Bürger", etc.). Hypothesis: a vague question like
# "Ich möchte etwas über Heidelberg erfahren" may score artificially
# high not because it's semantically well-matched, but because it
# shares these ubiquitous terms with generic overview/boilerplate
# content that ALSO scores high for many unrelated questions — the
# same Cause 1 contamination already documented elsewhere in this
# project, potentially masking the true vague/unanswerable signal.
# Standard German function words/stopwords — carry grammatical
# structure but zero topical content on their own. Combined with
# GENERIC_TERMS below for a genuinely complete strip, since the
# first version only removed domain-specific ubiquitous terms and
# left ordinary function words (die, für, bis, etc.) untouched.
GERMAN_STOPWORDS = [
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einen",
    "und", "oder", "aber", "doch", "wie", "was", "wo", "wann", "wer", "wen", "wem",
    "welche", "welcher", "welches", "welchen", "welchem",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten", "sein", "seine", "seiner", "seinen",
    "soll", "sollen", "sollte", "will", "wollen", "kann", "können", "könnte",
    "muss", "müssen", "darf", "dürfen", "mag", "möchte",
    "für", "mit", "bei", "bis", "durch", "gegen", "ohne", "um", "auf", "aus",
    "in", "im", "an", "am", "zu", "zum", "zur", "von", "vom", "nach", "über",
    "unter", "vor", "hinter", "neben", "zwischen",
    "sich", "es", "man", "ich", "du", "er", "sie", "wir", "ihr", "mir", "mich",
    "dich", "ihm", "ihn", "uns", "euch", "ihnen",
    "nicht", "kein", "keine", "auch", "noch", "schon", "nur", "sehr", "so",
    "als", "wenn", "dann", "denn", "weil", "dass", "ob",
]

GENERIC_TERMS = GERMAN_STOPWORDS + [
    "heidelberg", "stadt", "stek", "stek 2035", "stadtentwicklungskonzept",
    "2035", "konzept", "erfahren", "möchte", "etwas", "gibts", "erzähl",
    "neues", "beim", "sagt",  # "sagt" already recognized as generic in
                                # Task 10's FRAME_WORDS — was missing here,
                                # causing Q048 ("Was sagt STEK 2035?") to
                                # slip through this signal specifically
    # "tourist"/"touristen" added based on DIRECT evidence (Q049 manual
    # check): every corpus match for this term is citizen commentary
    # about MANAGING tourism's impact on the city ("nicht nur für
    # Tourist:innen", fear of Heidelberg becoming a "Touristendisneyland"),
    # not actual tourist recommendations — a confirmed false-positive
    # contamination source, same class of problem as Heidelberg/STEK's
    # ubiquity, just topic-specific rather than corpus-wide. Unlike the
    # universal terms above, this is a narrower, single-case-motivated
    # addition — worth revisiting if it turns out to suppress genuinely
    # relevant tourism-policy questions in the future.
    "tourist", "touristen", "touristisch", "touristische",
]


def strip_generic_terms(text: str) -> str:
    words = text.split()
    filtered = [w for w in words if w.lower().strip("?.,!") not in GENERIC_TERMS]
    return " ".join(filtered).strip()  # may legitimately be empty — handled by caller,
                                          # NOT papered over with a placeholder token


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


def compute_pool_statistics(question, model, chunk_embeddings):
    q_vec = model.encode(f"query: {question}")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-8)
    sims = chunk_embeddings @ q_vec
    top_sorted = np.sort(sims)[::-1]

    top_score = float(top_sorted[0])
    top_n = top_sorted[:TOP_N_FOR_SPREAD]
    score_spread = float(np.std(top_n))

    return top_score, score_spread


def categorize_question(q):
    if q["category"] in ("vague",):
        return "vague"
    if q["category"] in ("unanswerable",) or not q.get("answerable", True):
        return "unanswerable"
    if q.get("relevant_chunks"):
        return "answerable"
    return None


def main():
    print("=" * 70)
    print("Testing Candidate-Pool Statistics for Vague/Unanswerable Detection")
    print("=" * 70)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from: {CHUNKS_PATH}")
    chunks = load_chunks()
    chunk_embeddings = load_normalised(EMB_PATH)
    if chunk_embeddings.shape[0] != len(chunks):
        print(f"❌ Mismatch: {chunk_embeddings.shape[0]} embeddings vs {len(chunks)} chunks — aborting.")
        return

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    groups = {"vague": [], "unanswerable": [], "answerable": []}
    groups_stripped = {"vague": [], "unanswerable": [], "answerable": []}

    for q in gt_data["questions"]:
        category = categorize_question(q)
        if category is None:
            continue

        top_score, score_spread = compute_pool_statistics(q["question"], model, chunk_embeddings)
        groups[category].append({"q_id": q["q_id"], "question": q["question"],
                                    "top_score": top_score, "score_spread": score_spread})

        stripped_question = strip_generic_terms(q["question"])
        if not stripped_question:
            # question is composed ENTIRELY of generic/stop words — this
            # emptiness IS the signal (maximal genericness), not something
            # to embed via an arbitrary placeholder token (which was
            # confirmed to corrupt the spread statistic in an earlier
            # version of this script — 3 different questions all mapped
            # to the identical placeholder "[leer]", inflating spread
            # with a repeated arbitrary value rather than 3 real signals)
            groups_stripped[category].append({"q_id": q["q_id"], "question": "(fully generic — no content survived stripping)",
                                                 "top_score": None, "score_spread": None, "fully_generic": True})
            print(f"  [{category:<12}] {q['q_id']}: "
                  f"unstripped(top={top_score:.4f}, spread={score_spread:.4f})  "
                  f"stripped: FULLY GENERIC (no substantive content survived)")
            continue

        top_score_s, score_spread_s = compute_pool_statistics(stripped_question, model, chunk_embeddings)
        groups_stripped[category].append({"q_id": q["q_id"], "question": stripped_question,
                                             "top_score": top_score_s, "score_spread": score_spread_s,
                                             "fully_generic": False})

        print(f"  [{category:<12}] {q['q_id']}: "
              f"unstripped(top={top_score:.4f}, spread={score_spread:.4f})  "
              f"stripped(top={top_score_s:.4f}, spread={score_spread_s:.4f})  "
              f"'{stripped_question}'")

    def print_group_stats(label, data):
        print(f"\n{'='*70}")
        print(f"GROUP STATISTICS — {label}")
        print(f"{'='*70}")
        for cat in ["answerable", "vague", "unanswerable"]:
            rows = data[cat]
            if not rows:
                continue
            fully_generic_count = sum(1 for r in rows if r.get("fully_generic"))
            numeric_rows = [r for r in rows if not r.get("fully_generic")]
            print(f"\n{cat.upper()} (n={len(rows)}, "
                  f"{fully_generic_count} fully-generic/stripped-to-nothing):")
            if numeric_rows:
                top_scores = [r["top_score"] for r in numeric_rows]
                spreads = [r["score_spread"] for r in numeric_rows]
                print(f"  top_score:    mean={statistics.mean(top_scores):.4f}  "
                      f"min={min(top_scores):.4f}  max={max(top_scores):.4f}  (n={len(numeric_rows)})")
                print(f"  score_spread: mean={statistics.mean(spreads):.4f}  "
                      f"min={min(spreads):.4f}  max={max(spreads):.4f}  (n={len(numeric_rows)})")
            else:
                print(f"  (all questions in this group were fully generic — no numeric stats)")

    print_group_stats("UNSTRIPPED (original)", groups)
    print_group_stats("STRIPPED (generic terms removed)", groups_stripped)

    with open("vague_unanswerable_signal_test.json", "w", encoding="utf-8") as f:
        json.dump({"unstripped": groups, "stripped": groups_stripped}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: vague_unanswerable_signal_test.json")
    print(f"\n👉 Compare the STRIPPED group ranges — if stripping generic terms")
    print(f"   reveals separation that wasn't visible before, the original")
    print(f"   negative result was indeed caused by generic-term contamination.")
    print(f"   If ranges still overlap even after stripping, the hypothesis")
    print(f"   is not supported and this remains a genuine negative result.")


if __name__ == "__main__":
    main()