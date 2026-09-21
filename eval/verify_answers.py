# -*- coding: utf-8 -*-
"""
Answer verification helper — grounding pre-check + evidence pool.

For each draft answer it (1) retrieves the corpus chunks most likely to support
it, (2) measures how much of the answer's content actually appears in those
chunks (a groundedness score), and (3) flags weakly-grounded answers for human
review. It produces a verification worklist a human uses to confirm/correct the
answer and mark the supporting chunk ids.

Lexical only (no model needed). Reads Corpus 2 + the 105 Q&A files.
Writes: eval/answer_verification.csv
"""
import csv
import json
import re
from pathlib import Path

BASE = Path.cwd()
for cand in [BASE, *BASE.parents]:
    if (cand / "corpus" / "corpus_v2" / "corpus_v2_chunks.jsonl").exists():
        BASE = cand
        break

chunks = [json.loads(l) for l in (BASE / "corpus/corpus_v2/corpus_v2_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
questions = {q["id"]: q for q in json.loads((BASE / "eval/questions_100.json").read_text(encoding="utf-8"))}
answers = json.loads((BASE / "eval/answers_100.json").read_text(encoding="utf-8"))

STOP = set("der die das und oder für mit von den dem des ein eine einer eines einem einen "
           "auf aus bei nach über unter durch zur zum als wie werden wird soll sollen kann "
           "sowie durch mehr sehr auch nicht sich ist sind laut bzw etc".split())
TOP_K = 8

def toks(s):
    return [w for w in re.findall(r"[a-zäöüß]+", s.lower()) if len(w) >= 4 and w not in STOP]

chunk_tok_sets = [set(toks(c["text"])) for c in chunks]

rows = []
flagged = 0
for a in answers:
    q = questions[a["id"]]
    if not q["answerable"]:                       # unanswerable / vague: no grounding expected
        rows.append({"id": a["id"], "category": q["category"], "q_type": q["q_type"],
                     "question": q["question"], "draft_answer": a["reference_answer"],
                     "grounding": "N/A", "flag": "abstention/clarify",
                     "top_chunk_ids": "", "source_docs": "", "authority_levels": "", "evidence_snippets": ""})
        continue

    qa_terms = set(toks(q["question"]) + toks(a["reference_answer"]))
    scored = sorted(range(len(chunks)), key=lambda i: len(qa_terms & chunk_tok_sets[i]), reverse=True)[:TOP_K]
    evidence = " ".join(chunks[i]["text"].lower() for i in scored)

    ans_terms = set(toks(a["reference_answer"]))
    grounding = round(sum(1 for w in ans_terms if w in evidence) / max(len(ans_terms), 1), 2)
    flag = "REVIEW (low grounding)" if grounding < 0.6 else "ok"
    if flag.startswith("REVIEW"):
        flagged += 1

    rows.append({
        "id": a["id"], "category": q["category"], "q_type": q["q_type"],
        "question": q["question"], "draft_answer": a["reference_answer"],
        "grounding": grounding, "flag": flag,
        "top_chunk_ids": " | ".join(chunks[i]["chunk_uid"] for i in scored),
        "source_docs": " | ".join(sorted(set(chunks[i]["document_id"] for i in scored))),
        "authority_levels": " | ".join(sorted(set(str(chunks[i]["authority_level"]) for i in scored))),
        "evidence_snippets": "  ///  ".join(chunks[i]["text"][:200].replace("\n", " ") for i in scored[:3]),
    })

# verifier columns left empty to fill in
verifier_cols = ["answer_correct(y/n)", "corrected_answer", "relevant_chunk_ids", "answerable_ok(y/n)", "authority_ok(y/n)"]
cols = ["id", "category", "q_type", "question", "draft_answer", "grounding", "flag",
        "top_chunk_ids", "source_docs", "authority_levels", "evidence_snippets"] + verifier_cols

out = BASE / "eval/answer_verification.csv"
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({**{c: "" for c in verifier_cols}, **r})

answerable = [r for r in rows if r["grounding"] != "N/A"]
print(f"Total Q&A: {len(rows)} | answerable checked: {len(answerable)}")
print(f"Flagged for review (grounding < 0.6): {flagged}")
print(f"Well-grounded (>= 0.6): {len(answerable) - flagged}")
print(f"\nWrote verification worklist -> {out}")
print("Verifiers: read draft_answer against evidence_snippets/top_chunk_ids, then fill the last 5 columns.")
