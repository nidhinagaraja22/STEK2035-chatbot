# -*- coding: utf-8 -*-
"""
STEK 2035 -- Priority 8: Abstention accuracy evaluation
=======================================================
Tests whether the RAG system correctly REFUSES to answer questions the corpus
cannot support (saying the canonical refusal) instead of hallucinating.

Two question sets are scored together so the metric is honest:
  * UNANSWERABLE (eval/unanswerable_questions.json)  -> system SHOULD abstain
  * ANSWERABLE control (sampled from eval/gold_master.json) -> system SHOULD answer
A model that refuses everything scores perfectly on the first set alone; the
control set exposes that as over-refusal.

Confusion matrix (positive class = "abstain"):
  TP = unanswerable & abstained     (correct refusal)
  FN = unanswerable & answered      (HALLUCINATION -- the failure we care about)
  FP = answerable   & abstained     (OVER-REFUSAL -- refused a good question)
  TN = answerable   & answered      (correct answer)

Metrics:
  abstention_accuracy = TP / (TP+FN)          # recall on the unanswerable set (PRIMARY)
  over_refusal_rate   = FP / (FP+TN)          # on the answerable control set (lower=better)
  abstention_precision= TP / (TP+FP)
  abstention_f1       = 2PR/(P+R)
  overall_accuracy    = (TP+TN) / (TP+FN+FP+TN)

Where do the model responses come from?
  --live            : POST every question to a running rag_server (SERVER_URL/chat)
  --responses FILE  : read a JSON map {question_id: reply_text} you collected by hand
  (neither)         : write a blank responses template you can fill in, then re-run
                      with --responses.

Run examples:
    python eval/abstention_eval.py --live
    python eval/abstention_eval.py --responses eval/abstention_responses.json
    python eval/abstention_eval.py            # just emit the template
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parents[1]
UNANS = BASE / "eval/unanswerable_questions.json"
GOLD = BASE / "eval/gold_master.json"          # source of answerable controls
TEMPLATE = BASE / "eval/abstention_responses.json"
OUT_SUMMARY = BASE / "eval/abstention_results.csv"
OUT_DETAILS = BASE / "eval/abstention_details.csv"
SERVER_URL = "http://localhost:8000"
N_CONTROLS = 10                                # answerable questions to test for over-refusal
random.seed(42)

# --------------------------------------------------------------------------- #
# Abstention detector
# --------------------------------------------------------------------------- #
# The canonical refusal the prompt asks for, plus robust variants so we still
# recognise an abstention if the model paraphrases it slightly.
ABSTENTION_MARKERS = [
    r"nicht gen[üu]gend informationen",
    r"keine ausreichenden informationen",
    r"nicht ausreichend(e)? informationen",
    r"enthalten keine( ausreichenden)? informationen",
    r"liegen keine (ausreichenden )?informationen",
    r"lassen sich .{0,30}nicht beantworten",
    r"kann(?: ich)? .{0,30}nicht beantwortet? werden",
    r"keine (belastbaren )?(angaben|aussagen|informationen)",
    r"im (bereitgestellten )?kontext nicht enthalten",
    r"nicht (im|in den) (vorliegenden )?(stek-?)?dokument",
    r"do not contain sufficient information",
    r"not (enough|sufficient) information",
]
ABSTENTION_RE = re.compile("|".join(ABSTENTION_MARKERS), re.IGNORECASE)


def is_abstention(reply: str) -> bool:
    if not reply or not reply.strip():
        return False
    return bool(ABSTENTION_RE.search(reply))


# --------------------------------------------------------------------------- #
# Build the combined test set (unanswerable + answerable controls)
# --------------------------------------------------------------------------- #
def load_testset():
    unans = json.loads(UNANS.read_text(encoding="utf-8"))["questions"]
    items = [{"id": q["id"], "question": q["question"], "should_abstain": True,
              "type": q["unanswerable_type"]} for q in unans]

    controls = []
    if GOLD.exists():
        gold = json.loads(GOLD.read_text(encoding="utf-8"))
        answerable = [g for g in gold if g.get("relevant_chunk_ids") and g.get("answerable", True) not in ("no", False)]
        random.shuffle(answerable)
        for g in answerable[:N_CONTROLS]:
            controls.append({"id": g["id"], "question": g["question"],
                             "should_abstain": False, "type": "answerable_control"})
    return items + controls


# --------------------------------------------------------------------------- #
# Response sources
# --------------------------------------------------------------------------- #
def get_responses_live(testset):
    import requests
    replies = {}
    for it in testset:
        r = requests.post(f"{SERVER_URL}/chat", json={"message": it["question"]}, timeout=180)
        r.raise_for_status()
        replies[it["id"]] = r.json().get("reply", "")
        tag = "abstain?" if it["should_abstain"] else "answer? "
        print(f"  [{it['id']:>18}] {tag} {'ABSTAINED' if is_abstention(replies[it['id']]) else 'answered'}")
    return replies


def write_template(testset):
    tmpl = {it["id"]: "" for it in testset}
    TEMPLATE.write_text(json.dumps(tmpl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote blank responses template -> {TEMPLATE}")
    print("Fill in each model reply, then re-run with:")
    print(f"    python eval/abstention_eval.py --responses {TEMPLATE.relative_to(BASE)}")


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #
def score(testset, replies):
    TP = FN = FP = TN = 0
    rows = []
    for it in testset:
        reply = replies.get(it["id"], "")
        abst = is_abstention(reply)
        if it["should_abstain"]:
            outcome = "TP (correct refusal)" if abst else "FN (HALLUCINATION)"
            if abst: TP += 1
            else: FN += 1
        else:
            outcome = "FP (OVER-REFUSAL)" if abst else "TN (correct answer)"
            if abst: FP += 1
            else: TN += 1
        rows.append({"id": it["id"], "type": it["type"],
                     "should_abstain": it["should_abstain"],
                     "model_abstained": abst, "outcome": outcome,
                     "question": it["question"],
                     "reply_snippet": re.sub(r"\s+", " ", reply)[:160]})

    def safe(n, d): return n / d if d else float("nan")
    metrics = {
        "abstention_accuracy": safe(TP, TP + FN),   # PRIMARY (recall on unanswerable)
        "over_refusal_rate":   safe(FP, FP + TN),   # on answerable controls (lower better)
        "abstention_precision": safe(TP, TP + FP),
        "abstention_recall":   safe(TP, TP + FN),
        "overall_accuracy":    safe(TP + TN, TP + FN + FP + TN),
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "n_unanswerable": TP + FN, "n_answerable_control": FP + TN,
    }
    f1 = safe(2 * metrics["abstention_precision"] * metrics["abstention_recall"],
              metrics["abstention_precision"] + metrics["abstention_recall"])
    metrics["abstention_f1"] = f1
    return metrics, rows


def report(metrics, rows):
    print("\n" + "=" * 70)
    print("ABSTENTION EVALUATION  (positive class = abstain)")
    print("=" * 70)
    print(f"  Unanswerable questions : {metrics['n_unanswerable']}")
    print(f"  Answerable controls    : {metrics['n_answerable_control']}")
    print(f"\n  Confusion matrix:")
    print(f"    TP correct refusal   : {metrics['TP']}")
    print(f"    FN HALLUCINATION     : {metrics['FN']}   <- unsupported answers (bad)")
    print(f"    FP over-refusal      : {metrics['FP']}   <- refused good questions (bad)")
    print(f"    TN correct answer    : {metrics['TN']}")
    print(f"\n  >> ABSTENTION ACCURACY : {metrics['abstention_accuracy']:.3f}  (primary metric)")
    print(f"     over-refusal rate   : {metrics['over_refusal_rate']:.3f}")
    print(f"     precision / recall  : {metrics['abstention_precision']:.3f} / {metrics['abstention_recall']:.3f}")
    print(f"     F1                  : {metrics['abstention_f1']:.3f}")
    print(f"     overall accuracy    : {metrics['overall_accuracy']:.3f}")

    hall = [r for r in rows if r["outcome"].startswith("FN")]
    over = [r for r in rows if r["outcome"].startswith("FP")]
    if hall:
        print("\n  Hallucinations (should have abstained but answered):")
        for r in hall: print(f"    - {r['id']} [{r['type']}] {r['question']}")
    if over:
        print("\n  Over-refusals (should have answered but abstained):")
        for r in over: print(f"    - {r['id']} {r['question']}")

    with OUT_SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, round(v, 4) if isinstance(v, float) else v])
    with OUT_DETAILS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_SUMMARY.name} + {OUT_DETAILS.name}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="query a running rag_server")
    ap.add_argument("--responses", type=str, help="JSON map {question_id: reply}")
    args = ap.parse_args()

    testset = load_testset()
    print(f"Test set: {sum(t['should_abstain'] for t in testset)} unanswerable "
          f"+ {sum(not t['should_abstain'] for t in testset)} answerable controls")

    if args.responses:
        replies = json.loads(Path(args.responses).read_text(encoding="utf-8"))
    elif args.live:
        print(f"Querying {SERVER_URL}/chat ...")
        replies = get_responses_live(testset)
    else:
        write_template(testset)
        return

    metrics, rows = score(testset, replies)
    report(metrics, rows)


if __name__ == "__main__":
    main()
