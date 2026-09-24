# -*- coding: utf-8 -*-
"""
Add registry metadata + authority taxonomy to the sentence-aware chunks.
========================================================================
Joins corpus/document_registry.csv onto every chunk of a chunks file (matched
by `origin`/`source`) and writes each chunk back out with these fields added:

    document_id, document_title, source_url, publication_date, doc_type,
    authority_level (1-5), authority_label, citation_prefix,
    is_citizen_opinion, topic, section

This is the metadata step ONLY — it does not deduplicate or embed (those are
separate steps). The authority/citation conventions are identical to
build_corpus.py so the output is schema-compatible with corpus_v2.

Reads:
    corpus/document_registry.csv
    vector_store/chunks_v2.jsonl        (sentence-aware chunks; not modified)
Writes:
    vector_store/chunks_v2_meta.jsonl
"""
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# --- repo-root autodetection ------------------------------------------------ #
BASE = Path(__file__).resolve().parent
for cand in [BASE, *BASE.parents]:
    if (cand / "corpus" / "document_registry.csv").exists():
        BASE = cand
        break

REGISTRY = BASE / "corpus/document_registry.csv"
CHUNKS_IN = BASE / "vector_store/chunks_v2.jsonl"
OUT_PATH = BASE / "vector_store/chunks_v2_meta.jsonl"

# authority_label is derived from authority_level, so number (for reranking) and
# label (for prompts/citations) can never disagree.
AUTHORITY_LABELS = {
    1: "Official STEK strategy (approved policy)",
    2: "Official city report / administrative document",
    3: "Planning / background material",
    4: "Citizen participation summary",
    5: "Individual citizen opinion",
}

# Within participation docs (base L4), a chunk carrying first-person citizen
# voice is an individual contribution -> downgraded to L5. The rest stay L4
# (the city's editorial summary). Heuristic, documented for auditing.
CITIZEN_VOICE_RE = re.compile("|".join([
    r"\bich\b", r"\bwir brauchen\b", r"\bmeiner meinung\b", r"\bfinde ich\b",
    r"\bich (finde|wünsche|möchte|will|fände|hätte|bin)\b",
    r"\bwünsche mir\b", r"\bes wäre (schön|gut|toll|wichtig)\b",
]), re.IGNORECASE)


def _norm_name(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


# --- 1. load registry ------------------------------------------------------- #
rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8-sig")))
doc_rows = [r for r in rows if r["filename"].strip()
            and not r["document_id"].startswith("web_pdf_")
            and r["document_id"] != "website_html"]
docs_by_filename = {_norm_name(r["filename"]): r for r in doc_rows}
website_pdf_rows = [r for r in rows if r["document_id"].startswith("web_pdf_")]
website_html_rule = next((r for r in rows if r["document_id"] == "website_html"), None)


def registry_for(chunk):
    """Return (registry_row, source_url) for a chunk, or (None, None)."""
    origin, src = chunk.get("origin", ""), chunk.get("source", "")
    if src == "document":
        row = docs_by_filename.get(_norm_name(origin))
        return (row, row["source_url"] if row else None)
    if src == "website_pdf":
        for r in website_pdf_rows:
            if r["filename"] and r["filename"] in origin:
                return (r, origin)              # matched a specific scraped PDF
        return (website_html_rule, origin)      # fallback: unregistered web PDF
    if website_html_rule is not None:           # website_html
        return (website_html_rule, origin)      # keep the real URL as source_url
    return (None, None)


# --- 2. enrich every chunk -------------------------------------------------- #
def main():
    chunks = [json.loads(l) for l in CHUNKS_IN.read_text(encoding="utf-8").splitlines()]
    enriched, unmatched = [], []
    for c in chunks:
        row, source_url = registry_for(c)
        if row is None:
            unmatched.append(c.get("origin", ""))
            row = {"document_id": "unknown", "document_title": "", "publication_date": "",
                   "doc_type": "unknown", "authority_level": "", "is_participation": "FALSE"}
            source_url = c.get("origin", "")
        is_citizen = row.get("is_participation", "FALSE").strip().upper() == "TRUE"
        lvl = int(row["authority_level"]) if str(row["authority_level"]).strip().isdigit() else None
        if is_citizen and lvl == 4 and CITIZEN_VOICE_RE.search(c.get("text", "")):
            lvl = 5                              # verbatim citizen voice -> L5
        label = AUTHORITY_LABELS.get(lvl, "Unknown")
        pub = str(row["publication_date"]).strip()
        citation_prefix = f"{label} — {row['document_title']}" + (f" ({pub})" if pub else "")
        enriched.append({
            **c,                                 # keep original chunk fields
            "document_id": row["document_id"],
            "document_title": row["document_title"],
            "source_url": source_url or "",
            "publication_date": row["publication_date"],
            "doc_type": row["doc_type"],
            "authority_level": lvl,
            "authority_label": label,
            "citation_prefix": citation_prefix,
            "is_citizen_opinion": is_citizen,
            "topic": c.get("lda_topic"),
            "section": c.get("chunk_id"),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- report ---
    auth = Counter(r["authority_level"] for r in enriched)
    cit = sum(1 for r in enriched if r["is_citizen_opinion"])
    print(f"Registry: {len(docs_by_filename)} documents + {len(website_pdf_rows)} website PDFs "
          f"+ html rule={'yes' if website_html_rule else 'no'}")
    print(f"Chunks enriched: {len(enriched)}")
    print("Authority-level distribution:")
    for lvl in sorted(k for k in auth if k is not None):
        print(f"  L{lvl}: {auth[lvl]}")
    if None in auth:
        print(f"  unassigned (None): {auth[None]}")
    print(f"Citizen-opinion chunks: {cit}")
    if unmatched:
        uniq = sorted(set(unmatched))
        print(f"\nWARNING: {len(unmatched)} chunks from {len(uniq)} unmatched origins -> 'unknown':")
        for o in uniq[:10]:
            print("   ", o[:80])
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
