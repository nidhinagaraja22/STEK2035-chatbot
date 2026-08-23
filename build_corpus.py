# -*- coding: utf-8 -*-
"""
STEK 2035 -- Build Corpus 2 (cleaned + metadata + authority)
=============================================================
Reads Corpus 1 (vector_store/chunks.jsonl, left untouched), joins the document
registry metadata + authority taxonomy onto every chunk, removes near-duplicate
chunks (logging before/after counts), subsets the aligned e5-base embeddings so
Corpus 2 is immediately usable, and writes a changelog.

Reads:
    corpus/document_registry.csv
    vector_store/chunks.jsonl          (Corpus 1 - NOT modified)
    vector_store/embeddings.npy        (Corpus 1 - NOT modified)

Writes:
    corpus/corpus_v2/corpus_v2_chunks.jsonl
    corpus/corpus_v2/embeddings_v2_e5base.npy
    corpus/corpus_v2/meta_v2.json
    corpus/corpus_v2/changelog.md

Run from repo root:
    python build_corpus.py
"""
import csv
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np

REGISTRY = Path("corpus/document_registry.csv")
CHUNKS_IN = Path("vector_store/chunks.jsonl")
EMB_IN = Path("vector_store/embeddings.npy")
OUT_DIR = Path("corpus/corpus_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEAR_DUP_THRESHOLD = 0.97   # cosine >= this = near-duplicate (chunks are non-overlapping)

# Single source of truth: authority_label is derived from authority_level, so the
# number (for reranking) and the label (for prompts/citations) can never disagree.
AUTHORITY_LABELS = {
    1: "Official STEK strategy (approved policy)",
    2: "Official city report / administrative document",
    3: "Planning / background material",
    4: "Citizen participation summary",
    5: "Individual citizen opinion",
}


# --------------------------------------------------------------------------- #
# 1. Load the document registry
# --------------------------------------------------------------------------- #
def _norm_name(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())

rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8-sig")))
# the 12 core documents, matched by exact filename
doc_rows = [r for r in rows if r["filename"].strip()
            and not r["document_id"].startswith("web_pdf_")
            and r["document_id"] != "website_html"]
docs_by_filename = {_norm_name(r["filename"]): r for r in doc_rows}
# explicit website-PDF rows, matched by a distinctive filename substring in the URL
website_pdf_rows = [r for r in rows if r["document_id"].startswith("web_pdf_")]
website_html_rule = next((r for r in rows if r["document_id"] == "website_html"), None)
print(f"Registry: {len(docs_by_filename)} documents + {len(website_pdf_rows)} website PDFs + "
      f"html rule={'yes' if website_html_rule else 'no'}")


def registry_for(chunk):
    """Return (registry_row, source_url) for a chunk, or (None, None) if unmatched."""
    origin, src = chunk.get("origin", ""), chunk.get("source", "")
    if src == "document":
        row = docs_by_filename.get(_norm_name(origin))
        return (row, row["source_url"] if row else None)
    if src == "website_pdf":
        for r in website_pdf_rows:
            if r["filename"] and r["filename"] in origin:
                return (r, origin)          # matched a specific scraped PDF
        return (website_html_rule, origin)  # fallback: unregistered web PDF
    # website_html
    if website_html_rule is not None:
        return (website_html_rule, origin)  # keep the real URL as source_url
    return (None, None)


# --------------------------------------------------------------------------- #
# 2. Load Corpus 1 + join metadata / authority onto every chunk
# --------------------------------------------------------------------------- #
chunks = [json.loads(l) for l in CHUNKS_IN.read_text(encoding="utf-8").splitlines()]
embeddings = np.load(EMB_IN)
assert len(chunks) == embeddings.shape[0], "chunks and embeddings misaligned"
n_before = len(chunks)

enriched = []
unmatched = []
for c in chunks:
    row, source_url = registry_for(c)
    if row is None:
        unmatched.append(c.get("origin", ""))
        row = {"document_id": "unknown", "document_title": "", "publication_date": "",
               "doc_type": "unknown", "authority_level": "", "is_participation": "FALSE"}
        source_url = c.get("origin", "")
    is_citizen = row.get("is_participation", "FALSE").strip().upper() == "TRUE"
    lvl = int(row["authority_level"]) if str(row["authority_level"]).strip().isdigit() else None
    label = AUTHORITY_LABELS.get(lvl, "Unknown")
    pub = str(row["publication_date"]).strip()
    # ready-to-inject source tag for the LLM prompt / citations (derived, never hand-entered)
    citation_prefix = f"{label} — {row['document_title']}" + (f" ({pub})" if pub else "")
    enriched.append({
        **c,                                          # keep original fields
        "document_id": row["document_id"],
        "document_title": row["document_title"],
        "source_url": source_url or "",
        "publication_date": row["publication_date"],
        "doc_type": row["doc_type"],
        "authority_level": lvl,
        "authority_label": label,
        "citation_prefix": citation_prefix,
        "is_citizen_opinion": is_citizen,
        "topic": c.get("lda_topic"),                  # carry existing LDA topic
        "section": c.get("chunk_id"),                 # v1 chunks have no page info
    })

if unmatched:
    uniq = sorted(set(unmatched))
    print(f"WARNING: {len(unmatched)} chunks from {len(uniq)} unmatched origins -> tagged 'unknown':")
    for o in uniq[:10]:
        print("   ", o[:80])


# --------------------------------------------------------------------------- #
# 3. Deduplicate near-identical chunks (greedy, by cosine on the embeddings)
# --------------------------------------------------------------------------- #
# embeddings are L2-normalized -> dot product = cosine
sim = embeddings @ embeddings.T
kept_idx, removed = [], []
for i in range(n_before):
    if kept_idx:
        sims = sim[i, kept_idx]
        j = int(np.argmax(sims))
        if sims[j] >= NEAR_DUP_THRESHOLD:
            removed.append((i, kept_idx[j], float(sims[j])))
            continue
    kept_idx.append(i)

corpus2 = [enriched[i] for i in kept_idx]
emb2 = embeddings[kept_idx]
n_after = len(corpus2)


# --------------------------------------------------------------------------- #
# 4. Write Corpus 2 + aligned embeddings + meta + changelog
# --------------------------------------------------------------------------- #
with (OUT_DIR / "corpus_v2_chunks.jsonl").open("w", encoding="utf-8") as f:
    for r in corpus2:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
np.save(OUT_DIR / "embeddings_v2_e5base.npy", emb2)
(OUT_DIR / "meta_v2.json").write_text(json.dumps({
    "corpus_version": 2,
    "built_from": "vector_store/chunks.jsonl (Corpus 1)",
    "model": "intfloat/multilingual-e5-base",          # key name rag_server.py expects
    "embedding_model": "intfloat/multilingual-e5-base",
    "dim": int(emb2.shape[1]),
    "chunks_before": n_before,
    "chunks_after": n_after,
    "near_duplicates_removed": len(removed),
    "near_dup_threshold": NEAR_DUP_THRESHOLD,
}, indent=2), encoding="utf-8")

# authority + citizen-opinion distribution for the changelog
from collections import Counter
auth = Counter(r["authority_level"] for r in corpus2)
cit = sum(1 for r in corpus2 if r["is_citizen_opinion"])
changelog = [
    "# Corpus 2 - changelog",
    f"Built: {date.today().isoformat()}",
    "",
    "## Provenance",
    "- Corpus 1 (original): vector_store/chunks.jsonl - LEFT UNTOUCHED",
    "- Corpus 2 adds: registry metadata, authority taxonomy, citizen-opinion tags; removes near-duplicate chunks.",
    "",
    "## Deduplication",
    f"- Chunks before: {n_before}",
    f"- Near-duplicate chunks removed (cosine >= {NEAR_DUP_THRESHOLD}): {len(removed)}",
    f"- Chunks after:  {n_after}",
    "",
    "## Metadata added to every chunk",
    "- document_id, document_title, source_url, publication_date, doc_type,",
    "  authority_level, is_citizen_opinion, topic, section",
    "",
    "## Authority-level distribution (chunks)",
]
for lvl in sorted(k for k in auth if k is not None):
    changelog.append(f"- Level {lvl}: {auth[lvl]} chunks")
if None in auth:
    changelog.append(f"- Unassigned: {auth[None]} chunks")
changelog.append(f"- Citizen-opinion chunks (is_citizen_opinion=TRUE): {cit}")
(OUT_DIR / "changelog.md").write_text("\n".join(changelog), encoding="utf-8")

print(f"\nCorpus 1 chunks: {n_before}")
print(f"Near-duplicates removed: {len(removed)}")
print(f"Corpus 2 chunks: {n_after}")
print("Authority distribution:", dict(sorted((k, v) for k, v in auth.items() if k is not None)))
print(f"Citizen-opinion chunks: {cit}")
print(f"\nWrote -> {OUT_DIR}/")
