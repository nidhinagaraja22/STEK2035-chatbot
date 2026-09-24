# -*- coding: utf-8 -*-
"""
Sentence-aware chunking with token-based sizing and overlap
===========================================================
Splits the STEK input documents into retrieval chunks that (a) never cut a
sentence in half and (b) carry a fixed token overlap between neighbours, so an
answer spanning a boundary is not lost.

Why sentence-aware + overlap (vs fixed-size character windows):
  * fixed-size windows slice sentences and German compound words in half,
    producing chunks that embed poorly and read badly;
  * sizing in real *tokens* (the e5/bge-m3 tokenizer) makes the 256-token budget
    mean the same thing to us and to the embedding model;
  * ~40-token overlap keeps context continuous across chunk boundaries.

Pipeline per document:
  clean → sentence-split (spaCy DE) → measure tokens → greedily pack sentences
  up to TARGET_TOKENS → carry OVERLAP_TOKENS of trailing sentences into the next
  chunk → merge a too-small final chunk back.

Run (from the repo root, with the stek virtualenv):
    python merge_and_embedding/sentence_aware_chunker.py

Output:
    vector_store/chunks_sentence_aware.jsonl   (one JSON object per line)
"""
import json
import re
import unicodedata
from pathlib import Path

import spacy
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
TOKENIZER_NAME = "intfloat/multilingual-e5-base"   # XLM-R tokenizer, shared by e5 & bge-m3
TARGET_TOKENS = 256      # target chunk size (tokens)
OVERLAP_TOKENS = 40      # ~15% overlap, carried at sentence granularity
MIN_TOKENS = 50          # merge a trailing chunk smaller than this into the previous one

# Repo-root autodetection: walk up until we find the input folder.
BASE = Path(__file__).resolve().parent
for cand in [BASE, *BASE.parents]:
    if (cand / "Input text files").exists():
        BASE = cand
        break
INPUT_DIR = BASE / "Input text files"
OUT_DIR = BASE / "vector_store"
OUT_PATH = OUT_DIR / "chunks_sentence_aware.jsonl"

# Files to skip: `extracted_text.txt` is a combined re-extraction dump, not one
# of the 12 named STEK documents, so it duplicates content. Add names here to
# exclude them (case-insensitive).
SKIP_FILES = {"extracted_text.txt"}

# --------------------------------------------------------------------------- #
# Tokenizer + sentence splitter (loaded once)
# --------------------------------------------------------------------------- #
print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

nlp = spacy.blank("de")            # rule-based German sentence segmentation, no model needed
nlp.add_pipe("sentencizer")
nlp.max_length = 3_000_000


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text: str) -> list[str]:
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    # rejoin German words hyphenated across a line break: "Wohn-\nraum" -> "Wohnraum"
    text = re.sub(r"([a-zäöüß])-\s*\n\s*([a-zäöüß])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Sentence-aware chunking with overlap
# --------------------------------------------------------------------------- #
def _split_oversized(sentence: str, max_tokens: int) -> list[str]:
    """Hard-split a single over-long 'sentence' into <= max_tokens word-pieces
    (safeguard for run-ons / tables that survive sentence splitting)."""
    words, pieces, buf = sentence.split(), [], []
    for w in words:
        buf.append(w)
        if count_tokens(" ".join(buf)) >= max_tokens:
            pieces.append(" ".join(buf)); buf = []
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def sentence_aware_chunks(text: str, target=TARGET_TOKENS, overlap=OVERLAP_TOKENS,
                          min_tokens=MIN_TOKENS) -> list[dict]:
    # 1. sentences + token counts (splitting any oversized sentence)
    sents, counts = [], []
    for s in split_sentences(text):
        c = count_tokens(s)
        if c > target:
            for piece in _split_oversized(s, target):
                sents.append(piece); counts.append(count_tokens(piece))
        else:
            sents.append(s); counts.append(c)
    if not sents:
        return []

    # 2. greedy packing with sentence-level overlap
    chunks, i, N = [], 0, len(sents)
    while i < N:
        cur, j = 0, i
        while j < N and (cur + counts[j] <= target or j == i):
            cur += counts[j]; j += 1
        chunks.append({
            "text": " ".join(sents[i:j]),
            "sent_start": i, "sent_end": j - 1, "token_count": cur,
        })
        if j >= N:
            break
        # walk back to include trailing sentences totalling >= overlap tokens
        ov, k = 0, j
        while k > i and ov < overlap:
            k -= 1; ov += counts[k]
        i = k if k > i else j          # guarantee forward progress

    # 3. merge a too-small trailing chunk into the previous one
    if len(chunks) >= 2 and chunks[-1]["token_count"] < min_tokens:
        prev, last = chunks[-2], chunks.pop()
        prev["text"] += " " + last["text"]
        prev["sent_end"] = last["sent_end"]
        prev["token_count"] += last["token_count"]
    return chunks


# --------------------------------------------------------------------------- #
# Run over the corpus
# --------------------------------------------------------------------------- #
def doc_id_from_name(name: str) -> str:
    stem = re.sub(r"(?i)^\d+_pdf_", "", Path(name).stem)
    return re.sub(r"\s+", "_", stem.strip())


def main():
    files = sorted(p for p in INPUT_DIR.glob("*.txt")
                   if p.is_file() and p.name.lower() not in SKIP_FILES)
    if not files:
        raise SystemExit(f"No .txt files in {INPUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks, per_doc, tok_counts = [], {}, []
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw)
        chunks = sentence_aware_chunks(text)
        did = doc_id_from_name(path.name)
        for idx, ch in enumerate(chunks):
            all_chunks.append({
                "chunk_uid": f"{did}::{idx}",
                "document_id": did,
                "source_file": path.name,
                "chunk_index": idx,
                "text": ch["text"],
                "token_count": ch["token_count"],
                "sent_start": ch["sent_start"],
                "sent_end": ch["sent_end"],
            })
            tok_counts.append(ch["token_count"])
        per_doc[path.name] = len(chunks)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # --- stats ---
    n = len(all_chunks)
    avg = sum(tok_counts) / n if n else 0
    over = sum(1 for t in tok_counts if t > TARGET_TOKENS)
    small = sum(1 for t in tok_counts if t < MIN_TOKENS)
    print(f"\nConfig: target={TARGET_TOKENS}  overlap={OVERLAP_TOKENS}  min={MIN_TOKENS} tokens")
    print(f"Documents: {len(files)}   Chunks: {n}")
    print(f"Tokens/chunk: min={min(tok_counts)}  avg={avg:.1f}  max={max(tok_counts)}")
    print(f"  chunks over target: {over}   under min (kept as tails): {small}")
    print("\nPer document:")
    for name, c in per_doc.items():
        print(f"  {c:4d}  {name}")
    print(f"\nWrote {n} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
