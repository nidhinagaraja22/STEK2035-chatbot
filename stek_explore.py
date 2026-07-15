"""
STEK 2035 — Task 1: Corpus Extraction & Exploration
====================================================
Run locally on your machine with the 12 STEK PDFs.

Usage:
    pip install pymupdf
    python stek_explore.py --pdf_dir ./pdfs

Outputs (in ./stek_output/):
    - texts/<name>.txt              extracted plain text per PDF (upload these later!)
    - exploration_report.txt        compact diagnostics  -> send this back for analysis
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stek_explore")

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF missing. Install with:  pip install pymupdf")

# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_pdf(path: Path) -> dict:
    """Extract text page-by-page; return text + per-file metadata."""
    doc = fitz.open(path)
    pages = []
    empty_pages = 0
    for page in doc:
        txt = page.get_text("text")
        if len(txt.strip()) < 20:
            empty_pages += 1
        pages.append(txt)
    full = "\n".join(pages)
    full = unicodedata.normalize("NFC", full)
    meta = {
        "file": path.name,
        "pages": len(doc),
        "near_empty_pages": empty_pages,          # hints at scanned/image pages
        "chars": len(full),
        "chars_per_page": round(len(full) / max(len(doc), 1)),
    }
    doc.close()
    return {"text": full, "meta": meta}


# --------------------------------------------------------------------------
# Tokenization (lightweight, exploration only — NOT the final pipeline)
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{1,}", re.UNICODE)

# Minimal German function-word list purely for exploration readability.
DE_FUNCTION_WORDS = set("""
der die das den dem des ein eine einer eines einem einen und oder aber auch
in im für mit von vom zu zum zur auf aus bei nach über unter durch um an am
ist sind war waren wird werden wurde wurden kann können soll sollen muss
müssen hat haben nicht als wie bis sich es sie er wir ihr ich man dass da
sowie bzw mehr sehr noch nur schon dies diese dieser dieses werden
""".split())


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def explore(corpus: dict[str, str]) -> str:
    lines = []
    add = lines.append
    add("=" * 70)
    add("STEK 2035 CORPUS EXPLORATION REPORT")
    add("=" * 70)

    all_tokens = []
    add("\n--- PER-DOCUMENT STATS ---")
    add(f"{'file':<45}{'tokens':>9}{'uniq':>8}")
    doc_token_counts = {}
    for name, text in corpus.items():
        toks = tokenize(text)
        lower = [t.lower() for t in toks]
        all_tokens.extend(lower)
        doc_token_counts[name] = len(toks)
        add(f"{name[:44]:<45}{len(toks):>9}{len(set(lower)):>8}")

    n_docs = len(corpus)
    total = len(all_tokens)
    vocab = Counter(all_tokens)
    add("\n--- GLOBAL ---")
    add(f"documents:            {n_docs}")
    add(f"total tokens:         {total:,}")
    add(f"vocabulary size:      {len(vocab):,}")
    add(f"hapax legomena:       {sum(1 for c in vocab.values() if c == 1):,} "
        f"({100*sum(1 for c in vocab.values() if c == 1)/max(len(vocab), 1):.1f}% of vocab)")
    lens = sorted(doc_token_counts.values())
    add(
        f"doc length min/median/max: {lens[0]:,} / {lens[len(lens)//2]:,} / {lens[-1]:,}")

    content = Counter({w: c for w, c in vocab.items()
                       if w not in DE_FUNCTION_WORDS and len(w) > 3})
    add("\n--- TOP 60 CONTENT TERMS (function words removed) ---")
    for w, c in content.most_common(60):
        add(f"  {w:<35}{c:>7}")

    # Terms appearing in (almost) every document = boilerplate candidates
    df = Counter()
    for text in corpus.values():
        df.update(set(t.lower() for t in tokenize(text)))
    ubiquitous = sorted(
        [w for w, d in df.items() if d >= max(2, int(0.9 * n_docs))
         and w not in DE_FUNCTION_WORDS and len(w) > 3],
        key=lambda w: -vocab[w])[:50]
    add("\n--- BOILERPLATE CANDIDATES (in >=90% of docs, top 50 by freq) ---")
    add(", ".join(ubiquitous))

    # Noise patterns
    add("\n--- NOISE INDICATORS ---")
    sample = "\n".join(corpus.values())
    add(f"digit-containing 'words':     {len(re.findall(r'\\S*\\d\\S*', sample)):,}")
    add(
        f"hyphen-linebreak artifacts:   {len(re.findall(r'[a-zäöüß]-\\n[a-zäöüß]', sample)):,}")
    add(
        f"URLs:                         {len(re.findall(r'https?://\\S+', sample)):,}")
    add(f"page-number-like lines:       {len(re.findall(r'(?m)^\\s*\\d{{1,3}}\\s*$', sample)):,}")
    add(
        f"ALL-CAPS lines (headers?):    {len(re.findall(r'(?m)^[A-ZÄÖÜ \\-]{{8,}}$', sample)):,}")

    # Random raw snippets so I can see real formatting
    add("\n--- RAW SNIPPETS (formatting inspection) ---")
    for name, text in list(corpus.items())[:4]:
        mid = len(text) // 2
        add(f"\n>>> {name} [middle 400 chars]:")
        add(text[mid:mid + 400].replace("\n", "⏎"))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="./pdfs",
                    help="folder containing the 12 PDFs")
    ap.add_argument("--out_dir", default="./stek_output")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "texts").mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    if not pdfs:
        sys.exit(f"No PDFs found in {pdf_dir.resolve()} — check --pdf_dir")

    log.info("Found %d PDFs", len(pdfs))
    corpus, metas = {}, []
    for p in pdfs:
        try:
            res = extract_pdf(p)
            corpus[p.name] = res["text"]
            metas.append(res["meta"])
            (out_dir / "texts" / (p.stem + ".txt")).write_text(
                res["text"], encoding="utf-8")
            log.info("OK  %-45s %5d pages %8d chars",
                     p.name[:44], res["meta"]["pages"], res["meta"]["chars"])
            if res["meta"]["near_empty_pages"] > res["meta"]["pages"] * 0.3:
                log.warning(
                    "  -> %s: many near-empty pages; may be scanned (OCR needed)", p.name)
        except Exception as exc:  # keep going on a bad file
            log.error("FAILED %s: %s", p.name, exc)

    report = explore(corpus)
    report_path = out_dir / "exploration_report.txt"
    report_path.write_text(
        report + "\n\n--- EXTRACTION META ---\n" +
        json.dumps(metas, indent=1, ensure_ascii=False),
        encoding="utf-8")
    log.info("Report written to %s", report_path.resolve())
    print("\nDONE. Send back:  stek_output/exploration_report.txt")
    print("Keep the folder:  stek_output/texts/  (we'll need it for the modeling stage)")


if __name__ == "__main__":
    main()
