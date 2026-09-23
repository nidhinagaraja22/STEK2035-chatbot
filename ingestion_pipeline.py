"""
ingestion_pipeline.py — STEK 2035 corpus ingestion (Layer 1 + Layer 2 of architecture)

Runs chunking, cleaning, lemmatization (for BM25), embedding (for vector search),
and doc_type metadata tagging in ONE synchronized pass. This guarantees the BM25
index and the vector index are always built from the same chunk set at the same
time — re-embedding without re-lemmatizing (or vice versa) is what causes silent
retrieval failures on newly added documents.

Usage:
    python ingestion_pipeline.py

Requires:
    pip install spacy sentence-transformers chromadb
    python -m spacy download de_core_news_lg
"""

import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field

import spacy
import chromadb
import pdfplumber
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_WORDS = 100
CHUNK_OVERLAP = 20
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
CHROMA_PATH = str(Path(__file__).parent / "chroma_store")  # absolute, anchored to this
                                                             # script's own location —
                                                             # avoids silently writing to
                                                             # or reading from different
                                                             # locations depending on which
                                                             # directory a command is run from
COLLECTION_NAME = "stek2035"

# Manual synonym table for known German bureaucratic synonym pairs.
# Expand query tokens through this BEFORE running BM25, so exact-match
# search catches known synonyms without needing the embedding fallback.
# Boilerplate/administrative-artifact terms found via boilerplate_finder.py
# (document frequency > 5% of corpus). These are mostly credit-line and
# header fragments — "ERSTELLT DURCH KOKONSULT GMBH & CO. KG IM AUFTRAG DER
# STADT HEIDELBERG" and document-title/date headers repeated across chunks —
# NOT generic German stopwords. Confirmed by diagnostic testing: the same
# single chunk kept winning as "top match" for unrelated queries (Wohnungsbau,
# Grünflächen, Fachkräfte all pointed to one chunk near this boilerplate).
# Deliberately does NOT include "heidelberg"/"stadt" despite high frequency —
# needed for wrong-city detection (e.g. rejecting a München-phrased query).
DOMAIN_STOPWORDS = {
    "seite", "geben", "gut", "machen", "groß", "ort", "anderer",
    "stärken", "mensch",
    "stek", "gmbh", "co.", "kg", "auftrag", "erstellen", "dokumentation",
    "25.06.-25.07.24", "online-beteiligung",
    # REMOVED from the original list after checking against real queries:
    # "neu"     — used meaningfully in "neue Wohngebiete" (calibration query)
    # "brauchen" — used meaningfully in "Erlaubnis brauche ich" (early test case,
    #              a core permit-question scenario for this whole project)
    # "nutzen", "jahr", "ziel", "alt" — kept in, since these could plausibly
    #              appear in real future questions ("Landnutzung", "im Jahr
    #              2024", "Ziele des STEK", "historische/alte Bausubstanz") —
    #              excluding them was a frequency-based guess, not a verified
    #              safe removal. Revisit only if they show up causing problems
    #              in real testing, not preemptively.
}

SYNONYMS = {
    "erlaubnis": "genehmigung",
    "bewilligung": "genehmigung",
    "unterkunft": "wohnung",
    "unterkuenfte": "wohnung",
    "bauerlaubnis": "baugenehmigung",
    # Grünflächen word family — confirmed via term_coverage_check.py that the
    # corpus uses these related terms instead of "Grünfläche" itself:
    "stadtgrün": "grünfläche",
    "begrünung": "grünfläche",
    "grünanlage": "grünfläche",
    "grünanlagen": "grünfläche",
    "grünzug": "grünfläche",
    "grünzüge": "grünfläche",
    # REMOVED: "grün": "grünfläche" — this bare-word mapping was too broad.
    # It matched "grün" in ANY context (idioms, unrelated uses), inflating
    # "grünfläche"'s document frequency corpus-wide and LOWERING its IDF —
    # confirmed empirically: the Grünflächen query score barely moved after
    # adding this mapping (4.115 vs 4.190 raw before), because the folded
    # token became too common to carry much weight per occurrence.
}

nlp = spacy.load("de_core_news_lg")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str          # official_plan | citizen_opinion | event_documentation | working_group | historical
    year: int
    chunk_id: str = field(default="")
    bm25_tokens: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Strip repeated whitespace, page-number artifacts, and TOC dot-leaders."""
    text = re.sub(r"\.{4,}", " ", raw)          # TOC dot leaders: "Kapitel 1....... 5"
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    text = re.sub(r"\b\d{1,3}\s*$", "", text)    # trailing lone page numbers
    return text.strip()


def normalize_url(url: str) -> str:
    """Strip fragment identifiers so /95567, /95567#inhalt, /95567#navigation collapse to one."""
    return url.split("#")[0]


def extract_and_normalize_source_url(text: str) -> str | None:
    """
    Scraped chunks often embed a line like 'SOURCE_URL: https://...#fragment'.
    URL-fragment variants of the same page (#pageTop, #navigation, #inhalt)
    can have SLIGHTLY different scraped text (different embedded metadata
    like MATCHED_TOPICS), so exact-hash dedup alone doesn't catch them —
    confirmed: 4 fragment variants of heidelberg.de/HD/Leben/Wohnen.html all
    survived as separate chunks and flooded a query's top-5 results. This
    extracts the URL and strips the fragment so all variants can be deduped
    together regardless of small text differences.
    """
    match = re.search(r"SOURCE_URL:\s*(\S+)", text)
    if not match:
        return None
    return normalize_url(match.group(1))


def content_hash(text: str) -> str:
    """Used to drop near-duplicate chunks (e.g. URL-fragment variants with identical text)."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Step 2: Chunking (100-word sliding window, 20-word overlap)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < overlap:  # avoid tiny trailing fragment
            break
        chunks.append(" ".join(piece))
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Lemmatization for BM25 (MUST match the function used at query time)
# ---------------------------------------------------------------------------

def lemmatize(text: str) -> list[str]:
    """
    Content-word lemmatization: NOUN/PROPN/VERB/ADJ only, lowercase, base form.
    This exact function must be reused unchanged at query time — any drift
    between ingestion-time and query-time tokenization silently breaks BM25.

    IMPORTANT: spaCy's German lemmatizer often returns noun lemmas in their
    canonical CAPITALIZED form (e.g. "Grün") even when the input was already
    lowercased, because German lemma dictionaries store nouns capitalized.
    Without forcing .lower() here, SYNONYMS lookups silently never match
    (confirmed: 0/1779 chunks folded before this fix), since the dict keys
    are all lowercase.
    """
    doc = nlp(text.lower())
    tokens = [tok.lemma_.lower() for tok in doc if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")]
    tokens = [SYNONYMS.get(t, t) for t in tokens]
    return [t for t in tokens if t not in DOMAIN_STOPWORDS]  # fold known synonyms to a canonical form


# ---------------------------------------------------------------------------
# Step 4: Build chunks with metadata + dedup
# ---------------------------------------------------------------------------

def build_chunks(document_text: str, source: str, doc_type: str, year: int) -> list[Chunk]:
    cleaned = clean_text(document_text)
    raw_chunks = chunk_text(cleaned)

    seen_hashes = set()
    chunks = []
    for i, piece in enumerate(raw_chunks):
        h = content_hash(piece)
        if h in seen_hashes:
            continue  # drop exact/near-duplicate chunk (e.g. URL-fragment repeats)
        seen_hashes.add(h)

        chunks.append(Chunk(
            text=piece,
            source=source,
            doc_type=doc_type,
            year=year,
            chunk_id=f"{source}_{i}_{h[:8]}",
            bm25_tokens=lemmatize(piece),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Step 5: Embed + store in ChromaDB (vector index + BM25 tokens together)
# ---------------------------------------------------------------------------

def ingest_document(document_text: str, source: str, doc_type: str, year: int,
                     max_chunks: int | None = None) -> list[Chunk]:
    chunks = build_chunks(document_text, source, doc_type, year)

    # Corpus-imbalance cap (section 4.1): e.g. Online-Beteiligung 2024 capped at 500 chunks
    if max_chunks is not None and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    return chunks


def store_chunks(chunks: list[Chunk], client: chromadb.Client):
    if not chunks:
        return

    collection = client.get_or_create_collection(COLLECTION_NAME)

    passage_texts = [f"passage: {c.text}" for c in chunks]
    embeddings = embed_model.encode(passage_texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],           # raw text — shown to LLM + citizens
        embeddings=embeddings,
        metadatas=[{
            "source": c.source,
            "doc_type": c.doc_type,
            "year": c.year,
            "bm25_tokens": json.dumps(c.bm25_tokens),   # stored, never recomputed at query time
        } for c in chunks],
    )
    print(f"Stored {len(chunks)} chunks from '{chunks[0].source}' ({chunks[0].doc_type}, {chunks[0].year})")


# ---------------------------------------------------------------------------
# Real corpus loading — reads every .txt file from your input folder and
# matches it to the doc_type / year / chunk-cap table from report section 6.1
# ---------------------------------------------------------------------------

INPUT_DIR = "Input text files"

# Web-scraped content (section 4.3: heidelberg.de pages, often saved as PDF or
# HTML extracts under a topic folder). These typically aren't in the section 6.1
# metadata table by name, so they get a default doc_type unless a filename
# matches one of the DOC_METADATA_RULES keywords below (checked first either way).
SCRAPED_DIRS: list[tuple[str, str]] = [
    ("scraped_data_topics/pdf", "web_scraped"),
    ("scraped_data_topics/html", "web_scraped"),
]
SCRAPED_DEFAULT_YEAR = 2025  # adjust if your scrape date is known/different
SCRAPED_DIR = "scraped_data_topics"  # contains subfolders 'pdf'/'pdfs' and 'html'

# Filename keyword -> (doc_type, year, max_chunks). Matched case-insensitively
# against a substring of the filename. Order matters: first match wins, so put
# more specific keywords before more general ones.
DOC_METADATA_RULES: list[tuple[str, tuple[str, int, int | None]]] = [
    ("konzeptkarten",        ("map_data", 2025, 0)),              # excluded entirely, section 4.10
    ("stek_stadtentwicklungskonzept_2035", ("official_plan", 2035, None)),
    ("stek 2035",            ("official_plan", 2035, None)),
    ("mro_2035",             ("official_plan", 2025, None)),
    ("mro 2035",             ("official_plan", 2025, None)),
    ("nachhaltigkeitsbericht", ("official_report", 2025, None)),
    ("online-beteiligung",   ("citizen_opinion", 2024, 500)),      # capped, section 4.1
    ("online_beteiligung",   ("citizen_opinion", 2024, 500)),
    ("wege zu den zielen",   ("event_documentation", 2024, None)),
    ("wege_zu_den_zielen",   ("event_documentation", 2024, None)),
    ("zukunft gestalten",    ("event_documentation", 2024, None)),
    ("zukunft_gestalten",    ("event_documentation", 2024, None)),
    ("arbeitstreffen",       ("working_group", 2024, None)),
    ("ak stek 2",            ("working_group", 2024, None)),
    ("ak_stek_2",            ("working_group", 2024, None)),
    ("ak stek 1",            ("working_group", 2023, None)),
    ("ak_stek_1",            ("working_group", 2023, None)),
    ("zukunftsreise",        ("event_documentation", 2023, None)),
    ("2015",                 ("historical", 2015, None)),
    ("stadtentwicklungsplan", ("historical", 2015, None)),
]


def classify_filename(filename: str) -> tuple[str, int, int | None]:
    """Match a filename against DOC_METADATA_RULES; fall back to 'unclassified'."""
    lower = filename.lower()
    for keyword, meta in DOC_METADATA_RULES:
        if keyword in lower:
            return meta
    return ("unclassified", 0, None)


def load_corpus_from_folder(
    folder: str,
    default_doc_type: str | None = None,
    default_year: int | None = None,
) -> list[tuple[str, str, str, int, int | None]]:
    """
    Reads every .txt file in `folder`, classifies it via filename, and returns
    (text, source, doc_type, year, max_chunks) tuples ready for ingest_document().
    Files matched to 'map_data' with max_chunks=0 are skipped entirely (section 4.10).

    default_doc_type / default_year: used when a filename matches no keyword rule
    (instead of falling back to 'unclassified') — pass these for folders where you
    already know the general category, e.g. scraped web content.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"NOTE: folder not found, skipping: {folder}")
        return []

    documents = []
    for file_path in sorted(folder_path.glob("*.txt")):
        doc_type, year, max_chunks = classify_filename(file_path.name)

        if doc_type == "map_data" or max_chunks == 0:
            print(f"Skipping '{file_path.name}' — excluded per section 4.10 (map/noise data).")
            continue

        if doc_type == "unclassified":
            if default_doc_type is not None:
                doc_type, year = default_doc_type, (default_year or year)
            else:
                print(f"WARNING: '{file_path.name}' did not match any known keyword — "
                      f"tagging as 'unclassified'. Add a rule to DOC_METADATA_RULES if needed.")

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # source includes the parent folder name so duplicate filenames across
        # "Input text files" / "pdfs" / "html" don't collide as ChromaDB ids
        source = f"{folder_path.name}_{file_path.stem}"
        documents.append((text, source, doc_type, year, max_chunks))

    if not documents:
        print(f"No .txt files found in '{folder}'.")

    return documents


def load_full_corpus() -> list[tuple[str, str, str, int, int | None]]:
    """Loads the main corpus plus both scraped_data_topics subfolders."""
    all_documents = load_corpus_from_folder(INPUT_DIR)

    for scraped_dir, default_type in SCRAPED_DIRS:
        all_documents.extend(
            load_corpus_from_folder(scraped_dir, default_doc_type=default_type, default_year=SCRAPED_DEFAULT_YEAR)
        )

    return all_documents


# ---------------------------------------------------------------------------
# Run ingestion over the real corpus folder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    documents = load_full_corpus()

    # Dedup is applied per-document inside build_chunks(), but URL-fragment
    # variants (section 4.3) usually live in SEPARATE files — e.g. 95567.txt
    # and 95567_inhalt.txt with identical text — so we also dedup GLOBALLY
    # across every document before storing anything.
    seen_hashes_global: set[str] = set()
    url_to_best_doc: dict[str, tuple[str, str, str, int, int | None]] = {}  # url -> (text, source, doc_type, year, cap)
    non_url_documents: list[tuple[str, str, str, int, int | None]] = []
    total_kept, total_dropped, total_docs_skipped_url = 0, 0, 0

    # First pass: group documents by normalized URL, keeping the LONGEST
    # version of each duplicate cluster rather than just whichever file
    # happened to be read first alphabetically — a better proxy for "most
    # complete scrape" than arbitrary file order.
    for text, source, doc_type, year, cap in documents:
        normalized_url = extract_and_normalize_source_url(text[:500])
        if normalized_url is None:
            non_url_documents.append((text, source, doc_type, year, cap))
            continue

        existing = url_to_best_doc.get(normalized_url)
        if existing is None or len(text) > len(existing[0]):
            if existing is not None:
                total_docs_skipped_url += 1
            url_to_best_doc[normalized_url] = (text, source, doc_type, year, cap)
        else:
            total_docs_skipped_url += 1

    documents_to_ingest = non_url_documents + list(url_to_best_doc.values())

    # Second pass: chunk + exact-hash dedup as before
    for text, source, doc_type, year, cap in documents_to_ingest:
        chunks = ingest_document(text, source, doc_type, year, max_chunks=cap)

        deduped = []
        for c in chunks:
            h = content_hash(c.text)
            if h in seen_hashes_global:
                total_dropped += 1
                continue
            seen_hashes_global.add(h)
            deduped.append(c)
        total_kept += len(deduped)

        store_chunks(deduped, client)

    print(f"\nIngestion complete. {len(documents)} document(s) processed.")
    print(f"Chunks kept: {total_kept}, cross-file duplicates dropped: {total_dropped}")
    print(f"Whole documents skipped as URL-fragment duplicates: {total_docs_skipped_url}")
    print("Vector index and BM25 tokens are in sync.")