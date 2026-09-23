"""
verify_synonym_fold.py — confirms the SYNONYMS fold actually landed in the
stored bm25_tokens after re-ingestion (not just in the source code).

Usage:
    python verify_synonym_fold.py
"""

import json
import chromadb
from ingestion_pipeline import CHROMA_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
all_data = collection.get(include=["documents", "metadatas"])

texts = all_data["documents"]
metadatas = all_data["metadatas"]

target_token = "grünfläche"
source_words = ["stadtgrün", "begrünung", "grünanlage", "grünzug", "grün"]

found_with_fold = 0
checked = 0

for text, meta in zip(texts, metadatas):
    lower = text.lower()
    if any(w in lower for w in source_words):
        checked += 1
        tokens = json.loads(meta["bm25_tokens"])
        if target_token in tokens:
            found_with_fold += 1
        else:
            print(f"NOT folded — source: {meta['source']}")
            print(f"  text: {text[:100]}")
            print(f"  tokens: {tokens[:15]}")

print(f"\nChecked {checked} chunks containing a Grün-family word.")
print(f"{found_with_fold}/{checked} correctly folded to '{target_token}' in bm25_tokens.")
