# Corpus 2 - changelog
Built: 2026-08-24

## Provenance
- Corpus 1 (original): vector_store/chunks.jsonl - LEFT UNTOUCHED
- Corpus 2 adds: registry metadata, authority taxonomy, citizen-opinion tags; removes near-duplicate chunks.

## Deduplication
- Chunks before: 1307
- Near-duplicate chunks removed (cosine >= 0.97): 41
- Chunks after:  1266

## Metadata added to every chunk
- document_id, document_title, source_url, publication_date, doc_type,
  authority_level, is_citizen_opinion, topic, section

## Authority-level distribution (chunks)
- Level 1: 116 chunks
- Level 2: 481 chunks
- Level 3: 97 chunks
- Level 4: 572 chunks
- Citizen-opinion chunks (is_citizen_opinion=TRUE): 572