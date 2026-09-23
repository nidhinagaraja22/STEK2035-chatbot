# corpus_v2 clusters changelog

- Base (cleaned corpus_v2): 1258 chunks
- Added: 6 official STEK cluster chunks (heidelberg.de/HD/Rathaus/cluster+N.html), document_id cluster_1..6
- Total: 1264 chunks
- Position: appended at the END (rows 1258..1263), so existing embedding rows stay aligned to chunks 0..1257
- Authority: L1 (official strategy)
- ACTION REQUIRED: re-embed the corpus so all rows have embeddings.
