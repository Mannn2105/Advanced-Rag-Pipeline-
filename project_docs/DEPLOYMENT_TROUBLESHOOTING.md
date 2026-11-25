# Deployment Troubleshooting

## Railway startup stalls during first ingestion

### Observed symptoms

During the 2026-11-21 Railway deployment of the demo corpus:

- The container found four Markdown documents in `docs/`.
- The configured embedding model loaded successfully: `BAAI/bge-small-en-v1.5`.
- The fallback parser started, but no later parsing, indexing, or application-ready log appeared.
- The Railway service remained in `DEPLOYING`, and the `/api/status` health check did not receive a successful response.
- Railway metrics showed the service at its approximately 1 GB memory limit during the same period.

These observations make memory pressure during first-time ingestion the leading hypothesis. They do not, by themselves, prove an out-of-memory kill; confirm with deployment/runtime events after changing resources.

### Why this can happen

The Docker startup command runs `python ingest.py` whenever `/app/qdrant_db/nodes.pkl` is absent, and starts Uvicorn only after ingestion exits successfully. Ingestion loads the embedding model and uses it in the semantic splitter. The original `bge-large-en-v1.5` model was relatively memory intensive for this service allocation. The configured model is now `BAAI/bge-small-en-v1.5` (384 dimensions), with a smaller embedding batch size to reduce peak RAM. The vector index must be rebuilt when migrating from a 1024-dimensional index to 384 dimensions.

### Safe remediation paths

1. **Increase the service memory allocation** enough to accommodate the configured embedding model and indexing workload. Confirm the resulting cost/plan impact before applying the change. Then redeploy and verify that ingestion reaches the index-save and completion logs.
2. **Use a smaller embedding model** as a deliberate model migration. The selected configuration is `BAAI/bge-small-en-v1.5`, 384 dimensions, with a configurable embedding batch size (default 2). The ingestion and query paths must use the same model and batch configuration. Rebuild the vector collection; never mix vectors from different models or dimensions.
3. Keep the health-check endpoint lightweight and verify that it reports `ready: true` only after model and index loading have completed.

### Staged index rebuild and rollback

The configured collection is `rag_collection_v2`, separate from the older `rag_collection`. Ingestion now builds a complete replacement Qdrant database in a sibling staging directory. It writes the node catalogue and embedding manifest there, closes the local Qdrant client, and only then promotes the staged directory into the active `qdrant_db` path.

The previous active directory is retained as a sibling directory named like `qdrant_db.backup-<timestamp>-<id>`. If activation fails after the old directory is moved aside, the promotion helper attempts to restore it automatically. Chat history and the contextual-enrichment cache are copied into the staged directory before promotion. Keep the backup until the new deployment has passed authenticated query and citation checks; remove old backups only after confirming that rollback is no longer needed. Repeated ingestion can accumulate backups and consume disk space.

This is a directory-level rollback strategy for local Qdrant storage, not a zero-downtime online migration. Run ingestion while the application is stopped or before the server opens its Qdrant client. The embedding manifest records the model, dimension, and collection name so startup can detect stale indexes.

### Data-safety cautions

- Do not delete or reformat the persistent `/app/qdrant_db` volume as a troubleshooting shortcut.
- Back up the volume before rebuilding an index.
- Do not claim the deployment is healthy until Railway reports `SUCCESS`, `/api/status` returns successfully, and an authenticated end-to-end query has been tested.
- The public `/api/status` endpoint is intentionally exempt from Basic Auth; keep its response limited to readiness metadata.

### Verification checklist

- [ ] Deployment reaches terminal `SUCCESS`.
- [ ] Runtime logs show all expected documents parsed and the vector index saved.
- [ ] `GET /api/status` returns `ready: true` and a non-zero chunk count.
- [ ] Unauthenticated access to `/app` returns `401`.
- [ ] Authenticated access to `/app` succeeds.
- [ ] A question grounded in the demo corpus returns an answer with source attribution.
- [ ] A question absent from the corpus is handled without fabricated document claims.
