# Deployment Troubleshooting

## Railway startup stalls during first ingestion

### Observed symptoms

During the 2026-11-21 Railway deployment of the demo corpus:

- The container found four Markdown documents in `docs/`.
- The configured embedding model loaded successfully: `BAAI/bge-large-en-v1.5`.
- The fallback parser started, but no later parsing, indexing, or application-ready log appeared.
- The Railway service remained in `DEPLOYING`, and the `/api/status` health check did not receive a successful response.
- Railway metrics showed the service at its approximately 1 GB memory limit during the same period.

These observations make memory pressure during first-time ingestion the leading hypothesis. They do not, by themselves, prove an out-of-memory kill; confirm with deployment/runtime events after changing resources.

### Why this can happen

The Docker startup command runs `python ingest.py` whenever `/app/qdrant_db/nodes.pkl` is absent, and starts Uvicorn only after ingestion exits successfully. Ingestion loads the embedding model and uses it in the semantic splitter. The original `bge-large-en-v1.5` model was relatively memory intensive for this service allocation. The project is switching to `BAAI/bge-small-en-v1.5` (384 dimensions) and a smaller embedding batch size to reduce peak RAM. The vector index must be rebuilt because its dimension changes from 1024 to 384.

### Safe remediation paths

1. **Increase the service memory allocation** enough to accommodate the configured embedding model and indexing workload. Confirm the resulting cost/plan impact before applying the change. Then redeploy and verify that ingestion reaches the index-save and completion logs.
2. **Use a smaller embedding model** as a deliberate model migration. The selected configuration is `BAAI/bge-small-en-v1.5`, 384 dimensions, with embedding batches of four. The ingestion and query paths must use the same model and batch configuration. Rebuild the vector collection; never mix vectors from different models or dimensions.
3. Keep the health-check endpoint lightweight and verify that it reports `ready: true` only after model and index loading have completed.

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
