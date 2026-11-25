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

The current Docker startup command starts Uvicorn directly; it does not run ingestion automatically. The service therefore requires a valid prebuilt Qdrant index and node catalogue on its persistent volume before it can become ready. Ingestion loads the embedding model and uses it in the configured chunker. The original `bge-large-en-v1.5` model was relatively memory intensive for this service allocation. The configured model is now `BAAI/bge-small-en-v1.5` (384 dimensions), with a smaller embedding batch size to reduce peak RAM. The vector index must be rebuilt when migrating from a 1024-dimensional index to 384 dimensions.

### Safe remediation paths

1. **Increase the service memory allocation** enough to accommodate the configured embedding model and indexing workload. Confirm the resulting cost/plan impact before applying the change. Then redeploy and verify that ingestion reaches the index-save and completion logs.
2. **Use a smaller embedding model** as a deliberate model migration. The selected configuration is `BAAI/bge-small-en-v1.5`, 384 dimensions, with a configurable embedding batch size (default 2). The ingestion and query paths must use the same model and batch configuration. Rebuild the vector collection; never mix vectors from different models or dimensions.
3. Keep the health-check endpoint lightweight and verify that it reports `ready: true` only after model and index loading have completed.

### Staged index rebuild, persistent volumes, and rollback

The configured collection is `rag_collection_v2`, separate from the older `rag_collection`. Ingestion builds a replacement Qdrant database in a sibling staging directory, writes the node catalogue and embedding manifest, closes the local Qdrant client, and then promotes the staged directory.

**Important Railway mount constraint:** the existing Railway volume is mounted directly at `/app/qdrant_db`. That makes the configured Qdrant directory itself a mount point. A staging directory created beside it (for example, `/app/qdrant_db.staging-...`) is on the container filesystem, not inside the mounted volume. Renaming the mount point is not a safe or portable promotion strategy. The ingestion helper now checks filesystem device IDs and refuses promotion when staging and active directories are on different filesystems; it must not be bypassed.

Before using staged promotion on Railway, change the persistent volume mount to a parent directory such as `/app/data` and set `QDRANT_DIR=/app/data/qdrant_db`. The current volume contents will then appear at the new mount root and need a one-time, verified migration into the `qdrant_db` child directory while the application is stopped. Back up the volume first, validate the expected Qdrant files and manifest, and only then perform that migration. Do not change the mount point or move production data as part of an ordinary code deployment.

When the active index is a normal child directory on the same mounted filesystem as its staging sibling, the previous active directory is retained as a sibling named like `qdrant_db.backup-<timestamp>-<id>`. If activation fails after the old directory is moved aside, the promotion helper attempts to restore it automatically. Chat history and the contextual-enrichment cache are copied into the staged directory before promotion. Keep the backup until authenticated query and citation checks pass; repeated ingestion can accumulate backups and consume disk space.

This is a directory-level rollback strategy for local Qdrant storage, not a zero-downtime online migration. Run ingestion while the application is stopped and before the server opens its Qdrant client. The embedding manifest records the model, dimension, and collection name so startup can detect stale indexes.

### Data-safety cautions

- Do not delete, reformat, or remount the persistent volume until its contents are backed up and the one-time path migration is planned.
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
