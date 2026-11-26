# Railway Production Readiness and Qdrant Volume Migration Runbook

This runbook is for the Railway deployment of the Advanced RAG Pipeline. It is intentionally conservative: do not run a migration against the live persistent volume until a recoverable backup and a maintenance window are confirmed.

## Current deployment layout

The audited production environment has:

- One Railway service: `rag-api-live`.
- Persistent volume: `rag-qdrant-data`, mounted at `/app/qdrant_db`.
- Volume capacity: 500 MB.
- Runtime port: Railway routes to container port 8080.
- Health check: `/api/status`.
- Current application index: 29 chunks; embedding model `BAAI/bge-small-en-v1.5`, dimension 384.
- Current deployment source: `main`; the integration branch is not automatically the production deployment.

These are an audit snapshot, not permanent configuration guarantees. Re-check the Railway environment before every operation.

## Why the directory layout needs attention

The ingestion pipeline builds a staging directory beside `QDRANT_DIR` and promotes it with an atomic directory rename. If the Railway volume is mounted exactly at `QDRANT_DIR`, the staging directory is created on the container filesystem outside that mount. The device check correctly refuses promotion across that filesystem boundary.

The compatible layout is:

- Volume mount remains at `/app/qdrant_db`.
- Active index becomes the child directory `/app/qdrant_db/active`.
- Set `QDRANT_DIR=/app/qdrant_db/active`.

This puts both the active index and its sibling staging directories on the same persistent filesystem.

## Mandatory pre-migration gates

Do not proceed unless every item is true:

1. Confirm the exact Railway project, production environment, service, and volume mount path.
2. Schedule a maintenance window and ensure no ingestion or chat-history writes can occur during the move.
3. Create a recoverable backup of the entire persistent volume outside that volume. Verify the backup can be read and includes the Qdrant data, `nodes.pkl`, `embedding_config.json`, chat history, and any other application files. Do not treat a Git commit or Railway deployment rollback as a data backup.
4. Record the current deployment ID and commit, current `QDRANT_DIR` behavior, volume size/usage, embedding model, dimension, and collection name.
5. Confirm the new image contains the migration helper and the startup manifest checks. CI must pass on the exact commit intended for deployment.
6. Confirm enough free volume capacity for the migration and a fresh ingestion/staging directory. The current volume is 500 MB; inspect actual usage rather than assuming it is sufficient.

If the platform does not provide a suitable independent volume backup for this deployment, stop and arrange an explicit export/restore procedure before continuing.

## Dry-run inspection

With the service still stopped only for the actual move, first inspect the proposed file list using the migration helper's default dry-run mode:

```bash
python scripts/migrate_qdrant_volume.py --root /app/qdrant_db
```

The helper requires `nodes.pkl` and `embedding_config.json`, refuses a non-empty target directory, rejects unsafe target names and symlink targets, and does not move files unless `--apply` is supplied.

Review the entire output. Confirm that all expected Qdrant files and application state are included. Do not proceed if required files are missing, unexpected files are present, or the target directory already contains data.

## Controlled migration

Only after the backup and dry-run have been verified:

1. Stop the Railway service using the approved maintenance procedure. Confirm the service is no longer writing to the volume.
2. Run the migration helper inside the service/container context with the volume mounted:

   ```bash
   python scripts/migrate_qdrant_volume.py --root /app/qdrant_db --apply --confirm-service-stopped
   ```

3. Verify `/app/qdrant_db/active/nodes.pkl` and `/app/qdrant_db/active/embedding_config.json` exist and are readable. Verify the Qdrant database files and application state are present under `active/`.
4. Set the service variable `QDRANT_DIR=/app/qdrant_db/active`. Keep the volume mount at `/app/qdrant_db`.
5. Deploy only the reviewed integration commit. Do not change unrelated secrets, model-provider keys, or other production variables.
6. Confirm Railway reports deployment `SUCCESS`, then check `/api/status` returns `ready: true` and the expected non-zero chunk count.
7. Run an authenticated, document-grounded query and inspect the returned source passages/citations. Also test an out-of-corpus question and confirm the system does not fabricate document evidence.
8. Confirm a subsequent ingestion can create a sibling staging directory and promote it successfully. This final step should be performed only with a test corpus or during a separately approved ingestion window.

## Rollback

If startup, index loading, or retrieval fails:

1. Stop the service before changing persistent files.
2. Restore the complete verified backup to the original volume layout.
3. Restore the previous application deployment and remove the child-directory `QDRANT_DIR` override so the old application reads the original root layout.
4. Verify `/api/status`, chunk count, and an authenticated source-grounded query before ending maintenance.

Do not attempt a partial file-by-file rollback while Qdrant is running. Do not delete the backup or the migration's previous-index backup until the restored/new deployment has been validated and a separate retention decision has been made.

## Security and operational notes

- Never print or paste `APP_PASSWORD`, provider API keys, or other secret values into logs, issues, or this runbook.
- The health endpoint is intentionally public for Railway health checks; protected chat/API routes still require application authentication.
- A successful health check proves process/index readiness, not user authorization, answer correctness, citation quality, or multi-user data isolation.
- The current app uses a shared login, not per-user authorization or conversation isolation. Do not describe it as multi-tenant secure.
- Do not merge or deploy this branch solely because CI passes. Production data migration is a separate, explicitly approved operation.
