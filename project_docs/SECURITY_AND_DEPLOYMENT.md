# Security and Deployment Readiness Review

## Scope

This is a source-level review of the current FastAPI server and configuration. It is not a penetration test, and it does not certify a deployed instance as secure.

## Findings to address before public use

### 1. Authentication is implemented as a single shared login; per-user authorization is still missing

The current `server.py` applies a server-side HTTP Basic authentication gate to the landing page, chat UI, and API routes. `/api/status` remains public. The gate reads `APP_USERNAME` and `APP_PASSWORD` from the deployment environment and fails closed if either is missing.

**Remaining risk:** everyone who knows the shared credentials has the same access. Conversation history is still shared, and there is no per-user ownership check. This is a basic private-demo barrier, not multi-user authorization.

**Required action:** keep the service private and use HTTPS. Before accepting multiple users or sensitive documents, implement individual identity and enforce owner-scoped access to every conversation and document.

### 2. CORS is not authentication

`config.py` reads allowed browser origins from `ALLOWED_ORIGINS`, and the server applies that list to CORS. This controls which browser origins can read cross-origin responses; it does not prevent direct HTTP clients from calling the API.

**Required action:** retain a restrictive origin allowlist, but implement real server-side authentication and authorization independently.

### 3. Request controls are partial; rate limits and query budgets remain missing

The query route now validates JSON shape, requires a non-empty question, caps question length (default 4,000 characters via `MAX_QUESTION_CHARS`), and limits simultaneous query streams (default 2 via `MAX_CONCURRENT_QUERIES`). Requests above the concurrency cap receive HTTP 429 with `Retry-After: 2`.

**Limitations:** the concurrency guard is process-local, not shared across multiple workers or replicas. It is not a per-IP or per-user rate limiter, does not impose a provider-spend budget, and does not yet enforce an end-to-end query timeout.

**Required action:** add distributed or gateway-level rate limits, per-user quotas after individual identity is implemented, request timeouts, and provider spending alerts/quotas.

### 4. Conversation data needs access controls

Conversation history is persisted in a JSON file under the configured Qdrant data directory. The routes that read and mutate conversations need authorization before this data is exposed by a public service.

**Required action:** scope history by authenticated user, protect the persistent volume, define retention/deletion behavior, and avoid logging sensitive prompts or answers.

### 5. Readiness should reflect usable service state

The `/api/status` route reports the model-loading flag and indexed chunk count. A reachable landing page alone does not establish that the index, model providers, and query path are operational.

**Required action:** keep liveness and readiness distinct. Readiness should only be true after the index and required dependencies are usable, and a deployment smoke test should submit a safe test query.

## Pre-public-deployment checklist

- [ ] Authentication enabled for all non-public API routes.
- [ ] Authorization scopes every conversation and document to its owner.
- [ ] Rate limits, request-size limits, timeouts, and concurrency caps configured.
- [ ] CORS allowlist contains only required front-end origins.
- [ ] Secrets stored only in the hosting provider's secret manager.
- [ ] Persistent storage access and backup/retention policies reviewed.
- [ ] Logs avoid secrets, personal documents, and full sensitive prompts.
- [ ] Health and readiness endpoints accurately reflect operational state.
- [ ] A smoke test verifies a real query and source response after deployment.
- [ ] Provider quotas and spend alerts configured.

## Current decision

Treat the service as a development/demo environment until the checklist is addressed. Do not describe the public endpoint as production-secure or production-ready based only on the page loading.

This review is a prioritized engineering checklist, not legal, compliance, or penetration-test advice.


## Query input and concurrency controls

`request_validation.py` rejects malformed query payloads, trims and bounds the question, and validates conversation IDs. The default question limit is 4,000 characters and can be changed with `MAX_QUESTION_CHARS` (must be a positive integer). `query_limiter.py` caps simultaneous expensive query streams at 2 per application process by default; configure `MAX_CONCURRENT_QUERIES` with a positive integer. Requests over the cap receive HTTP 429 and a short `Retry-After` hint. The cap is process-local, so multi-worker or multi-replica deployments need a shared limiter or gateway control. These controls are not a substitute for per-user/IP rate limiting or spend quotas.

## Implemented access gate (single shared account)

The application now has a server-side HTTP Basic authentication gate for the landing page, chat UI, and API routes. The public `/api/status` endpoint remains available for a basic readiness check. The gate reads `APP_USERNAME` and `APP_PASSWORD` from environment variables; if either is missing, protected routes fail closed with HTTP 503. Credentials are checked on the server and are not embedded in browser JavaScript.

For Railway, set both variables in the service's Variables panel using long, unique values. Use the app only over its HTTPS Railway domain. The browser should prompt for credentials when opening the protected page. Do not put these values in source control, chat, or frontend code.

**Scope limitation:** this is a single shared login, not individual user accounts. It does not provide per-user conversation ownership or isolation, and the current JSON conversation store remains shared by anyone with the shared login. It is a basic access barrier for a private demo, not a production multi-tenant security architecture. Keep the service private and do not put sensitive documents into it until owner-scoped storage, stronger identity management, rate limiting, and operational controls are implemented.
