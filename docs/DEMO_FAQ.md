# RAG Pipeline Demo FAQ

## Why does the application say it is not ready?

The server reports ready only after model initialization, vector-index loading, and loading the BM25 node snapshot. Check the deployment logs for missing provider credentials, a missing vector collection, or a missing `qdrant_db/nodes.pkl` file.

## Why are there no answers about my documents?

The application can only answer from documents that have been added to the ingestion corpus and successfully indexed. Check that supported files exist in `docs/`, then run the ingestion process and verify that the expected chunk count is greater than zero.

## What does ingestion create?

The ingestion process parses source files, splits them into chunks, optionally adds contextual enrichment, writes `nodes.pkl` for BM25 retrieval, and builds the Qdrant vector collection.

## Which credentials are needed?

Provider credentials are configured as environment variables and must never be committed to source control.

- `OPENROUTER_API_KEY` is used by the configured final-answer model.
- `GROQ_API_KEY` is used by configured fast-model and optional contextual-enrichment paths.
- `COHERE_API_KEY` is used for Cohere reranking when enabled.
- `TAVILY_API_KEY` is used for web fallback when enabled.
- `APP_USERNAME` and `APP_PASSWORD` protect the hosted demo with a shared HTTP Basic login.

Not every optional provider is required for every ingestion or retrieval stage. Missing provider keys can disable features or prevent model initialization, depending on the code path.

## Is the hosted demo multi-user ready?

No. The current access gate uses one shared login. It does not provide individual accounts, per-user conversation ownership, or document-level permissions. Keep the demo private and do not upload sensitive or confidential documents.

## How should I evaluate answer quality?

Use a fixed set of questions with expected source documents and reference answers. Track retrieval recall, precision, reciprocal rank, and whether the final answer is supported by retrieved passages. Do not treat a small synthetic test dataset as a production benchmark.

## What should I do if an answer is unsupported?

Check whether the relevant document was ingested, inspect the retrieved source passages, and verify the provider configuration. If the corpus does not contain the answer, the system should say that the documents do not provide enough information rather than inventing a response.
