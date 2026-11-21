# Demo Usage and Troubleshooting

## What can I ask?

Ask questions about the project architecture, retrieval stages, storage, deployment, and security limitations described in the other demo knowledge-base documents.

## What does an indexed document mean?

A document is indexed only after the ingestion script successfully parses it, creates chunks, writes the BM25 node snapshot, and creates the Qdrant vector collection. A file merely existing in the docs directory does not prove it was indexed.

## Why might the app report not ready?

The server loads its embedding model and language-model clients, then opens the Qdrant collection and reads qdrant_db/nodes.pkl. If the node snapshot or collection is absent, initialization cannot complete. Inspect the Railway deployment logs for the exact startup error.

## Why might a query fail even when the app is ready?

Readiness confirms that the server loaded its local models and index; it does not prove that remote provider credentials are valid. The configured pipeline uses Groq for fast query operations, OpenRouter for final answer generation, and Cohere for reranking. Missing or invalid credentials can cause query stages to fail. Tavily is used for web fallback when that feature is invoked.

## How should a demo answer be checked?

Compare the answer with the source document passages returned by retrieval. An answer should not claim facts that are absent from the indexed corpus. For questions outside the corpus, the system should acknowledge that the available documents do not contain enough information.

## How is the corpus updated?

Place supported source files in the docs directory and run the ingestion script again. In the hosted demo, the current repository-based deployment includes the docs directory in the application image. The vector index and BM25 snapshot are written to the persistent storage mount.
