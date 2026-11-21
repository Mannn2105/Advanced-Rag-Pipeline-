# RAG Pipeline Project Guide

## Purpose

This repository contains a document-question-answering application. Users can ask questions about a document collection, and the application retrieves relevant passages before generating a response.

The quality of an answer depends on the documents that have been ingested, the retrieval results, and the configured language-model providers. The application should not be treated as a source of truth when the indexed documents do not support an answer.

## Main components

- **FastAPI server**: exposes the chat interface and API endpoints.
- **Document ingestion**: parses supported files, creates text chunks, optionally enriches chunks with contextual descriptions, and saves the searchable index.
- **Vector retrieval**: uses local Qdrant storage for semantic similarity search.
- **Keyword retrieval**: uses BM25 to find passages that share important terms with the question.
- **Retrieval fusion and reranking**: combines candidate passages, applies source diversity, and reranks results.
- **Answer generation**: sends the selected context to the configured language model and streams the response to the browser.
- **Conversation history**: stores conversation data in a JSON file under the configured storage directory.

## Query flow

At a high level, a question moves through these stages:

1. Query decomposition creates focused sub-questions.
2. HyDE may generate hypothetical search text to improve retrieval.
3. Vector and BM25 retrieval collect candidate passages.
4. RAG-Fusion combines candidate lists and source-diversity rules limit repeated sources.
5. A reranker orders the most relevant passages.
6. Context compression prepares a concise context for generation.
7. A retrieval-sufficiency check can trigger web fallback when configured.
8. The final model generates an answer using the selected context.
9. A self-critique stage can review the generated answer.

The exact stages that run depend on provider configuration and the current implementation.

## Storage

The default paths are:

- `./docs/` for source documents to ingest.
- `./qdrant_db/` for the local vector database and the BM25 node snapshot.

In the hosted Railway service, `/app/qdrant_db` is mounted to persistent storage. Source documents placed in `docs/` are part of the deployed application image unless a separate persistent document-storage strategy is configured.

## Supported input formats

The ingestion script currently looks for PDF, TXT, Markdown, DOCX, and CSV files in the docs directory. Parsing behavior depends on the optional parser dependencies installed in the environment.

## Important limitations

- An empty docs directory produces no searchable corpus.
- A missing `nodes.pkl` means the BM25 node snapshot has not been generated.
- A successful web deployment does not mean ingestion or model-provider configuration succeeded.
- A generated answer can be incomplete or incorrect; verify important claims against the cited/retrieved source material.
- The current shared-login gate is not per-user authorization. Conversation history is not isolated by user.
