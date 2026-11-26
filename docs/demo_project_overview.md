# Advanced RAG Pipeline — Demo Knowledge Base

## Project purpose

This project demonstrates retrieval-augmented generation (RAG) for question answering over a private collection of documents. It retrieves relevant text passages from the indexed corpus and supplies those passages to a language model as context for an answer.

The application is intended as a portfolio and engineering demonstration. It is not a production multi-tenant document platform.

## Main architecture

The application has a FastAPI backend and a browser-based chat interface. The ingestion process reads supported files from the docs directory, parses them, splits their content into chunks, and stores the chunks in a local Qdrant vector database. It also writes a nodes.pkl snapshot used by BM25 keyword retrieval.

At query time, the pipeline can decompose a question into sub-questions, generate hypothetical search text (HyDE), retrieve candidates using vector search and BM25, merge rankings using RAG-Fusion, apply source diversity, rerank candidates, compress context, check retrieval sufficiency, and generate an answer.

## Retrieval components

- Vector retrieval finds semantically similar passages.
- BM25 retrieval finds passages with useful keyword overlap.
- RAG-Fusion combines ranked candidate lists.
- Source diversity limits over-representation from one source file.
- Reranking reorders candidates for relevance to the question.
- Context compression prepares a smaller context for answer generation.

## Persistence

The hosted demo mounts Railway persistent storage at /app/qdrant_db. The Qdrant database, BM25 node snapshot, and conversation history are intended to persist in this directory. Source documents are read from the docs directory in the application image.

## Current security scope

The hosted demo uses a single shared HTTP Basic login. It does not implement individual user accounts, per-user conversation isolation, or document-level permissions. Do not upload confidential, personal, or regulated information.

## Demo facts

- Backend framework: FastAPI.
- Vector store: local Qdrant.
- Keyword retrieval: BM25.
- Embedding model: BAAI/bge-large-en-v1.5.
- Conversation persistence: JSON file under the storage directory.
- Deployment target: Railway.
- This document is a sample corpus item for validating ingestion and retrieval; it is not a benchmark dataset.
