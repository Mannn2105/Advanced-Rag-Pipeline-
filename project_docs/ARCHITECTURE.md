# Architecture Notes

## Purpose

Orchestrated RAG answers questions over a collection of user-provided documents. It combines lexical and semantic retrieval, ranking, context filtering, and language-model generation behind a terminal interface and a FastAPI streaming server.

## End-to-end flow

1. **Ingestion and parsing** — `ingest.py` reads supported documents. It can use Docling for structure-aware parsing or a LlamaIndex fallback parser.
2. **Contextual enrichment** — each chunk can be prefixed with generated document context before indexing. The ingestion code includes a cache to support resuming enrichment.
3. **Indexing** — embeddings are generated with the configured Hugging Face model and stored in Qdrant. A node store is also used by BM25 retrieval.
4. **Query preparation** — the query pipeline can decompose a complex question and generate a hypothetical answer (HyDE) to broaden semantic retrieval.
5. **Hybrid retrieval** — Qdrant vector retrieval and BM25 keyword retrieval produce candidate passages.
6. **Fusion and diversity** — reciprocal rank fusion combines result lists; a per-file cap reduces the chance that one document dominates the context.
7. **Reranking** — Cohere reranks candidate passages for the user's original question.
8. **Context compression** — a local cross-encoder scores sentences for relevance and retains a compact context.
9. **Context sufficiency and fallback** — the pipeline can use Tavily when retrieved document context is judged insufficient.
10. **Generation and review** — the configured LLM generates a response, followed by a self-critique/review stage.
11. **Delivery** — the FastAPI server streams the response and source information to the browser client. A terminal query path is also provided.

## Component map

| Component | Responsibility |
|---|---|
| `config.py` | API configuration, model names, retrieval limits, and local paths |
| `ingest.py` | Document parsing, contextual enrichment, embeddings, and Qdrant indexing |
| `pipeline.py` | Query expansion, retrieval, fusion, reranking, compression, fallback, and generation |
| `query.py` | Terminal question-answering entry point |
| `server.py` | FastAPI endpoints and streaming response delivery |
| `app.html` | Browser chat interface |
| `tests.py` | Predefined question suite and test-result reporting |

## Design considerations

- **Hybrid search:** vector retrieval helps with semantic similarity; BM25 can recover exact terms, identifiers, and names.
- **Reciprocal rank fusion:** merges ranked lists without requiring their raw scores to share a common scale.
- **Source diversity:** limits repeated chunks from a single file to encourage broader evidence coverage.
- **Reranking:** spends a more expensive relevance model on a smaller candidate set.
- **Compression:** reduces prompt context, but aggressive filtering can remove useful evidence. The implementation keeps fallback behavior for very small compressed results.
- **Web fallback:** may introduce information outside the uploaded corpus. Source attribution and careful review remain important.

## Trust boundaries

Treat document text, retrieved passages, and web-search results as untrusted input. Do not assume that a retrieved instruction is safe to follow. Keep provider keys in environment variables, avoid committing private documents or local indexes, and add authentication and rate limiting before deploying the server publicly.
