# Orchestrated RAG — Advanced Retrieval-Augmented Generation

A document question-answering project that combines multiple retrieval and reasoning techniques to improve the relevance of answers grounded in a user's own files. It includes a FastAPI backend, a browser-based chat interface, streaming responses, and source attribution.

The project explores how hybrid search, query expansion, reranking, context filtering, and fallback retrieval can work together in a multi-stage RAG workflow.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10.68-purple?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Project Highlights

- **Hybrid retrieval:** combines Qdrant vector search with BM25 keyword search.
- **Query decomposition:** breaks complex questions into smaller sub-queries.
- **HyDE and RAG-Fusion:** expands retrieval coverage and combines ranked results.
- **Contextual enrichment:** adds generated context to document chunks during ingestion.
- **Reranking:** uses Cohere reranking to improve the ordering of retrieved passages.
- **Context compression:** filters retrieved text before generation.
- **CRAG-style fallback:** can use Tavily web search when the document context is insufficient.
- **Answer review:** includes a self-critique stage intended to identify unsupported responses.
- **Streaming interface:** FastAPI server streams answers to a browser UI, with conversation history and source information.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| API server | FastAPI |
| RAG orchestration | LlamaIndex |
| Vector database | Qdrant |
| Keyword retrieval | BM25 |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Reranking | Cohere |
| LLM providers | Groq, OpenRouter / NVIDIA |
| Web fallback | Tavily |
| Document parsing | Docling (optional advanced parsing) |
| Interface | HTML, CSS and JavaScript |

## Architecture

The workflow is designed as a sequence of retrieval and answer-generation stages:

```text
                    User question
                          |
                 Query decomposition
                          |
                    HyDE expansion
                          |
                 +--------+--------+
                 |                 |
           Qdrant vector       BM25 keyword
              search             search
                 |                 |
                 +--------+--------+
                          |
                     RAG-Fusion
                          |
                    Cohere rerank
                          |
                  Context compression
                          |
                 Context sufficiency gate
                    /           \
             Sufficient       Insufficient
                 |                 |
                 |           Tavily search
                 |                 |
                 +--------+--------+
                          |
                    LLM generation
                          |
                    Self-critique
                          |
                  Streamed final answer
                    with source details
```

## Repository Structure

```text
rag_project/
├── docs/                # Input documents
├── qdrant_db/           # Persistent local vector store
├── graph_cache/         # Cache directory
├── config.py            # Model and retrieval configuration
├── ingest.py            # Document parsing and indexing
├── pipeline.py          # Multi-stage RAG pipeline
├── query.py             # Terminal Q&A interface
├── server.py            # FastAPI server and streaming endpoint
├── app.html             # Browser interface
├── tests.py             # Test suite
├── requirements.txt     # Python dependencies
└── README.md
```

Generated data directories and local secrets may be created at runtime and should not be committed unless intentionally required.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Mannn2105/Advanced-Rag-Pipeline-.git
cd Advanced-Rag-Pipeline-
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For optional advanced document parsing, install the Docling-related packages used by the project:

```bash
pip install docling llama-index-readers-docling llama-index-node-parser-docling
```

### 4. Configure API keys

Create a local `.env` file in the project root and add the credentials required by the providers and services enabled in your configuration.

```dotenv
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
COHERE_API_KEY=your_cohere_key
TAVILY_API_KEY=your_tavily_key
```

Keep real credentials private. Do not commit the `.env` file or paste API keys into issues, screenshots, or documentation.

### 5. Add documents and build the index

Place supported files in the `docs/` directory. The project README describes support for PDF, DOCX, TXT, Markdown, and CSV files.

Run ingestion:

```bash
python ingest.py
```

### 6. Run the application

Start the web application:

```bash
python server.py
```

Then open [http://localhost:8000](http://localhost:8000).

For terminal-based question answering:

```bash
python query.py
```

## Configuration

The main configuration is managed in `config.py`. The current project defaults documented here include:

| Setting | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Fast model for supporting pipeline tasks |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Answer generation |
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Local embedding model |
| `COHERE_RERANK` | `rerank-english-v3.0` | Reranking |
| `VECTOR_TOP_K` | `15` | Vector retrieval candidates |
| `BM25_TOP_K` | `10` | BM25 retrieval candidates |
| `RERANK_TOP_N` | `8` | Candidates retained after reranking |
| `CHUNK_SIZE` | `512` | Fallback chunk size |
| `VERBOSE` | `True` | Pipeline logging |

Check `config.py` for the authoritative settings and provider requirements for your checkout.

## Testing

Run the test suite:

```bash
python tests.py
```

Run an individual test by ID (where supported by the test runner):

```bash
python tests.py --test P-02
```

## Performance and Evaluation

The original project notes report approximate query times of 5–10 seconds on a CPU-based setup and a 35–49% reduction in retrieval failures attributed to contextual enrichment.

These figures are project-reported results, not independently verified benchmarks. Actual performance depends on document size, hardware, model/provider latency, network conditions, and configuration. Before comparing versions or presenting these figures as measured results, record the dataset, hardware, baseline, evaluation questions, retrieval metric, and repeated-run methodology.

A useful next evaluation step is to compare:
- Vector-only retrieval
- BM25-only retrieval
- Hybrid retrieval
- Hybrid retrieval with contextual enrichment and reranking

Track retrieval metrics such as Recall@k and MRR, along with answer quality, latency, and API cost.

## Security and Responsible Use

- Keep API keys in environment variables and out of source control.
- Treat retrieved documents and web results as untrusted input.
- Validate and constrain any generated actions or queries before execution.
- Review source passages when answers are important; retrieval and self-critique do not guarantee factual correctness.
- Apply rate limits and appropriate access controls before exposing the server beyond a trusted local environment.

## Possible Future Improvements

- Add a reproducible benchmark dataset and evaluation script.
- Add automated tests for ingestion, retrieval, and API endpoints.
- Support configurable retrieval ablation experiments.
- Improve observability for latency, token usage, and provider errors.
- Add deployment documentation and authentication before public hosting.

## What This Project Demonstrates

- Building a multi-stage RAG pipeline in Python
- Combining vector and keyword retrieval
- Applying query expansion, fusion, reranking, and context compression
- Integrating external model and search APIs
- Exposing an AI workflow through FastAPI and a streaming web interface
- Thinking about evaluation, security, and retrieval quality

## License

This repository is marked as MIT licensed. See the repository's `LICENSE` file for the license text.

---

Built with Python, LlamaIndex, Qdrant, FastAPI, and modern retrieval techniques.
