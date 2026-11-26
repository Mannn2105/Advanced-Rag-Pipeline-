# Advanced RAG Pipeline

A Python-based document question-answering system that combines dense and sparse retrieval, query expansion, reranking, context compression, and answer generation. The project exposes the pipeline through a **FastAPI backend** and a browser-based chat interface with streaming responses and source attribution.

<p align="center">
  <a href="https://github.com/Mannn2105/Advanced-Rag-Pipeline-"><strong>Source code</strong></a>
</p>

## Hosted deployment

- **API status:** [Check service readiness](https://rag-api-live-production.up.railway.app/api/status)
- **Service:** Railway-hosted FastAPI application
- **Embedding model:** `BAAI/bge-small-en-v1.5`
- **Demo corpus:** Four included Markdown documents, previously indexed into 29 chunks
- **Access:** The chat interface is protected with HTTP Basic Authentication. Credentials are stored in Railway environment variables and are not published in this repository.

The hosted environment is configured for lightweight sentence-based chunking and skips optional contextual enrichment during initial indexing. A successful deployment status alone does not establish end-to-end retrieval quality; validate an authenticated query and its source attribution before relying on the service.

## What it demonstrates

- **Hybrid retrieval:** combines Qdrant vector search with BM25 keyword search.
- **Query expansion:** uses HyDE and query decomposition to improve retrieval coverage.
- **RAG-Fusion:** merges results from multiple searches using reciprocal rank fusion.
- **Contextual enrichment:** adds document-level context to chunks during ingestion.
- **Reranking:** applies Cohere reranking to improve the relevance of retrieved passages.
- **Context compression:** uses a cross-encoder to prioritize query-relevant content.
- **Web fallback:** can use Tavily when the configured pipeline determines that document context is insufficient.
- **Answer review:** includes a self-critique stage intended to help identify unsupported answers.
- **Streaming API and UI:** serves responses through FastAPI and a browser chat interface.

## Technology stack

| Area | Technologies |
|---|---|
| Language | Python |
| API | FastAPI |
| RAG framework | LlamaIndex |
| Vector database | Qdrant |
| Keyword retrieval | BM25 |
| Embeddings | BAAI/bge-small-en-v1.5 (configured in `config.py`) |
| Reranking | Cohere |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Model providers | Groq, OpenRouter |
| Web search fallback | Tavily |
| Document parsing | Docling (optional) |
| Interface | HTML, CSS, JavaScript |

## Architecture

The following diagram summarizes the main workflow. Query decomposition, web fallback, and some enrichment steps depend on the query and configuration.

```mermaid
flowchart TD
    A[User question] --> B[Query decomposition]
    B --> C[HyDE query expansion]
    C --> D1[Qdrant vector retrieval]
    C --> D2[BM25 keyword retrieval]
    B --> D1
    B --> D2
    D1 --> E[RAG-Fusion / reciprocal rank fusion]
    D2 --> E
    E --> F[Source diversity filtering]
    F --> G[Cohere reranking]
    G --> H[Cross-encoder context compression]
    H --> I{Context sufficient?}
    I -- Yes --> J[LLM answer generation]
    I -- No --> K[Tavily web fallback]
    K --> J
    J --> L[Answer review]
    L --> M[Stream response with sources]
```

See [Architecture Notes](project_docs/ARCHITECTURE.md) for more detail. Review the [Security and Deployment Readiness checklist](project_docs/SECURITY_AND_DEPLOYMENT.md), [Authentication and Conversation Isolation Design](project_docs/AUTHENTICATION_DESIGN.md), [Provenance Review](project_docs/PROVENANCE.md), and [Railway Deployment Troubleshooting](project_docs/DEPLOYMENT_TROUBLESHOOTING.md), including the dry-run-first Qdrant volume migration procedure before exposing the API publicly.

## Repository structure

```text
.
├── project_docs/        # Architecture, provenance, security and evaluation documentation
├── evaluation/          # Retrieval metrics, sample data and unit tests
├── scripts/             # Guarded persistent-volume migration utilities
├── .env.example         # Placeholder environment configuration
├── docs/                # Documents available for ingestion
├── config.py            # Model, retrieval and application settings
├── ingest.py            # Document parsing, enrichment and indexing
├── pipeline.py          # Retrieval and generation pipeline
├── query.py             # Terminal question-answering interface
├── server.py            # FastAPI server and streaming endpoint
├── app.html             # Browser chat interface
├── landing.html         # Project landing page
├── tests.py             # Question-based regression/evaluation runner
└── requirements.txt     # Python dependencies
```

Generated vector stores, caches, local documents, test outputs, and secrets should remain local unless intentionally prepared for sharing.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Mannn2105/Advanced-Rag-Pipeline-.git
cd Advanced-Rag-Pipeline-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For optional structure-aware parsing with Docling, install the additional packages documented by the project:

```bash
pip install docling llama-index-readers-docling llama-index-node-parser-docling
```

### 4. Configure environment variables

Create a local `.env` file in the project root. Add only the API credentials for services you intend to use:

```dotenv
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
COHERE_API_KEY=your_cohere_key
TAVILY_API_KEY=your_tavily_key
```

The repository includes `.env.example` with placeholders. Never commit real API keys or paste secrets into issue reports.

### 5. Add documents and build the index

Place supported documents (such as PDF, DOCX, TXT, Markdown, or CSV) in the runtime `docs/` directory, then run:

```bash
python ingest.py
```

### 6. Start the application

```bash
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

For terminal-based question answering:

```bash
python query.py
```

## Configuration

The main settings are defined in `config.py`. Current defaults include:

| Setting | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Supporting pipeline tasks |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Answer generation |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `COHERE_RERANK` | `rerank-english-v3.0` | Passage reranking |
| `VECTOR_TOP_K` | `15` | Vector retrieval candidates |
| `BM25_TOP_K` | `10` | BM25 retrieval candidates |
| `RERANK_TOP_N` | `8` | Candidates retained after reranking |
| `CHUNK_SIZE` | `512` | Sentence chunk size
| `RAG_SIMPLE_CHUNKING` | `true` | Use lightweight sentence chunking
| `RAG_SKIP_CONTEXTUAL_ENRICHMENT` | `true` | Disable optional ingestion-time enrichment |

Check `config.py` for the authoritative settings and provider requirements.

## Tests and evaluation

Run the question-based regression/evaluation runner:

```bash
python tests.py
```

The runner may also support selecting a group or test ID:

```bash
python tests.py --group R
python tests.py --test R-01
```

Evaluation requires the local index, source documents, and relevant provider credentials. Results depend on the selected documents, models, and configuration. Do not interpret research-reported improvements or illustrative figures as benchmark results measured by this repository unless you reproduce and document the experiment.

See [Evaluation Guide](project_docs/EVALUATION.md) for metric definitions and a reproducible comparison plan. The independent [retrieval-metrics utility](evaluation/README.md) runs without provider credentials; the hosted service still requires valid provider credentials for full chat functionality.

## Security and responsible use

- Keep credentials in environment variables; never commit a populated `.env` file.
- Treat retrieved documents and web results as untrusted input.
- Review cited source passages for important decisions; retrieval and self-critique do not guarantee factual correctness.
- Add authentication, rate limiting, and appropriate access controls before exposing the service publicly.
- Do not index confidential or personal documents unless you have permission and understand how embeddings and external providers process the data.

## Limitations and possible improvements

- Retrieval quality depends on document quality, chunking, embeddings, and retrieval configuration.
- External model and reranking services require valid credentials and may have rate or usage limits.
- Evaluation should be repeated on a representative, held-out question set before making quantitative performance claims.
- A production deployment should include authentication, monitoring, persistent storage, and explicit data-retention controls.

## License

No license file is currently included in this repository. Unless a license is added by the relevant rights holder, the source code remains subject to the default copyright rules; do not assume it is MIT-licensed or reuse it under MIT terms.

---

Built with Python, FastAPI, LlamaIndex, Qdrant, and retrieval-augmented generation techniques.
