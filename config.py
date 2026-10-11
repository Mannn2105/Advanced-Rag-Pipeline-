# ============================================================
#  config.py — All settings live here. Edit before running.
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()  # reads .env file into os.environ

# ── API Keys (all free tiers) ────────────────────────────────
GROQ_API_KEY   = os.getenv('GROQ_API_KEY', 'your_groq_key_here')       # https://console.groq.com
OPENROUTER_API_KEY  = os.getenv('OPENROUTER_API_KEY', 'your_openrouter_key_here')
COHERE_API_KEY = os.getenv('COHERE_API_KEY', 'your_cohere_key_here')     # https://dashboard.cohere.com
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', 'your_tavily_key_here')     # https://app.tavily.com

# ── Model Selection ──────────────────────────────────────────
GROQ_MODEL     = "llama-3.1-8b-instant"   # fast, free — for HyDE/CRAG/critique
OPENROUTER_MODEL ="nvidia/nemotron-3-super-120b-a12b:free"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"  # runs locally on CPU, no API needed
COHERE_RERANK  = "rerank-english-v3.0"     # best free reranker
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # sentence scoring (22MB)

# ── Retrieval Settings ───────────────────────────────────────
VECTOR_TOP_K          = 15    # chunks per vector search (per sub-query × 2: HyDE + original)
BM25_TOP_K            = 10    # chunks per BM25 search
BM25_SCORE_THRESHOLD  = 0.2   # filter junk BM25 matches below this score
RERANK_TOP_N          = 8     # chunks surviving reranking — raised to 6 to catch late-doc chunks
MAX_CHUNKS_PER_FILE   = 5     # source diversity cap per document

# ── Compression Settings (cross-encoder) ─────────────────────
COMPRESS_SCORE_THRESHOLD = 0.2   # keep sentences scoring above this (0-1 scale)
COMPRESS_TOP_K           = 25   # always keep at least this many top sentences

# ── Chunking Settings ────────────────────────────────────────
# SemanticSplitter auto-detects boundaries — these are fallback controls
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 100

# ── Pipeline Behavior ────────────────────────────────────────
VERBOSE         = True   # set False for cleaner output
MAX_WEB_RESULTS = 3      # fallback web search results if CRAG gate fails

# ── Contextual Enrichment (ingest-time) ──────────────────────
# Delay between Groq calls during enrichment — keeps under 30 RPM
ENRICH_DELAY_SECONDS = 1.2   # safe for free tier (30 RPM = 1 req/2s, we go faster since it's batched)

# ── Security ──────────────────────────────────────────────────
# Comma-separated list of allowed origins for CORS
_allowed_origins_raw = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000')
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins_raw.split(',') if origin.strip()]

# ── Paths ────────────────────────────────────────────────────
DOCS_DIR        = "./docs"
QDRANT_DIR      = "./qdrant_db"       # local file-based, no server needed
COLLECTION_NAME = "rag_collection"