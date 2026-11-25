"""Runtime configuration for the Advanced RAG application.

All deployment-specific values are read from environment variables. No API
credentials are stored in source control.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _positive_float_env(name: str, default: float) -> float:
    """Read a positive timeout value from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _csv_env(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def contextual_enrichment_enabled(environ=None) -> bool:
    """Honor the current flag, falling back to the legacy inverse skip flag."""
    env = os.environ if environ is None else environ
    if "RAG_ENABLE_CONTEXTUAL_ENRICHMENT" in env:
        return _env_bool(env.get("RAG_ENABLE_CONTEXTUAL_ENRICHMENT"))
    return not _env_bool(env.get("RAG_SKIP_CONTEXTUAL_ENRICHMENT"), True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GROQ_MODEL     = "openai/gpt-oss-20b"   # current Groq replacement for HyDE/CRAG/critique
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
RAG_LLM_TIMEOUT_SECONDS = _positive_float_env("RAG_LLM_TIMEOUT_SECONDS", 60.0)

EMBED_MODEL    = "BAAI/bge-small-en-v1.5"  # 384-dim model; matches the existing persistent Qdrant index
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "384"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "2"))
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COHERE_RERANK = os.getenv("COHERE_RERANK", "rerank-english-v3.0")

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "12"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "5"))
BM25_SCORE_THRESHOLD = float(os.getenv("BM25_SCORE_THRESHOLD", "0.0"))
COMPRESS_SCORE_THRESHOLD = float(os.getenv("COMPRESS_SCORE_THRESHOLD", "0.2"))
COMPRESS_TOP_K = int(os.getenv("COMPRESS_TOP_K", "25"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
ENRICH_DELAY_SECONDS = float(os.getenv("ENRICH_DELAY_SECONDS", "1.2"))
RAG_ENABLE_CONTEXTUAL_ENRICHMENT = contextual_enrichment_enabled()
RAG_SIMPLE_CHUNKING = _env_bool(os.getenv("RAG_SIMPLE_CHUNKING"), True)
MAX_WEB_RESULTS = int(os.getenv("MAX_WEB_RESULTS", "3"))
VERBOSE = os.getenv("VERBOSE", "true").lower() in {"1", "true", "yes", "on"}

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection_v2")
ALLOWED_ORIGINS = _csv_env(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)

def validate_runtime() -> list[str]:
    problems = []
    if EMBED_DIMENSION <= 0:
        problems.append("EMBED_DIMENSION must be a positive integer")
    if EMBED_BATCH_SIZE <= 0:
        problems.append("EMBED_BATCH_SIZE must be a positive integer")
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append("Chunk settings require CHUNK_SIZE > CHUNK_OVERLAP >= 0")
    if not COLLECTION_NAME.strip():
        problems.append("COLLECTION_NAME cannot be empty")
    return problems


def manifest_matches(manifest: dict | None) -> bool:
    """Return whether a persisted index matches the active embedding setup.

    Ingestion records additional metadata such as chunk counts and chunking
    parameters. Those fields must not make a compatible index appear stale.
    """
    if not isinstance(manifest, dict):
        return False
    expected = {
        "model": EMBED_MODEL,
        "dimension": EMBED_DIMENSION,
        "collection": COLLECTION_NAME,
    }
    return all(manifest.get(key) == value for key, value in expected.items())
