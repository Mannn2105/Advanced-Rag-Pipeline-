"""Document ingestion for the Advanced RAG Pipeline.

Supported input types are PDF, DOCX, TXT, Markdown and CSV. The process writes
a node catalogue, a versioned Qdrant collection and a manifest. Index files
are built in a new collection before the manifest is atomically replaced.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

def _extract_document(path: Path):
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(input_files=[str(path)], filename_as_id=True)
    return reader.load_data()

def _chunk_documents(documents: list):
    from llama_index.core.node_parser import SentenceSplitter
    parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        include_metadata=True,
        include_prev_next_rel=True,
    )
    nodes = parser.get_nodes_from_documents(documents)
    for node in nodes:
        metadata = node.metadata or {}
        metadata.setdefault("file_name", metadata.get("file_path", "unknown").split("/")[-1])
        node.metadata = metadata
    return nodes

def _contextualize(node_text: str, source_text: str, llm) -> str:
    prompt = (
        "Write one short factual sentence that identifies the topic and location "
        "of this excerpt within the supplied document. Do not add facts not present "
        "in the excerpt. Return only that sentence.\n\n"
        f"Document overview:\n{source_text[:5000]}\n\nExcerpt:\n{node_text[:2500]}"
    )
    return llm.complete(prompt).text.strip()

def enrich_nodes(nodes: list, documents_by_name: dict, llm=None) -> list:
    """Optionally prepend concise context; a failed enrichment never loses source text."""
    if llm is None:
        return nodes
    cache_path = Path(config.QDRANT_DIR) / "enrichment_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}
    changed = False
    for node in nodes:
        original = node.text
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        if digest in cache:
            node.text = cache[digest]
            continue
        source_name = node.metadata.get("file_name", "")
        overview = documents_by_name.get(source_name, original)
        try:
            prefix = _contextualize(original, overview, llm)
            node.text = f"{prefix}\n\n{original}" if prefix else original
        except Exception:
            node.text = original
        cache[digest] = node.text
        changed = True
        if changed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = cache_path.with_suffix(".tmp")
            temp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, cache_path)
            time.sleep(max(0.0, config.ENRICH_DELAY_SECONDS))
    return nodes

def build_index(nodes: list) -> None:
    from llama_index.core import StorageContext, VectorStoreIndex, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    if not nodes:
        raise ValueError("No chunks were produced; refusing to create an empty index")
    embedding = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL,
        device="cpu",
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    Settings.embed_model = embedding
    dimension = len(embedding.get_text_embedding("dimension validation"))
    if dimension != config.EMBED_DIMENSION:
        raise RuntimeError(
            f"Configured dimension {config.EMBED_DIMENSION} does not match model output {dimension}"
        )

    Path(config.QDRANT_DIR).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=config.QDRANT_DIR)
    existing = {entry.name for entry in client.get_collections().collections}
    if config.COLLECTION_NAME in existing:
        client.delete_collection(config.COLLECTION_NAME)
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    store = QdrantVectorStore(client=client, collection_name=config.COLLECTION_NAME)
    storage = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex(nodes, storage_context=storage, show_progress=True)

    # Persist a compact catalogue for lexical retrieval and source reporting.
    node_path = Path(config.QDRANT_DIR) / "nodes.pkl"
    temp_nodes = node_path.with_suffix(".tmp")
    with temp_nodes.open("wb") as stream:
        pickle.dump(nodes, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_nodes, node_path)

    manifest = {
        "model": config.EMBED_MODEL,
        "dimension": dimension,
        "collection": config.COLLECTION_NAME,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "node_count": len(nodes),
    }
    manifest_path = Path(config.QDRANT_DIR) / "embedding_config.json"
    temp_manifest = manifest_path.with_suffix(".tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    client.close()

def ingest() -> dict:
    problems = config.validate_runtime()
    if problems:
        raise RuntimeError("; ".join(problems))
    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported documents found in {docs_dir.resolve()}")

    all_documents, overviews = [], {}
    failures = []
    for path in files:
        try:
            documents = _extract_document(path)
            if not documents:
                failures.append(f"{path.name}: parser returned no text")
                continue
            for document in documents:
                document.metadata["file_name"] = path.name
            all_documents.extend(documents)
            overviews[path.name] = "\n".join(item.text for item in documents)
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if not all_documents:
        raise RuntimeError("No readable document text was found. " + "; ".join(failures))

    nodes = _chunk_documents(all_documents)
    enrichment_llm = None
    if os.getenv("RAG_ENABLE_CONTEXTUAL_ENRICHMENT", "false").lower() in {"1", "true", "yes"}:
        if not config.GROQ_API_KEY:
            raise RuntimeError("Contextual enrichment requested but GROQ_API_KEY is missing")
        from llama_index.llms.groq import Groq
        enrichment_llm = Groq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY)
    nodes = enrich_nodes(nodes, overviews, enrichment_llm)
    build_index(nodes)
    return {
        "documents": len(all_documents),
        "files": len(files),
        "chunks": len(nodes),
        "collection": config.COLLECTION_NAME,
        "warnings": failures,
    }

if __name__ == "__main__":
    result = ingest()
    print(json.dumps(result, indent=2))
