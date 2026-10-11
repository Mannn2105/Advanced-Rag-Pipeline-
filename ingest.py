# TF guard — stops broken TensorFlow DLL on Windows
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ============================================================
#  ingest.py — Production-Grade Two-Pass Ingestion Pipeline
#
#  PASS 1: Docling Parsing
#    Converts any document (PDF, DOCX, TXT, MD) into a
#    structure-aware model: tables, headings, sections,
#    signature blocks all become distinct elements.
#    Falls back to SimpleDirectoryReader if Docling unavailable.
#
#  PASS 2: Contextual Enrichment (Anthropic Contextual Retrieval)
#    For every chunk, prepend a short LLM-generated context
#    that situates it within the whole document.
#    Based on Anthropic research — reduces retrieval failures 35-49%.
#    Uses Groq free tier with rate-limit safety (backoff + resume).
#
#  OUTPUT: Qdrant vector store + nodes.pkl for BM25
# ============================================================

import sys
import time
import json
import pickle
import hashlib
import importlib.util
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import config

console = Console()


# ── Docling availability check ────────────────────────────────
def _docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


# ── PASS 1A: Parse with Docling (structure-aware) ─────────────
def parse_with_docling(doc_path: str) -> list:
    """
    Parse a document using Docling's layout-aware engine.
    Returns LlamaIndex TextNodes with rich structural metadata.
    Tables, headings, signature blocks — all preserved as distinct elements.
    """
    from llama_index.readers.docling import DoclingReader
    from llama_index.node_parser.docling import DoclingNodeParser

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    docs = reader.load_data(file_path=doc_path)

    node_parser = DoclingNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents=docs)
    return nodes


# ── PASS 1B: Fallback parser (plain text, structure-aware chunking) ──
def parse_with_fallback(doc_path: str, embed_model) -> list:
    """
    Fallback when Docling isn't installed.
    Uses SemanticSplitter with a post-pass that extracts structural
    anchors (signature blocks, tables, etc.) as standalone nodes.
    """
    import re
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
    from llama_index.core.schema import TextNode

    # Load single file
    reader = SimpleDirectoryReader(input_files=[doc_path])
    documents = reader.load_data()

    # Semantic chunking
    try:
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
            include_metadata=True,
            include_prev_next_rel=True
        )
        nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
    except Exception:
        splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        nodes = splitter.get_nodes_from_documents(documents)

    # Structural anchor injection — extract end-of-doc structural blocks
    anchor_patterns = [
        r'SIGNATURES?\s*:',
        r'SIGNED\s+BY\s*:',
        r'EXECUTED\s+BY\s*:',
        r'CONCLUSION\s*[\n:]',
        r'SUMMARY\s*[\n:]',
        r'ABSTRACT\s*[\n:]',
    ]
    for doc in documents:
        for pattern in anchor_patterns:
            match = re.search(pattern, doc.text, re.IGNORECASE)
            if match:
                anchor_text = doc.text[match.start():].strip()
                if len(anchor_text) > 30:
                    anchor_node = TextNode(
                        text=anchor_text,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "structural_anchor",
                            "injected": True,
                        }
                    )
                    nodes.append(anchor_node)

    return nodes


# ── PASS 2: Contextual Enrichment (Anthropic Contextual Retrieval) ──
def enrich_chunk_with_context(
    chunk_text: str,
    full_document: str,
    groq_client,
    retries: int = 3
) -> str:
    """
    Prepend a short LLM-generated context to each chunk.
    This is Anthropic's Contextual Retrieval technique:
    https://www.anthropic.com/news/contextual-retrieval

    The context situates the chunk within the whole document,
    making it retrievable even when the chunk text alone is
    ambiguous (e.g. a signature block, a bullet list item,
    a table row without headers).

    Uses Groq free tier with exponential backoff on 429s.
    """
    prompt = (
        "<document>\n"
        f"{full_document[:8000]}\n"  # cap at 8k chars to stay within token limits
        "</document>\n\n"
        "Here is a chunk from the document:\n"
        "<chunk>\n"
        f"{chunk_text}\n"
        "</chunk>\n\n"
        "Give a short (2-3 sentence) context that situates this chunk "
        "within the overall document, to improve search retrieval. "
        "Include: document type, relevant section/article, key entities "
        "(company names, people, dates if present in chunk). "
        "Be specific and factual. Answer only with the context, nothing else."
    )

    for attempt in range(retries):
        try:
            response = groq_client.complete(prompt)
            context = response.text.strip()
            # Prepend context to chunk — this is what gets embedded
            return f"{context}\n\n{chunk_text}"
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                wait = (2 ** attempt) * 2  # 2s, 4s, 8s
                console.print(f"[yellow][ENRICH] Rate limited — waiting {wait}s...[/yellow]")
                time.sleep(wait)
            else:
                console.print(f"[yellow][ENRICH] Error: {e} — using chunk without context[/yellow]")
                return chunk_text  # return unenriched on non-rate-limit errors

    console.print("[yellow][ENRICH] Max retries reached — using chunk without context[/yellow]")
    return chunk_text


# ── Resume cache: skip already-enriched chunks ────────────────
def _chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_enrich_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_enrich_cache(cache: dict, cache_path: str):
    with open(cache_path, "w") as f:
        json.dump(cache, f)


# ── Main ingestion ─────────────────────────────────────────────
def ingest():
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.groq import Groq
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    start = time.time()

    # ── Check docs folder ─────────────────────────────────────
    if not os.path.exists(config.DOCS_DIR):
        os.makedirs(config.DOCS_DIR)
        console.print(f"[yellow]Created {config.DOCS_DIR}/ — add documents, then re-run.[/yellow]")
        sys.exit(0)

    doc_files = [
        f for f in Path(config.DOCS_DIR).iterdir()
        if f.is_file() and not f.name.startswith(".")
        and f.suffix.lower() in {".pdf", ".txt", ".md", ".docx", ".csv"}
    ]

    if not doc_files:
        console.print(f"[red]No documents found in {config.DOCS_DIR}/[/red]")
        sys.exit(0)

    console.print(f"\n[bold green]Found {len(doc_files)} document(s):[/bold green]")
    for f in doc_files:
        console.print(f"  • {f.name}")

    # ── Load embedding model ──────────────────────────────────
    console.print(f"\n[cyan][EMBED][/cyan] Loading {config.EMBED_MODEL} onto CPU...")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL, device="cpu")
    Settings.embed_model = embed_model
    console.print("[green][EMBED][/green] Embedding model ready ✓")

    # ── PASS 1: Parse all documents ───────────────────────────
    use_docling = _docling_available()
    parser_name = "Docling (structure-aware)" if use_docling else "SemanticSplitter (fallback)"
    console.print(f"\n[cyan][PARSE][/cyan] Parser: {parser_name}")

    if not use_docling:
        console.print(
            "[dim]Tip: pip install docling llama-index-readers-docling "
            "llama-index-node-parser-docling for better table/structure parsing[/dim]"
        )

    all_nodes = []
    all_documents_text = {}  # filename → full text (for contextual enrichment)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        task = progress.add_task("[cyan]Parsing documents...", total=len(doc_files))

        for doc_path in doc_files:
            progress.update(task, description=f"[cyan]Parsing {doc_path.name}...")
            try:
                if use_docling:
                    nodes = parse_with_docling(str(doc_path))
                else:
                    nodes = parse_with_fallback(str(doc_path), embed_model)

                # Store full document text for contextual enrichment
                with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                    all_documents_text[doc_path.name] = f.read()

                # Tag nodes with source file
                for node in nodes:
                    if "file_name" not in node.metadata:
                        node.metadata["file_name"] = doc_path.name

                all_nodes.extend(nodes)
                progress.advance(task)
                console.print(f"[green][PARSE][/green] {doc_path.name} → {len(nodes)} chunks ✓")

            except Exception as e:
                console.print(f"[red][PARSE][/red] Failed on {doc_path.name}: {e}")
                console.print("[yellow]Falling back to plain text loading...[/yellow]")
                nodes = parse_with_fallback(str(doc_path), embed_model)
                all_nodes.extend(nodes)
                progress.advance(task)

    console.print(f"\n[green][PARSE][/green] Total chunks after parsing: {len(all_nodes)} ✓")

    # ── PASS 2: Contextual Enrichment ─────────────────────────
    console.print("\n[cyan][ENRICH][/cyan] Starting contextual enrichment...")
    console.print(
        "[dim]Anthropic Contextual Retrieval: prepending document context to each chunk[/dim]"
    )
    console.print(
        "[dim]Reduces retrieval failures by 35-49% (Anthropic research, 2024)[/dim]"
    )
    console.print(
        "[dim]Using Groq free tier with rate-limit backoff. "
        "Progress saved — safe to Ctrl+C and resume.[/dim]"
    )

    if config.GROQ_API_KEY == "your_groq_key_here":
        console.print("[yellow][ENRICH] GROQ_API_KEY not set — skipping enrichment.[/yellow]")
        console.print("[yellow]Add your key to config.py and re-run to get the 35-49% accuracy boost.[/yellow]")
        enriched_nodes = all_nodes
    else:
        groq_client = Groq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY)

        # Load resume cache
        os.makedirs(config.QDRANT_DIR, exist_ok=True)
        cache_path = os.path.join(config.QDRANT_DIR, "enrich_cache.json")
        cache = load_enrich_cache(cache_path)

        enriched_nodes = []
        skipped = 0
        enriched_count = 0

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TimeElapsedColumn(), console=console
        ) as progress:
            task = progress.add_task("[cyan]Enriching chunks...", total=len(all_nodes))

            for i, node in enumerate(all_nodes):
                chunk_text = node.text
                chunk_id = _chunk_hash(chunk_text)

                # Check resume cache
                if chunk_id in cache:
                    node.text = cache[chunk_id]
                    enriched_nodes.append(node)
                    skipped += 1
                    progress.advance(task)
                    continue

                # Get full document for this chunk
                fname = node.metadata.get("file_name", "")
                full_doc = all_documents_text.get(fname, chunk_text)

                progress.update(task, description=f"[cyan]Enriching chunk {i+1}/{len(all_nodes)} ({fname})...")

                enriched_text = enrich_chunk_with_context(
                    chunk_text=chunk_text,
                    full_document=full_doc,
                    groq_client=groq_client
                )

                node.text = enriched_text
                cache[chunk_id] = enriched_text
                enriched_nodes.append(node)
                enriched_count += 1

                # Save cache after every chunk — safe to interrupt
                save_enrich_cache(cache, cache_path)

                # Respect Groq rate limits: ~1 second between calls
                # keeps us well under 30 RPM on the free tier
                time.sleep(config.ENRICH_DELAY_SECONDS)
                progress.advance(task)

        console.print(
            f"[green][ENRICH][/green] Done! "
            f"{enriched_count} enriched, {skipped} from cache ✓"
        )

    # ── Save nodes for BM25 ───────────────────────────────────
    os.makedirs(config.QDRANT_DIR, exist_ok=True)
    nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
    with open(nodes_path, "wb") as f:
        pickle.dump(enriched_nodes, f)
    console.print(f"[green][CHUNK][/green] Saved {len(enriched_nodes)} nodes for BM25 ✓")

    # ── Build Qdrant vector index ─────────────────────────────
    console.print(f"\n[cyan][INDEX][/cyan] Building vector index → {config.QDRANT_DIR}/...")

    qdrant_client = QdrantClient(path=config.QDRANT_DIR)
    try:
        qdrant_client.delete_collection(config.COLLECTION_NAME)
        console.print("[yellow][INDEX][/yellow] Cleared existing index")
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        enriched_nodes,
        storage_context=storage_context,
        show_progress=True
    )

    elapsed = round(time.time() - start, 1)
    console.print(f"\n[bold green]✅ Ingestion complete in {elapsed}s![/bold green]")
    console.print(f"   • Documents:  [white]{len(doc_files)}[/white]")
    console.print(f"   • Chunks:     [white]{len(enriched_nodes)}[/white]")
    console.print(f"   • Parser:     [white]{parser_name}[/white]")
    console.print(f"   • Enriched:   [white]{'Yes' if config.GROQ_API_KEY != 'your_groq_key_here' else 'No (add GROQ_API_KEY)'}[/white]")
    console.print(f"   • Stored in:  [white]{config.QDRANT_DIR}/[/white]")
    console.print("\n[bold cyan]Now run:[/bold cyan] python query.py")


if __name__ == "__main__":
    ingest()