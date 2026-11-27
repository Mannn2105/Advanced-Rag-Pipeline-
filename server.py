# TF guard — stops broken TensorFlow DLL from crashing sentence-transformers on Windows
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ============================================================
#  server.py — FastAPI Web Server for the RAG Pipeline
#
#  Wraps the existing pipeline with:
#   • SSE streaming for token-by-token answer delivery
#   • Pipeline stage progress events
#   • Conversation persistence (JSON file)
#   • REST endpoints for chat history management
#
#  Run:  python server.py
#  Open: http://localhost:8000
# ============================================================

import json
import time
import pickle
import asyncio
import traceback
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from pipeline import init_models, load_index, run_pipeline, log
from pipeline import (
    decompose_query, hyde_expand, hybrid_retrieve_single, build_hybrid_retrievers,
    rag_fusion, enforce_source_diversity, rerank,
    compress_context, crag_gate, web_fallback,
    final_generate, final_generate_with_fallback, self_critique
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from conversation_utils import resolve_conversation_id
from retrieval_utils import normalize_hyde_query, is_simple_query
from generation_utils import normalize_generated_answer
from document_utils import (
    build_document_catalog, get_reconstructed_document, build_document_diagnostics,
    recovered_download_details, build_recovered_documents_zip,
)
from auth_utils import is_basic_auth_valid
from urllib.parse import quote

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(title="RAG Pipeline UI", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_shared_basic_auth(request: Request, call_next):
    # Keep the platform health check public so Railway can probe startup.
    # CORS preflight requests must also pass through without credentials.
    if request.url.path == "/api/status" or request.method == "OPTIONS":
        return await call_next(request)

    username = os.getenv("APP_USERNAME", "")
    password = os.getenv("APP_PASSWORD", "")

    # Local development remains convenient when neither value is configured.
    # A partial configuration fails closed instead of silently disabling auth.
    if not username and not password:
        return await call_next(request)
    if not username or not password:
        return JSONResponse(
            status_code=503,
            content={"detail": "Application authentication is misconfigured."},
        )

    if not is_basic_auth_valid(
        request.headers.get("Authorization"),
        username,
        password,
    ):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="RAG Pipeline", charset="UTF-8"'},
        )

    return await call_next(request)

# ── Global State ──────────────────────────────────────────────
MODELS_LOADED = False
embed_model = None
fast_llm = None
final_llm = None
index = None
nodes_store = None

CHAT_HISTORY_PATH = os.path.join(config.QDRANT_DIR, "chat_history.json")


# ── Chat History Helpers ──────────────────────────────────────
def _load_history() -> dict:
    if os.path.exists(CHAT_HISTORY_PATH):
        try:
            with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_history(data: dict):
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Model Loading ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_load_models():
    global MODELS_LOADED, embed_model, fast_llm, final_llm, index, nodes_store
    print("\n🧠 Loading models and index... (this may take a minute on first run)")
    try:
        embed_model, fast_llm, final_llm = init_models()
        index = load_index()
        nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
        with open(nodes_path, "rb") as f:
            nodes_store = pickle.load(f)
        MODELS_LOADED = True
        print(f"✅ Ready! Index loaded with {len(nodes_store)} chunks.\n")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("   Make sure you've run 'python ingest.py' first.\n")


# ── Serve Frontend ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    landing_path = os.path.join(os.path.dirname(__file__), "landing.html")
    async with aiofiles.open(landing_path, mode="r", encoding="utf-8") as f:
        content = await f.read()
        return HTMLResponse(content=content)


@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    app_path = os.path.join(os.path.dirname(__file__), "app.html")
    async with aiofiles.open(app_path, mode="r", encoding="utf-8") as f:
        content = await f.read()
        return HTMLResponse(content=content)


# ── Status Endpoint ───────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
    return {
        "ready": MODELS_LOADED,
        "chunks": len(nodes_store) if nodes_store else 0,
        "documents": len(build_document_catalog(nodes_store)) if nodes_store else 0,
        "snapshot_loaded": nodes_store is not None,
        "snapshot_exists": os.path.isfile(nodes_path),
        "qdrant_dir": os.path.abspath(config.QDRANT_DIR),
        "model": config.OPENROUTER_MODEL,
        "embed_model": config.EMBED_MODEL,
    }



# ── Recovered Document Library ─────────────────────────────────
@app.get("/api/documents/diagnostics")
async def document_diagnostics():
    """Expose safe health metadata to verify the persisted text snapshot."""
    nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
    return build_document_diagnostics(nodes_store, nodes_path)


@app.get("/api/documents")
async def list_documents():
    """List source documents represented in the persistent indexed chunks."""
    if not nodes_store:
        return []
    return build_document_catalog(nodes_store)


@app.get("/api/documents/{document_id}")
async def preview_document(document_id: str):
    """Return reconstructed text for a document represented in nodes.pkl."""
    if not nodes_store:
        return JSONResponse(status_code=503, content={"error": "Document index is not loaded"})
    document = get_reconstructed_document(nodes_store, document_id)
    if document is None:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    return {
        "id": document["id"],
        "name": document["name"],
        "text": document["text"],
        "chunk_count": document["chunk_count"],
        "pages": document["pages"],
        "reconstructed": True,
        "notice": "Recovered from indexed text chunks. Original PDF/DOCX formatting and embedded media are not available."
    }


@app.get("/api/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download reconstructed indexed text without exposing filesystem paths."""
    if not nodes_store:
        return JSONResponse(status_code=503, content={"error": "Document index is not loaded"})
    document = get_reconstructed_document(nodes_store, document_id)
    if document is None:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    filename, media_type = recovered_download_details(document["name"])
    safe_header = quote(filename)
    return Response(
        content=document["text"],
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_header}"}
    )


@app.get("/api/documents/archive/download-all")
async def download_all_documents():
    """Download all recoverable indexed text files in one ZIP archive."""
    if not nodes_store:
        return JSONResponse(status_code=503, content={"error": "Document index is not loaded"})
    archive = build_recovered_documents_zip(nodes_store)
    if not archive:
        return JSONResponse(status_code=404, content={"error": "No recoverable documents found"})
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="recovered-documents.zip"'}
    )


# ── Conversation Endpoints ────────────────────────────────────
@app.get("/api/conversations")
async def list_conversations():
    history = _load_history()
    convos = []
    for cid, data in history.items():
        messages = data.get("messages", [])
        title = data.get("title", "Untitled")
        created = data.get("created", "")
        updated = data.get("updated", "")
        convos.append({
            "id": cid,
            "title": title,
            "message_count": len(messages),
            "created": created,
            "updated": updated,
        })
    # Sort by updated time, newest first
    convos.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return convos


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    history = _load_history()
    if conversation_id not in history:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return history[conversation_id]


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    history = _load_history()
    if conversation_id in history:
        del history[conversation_id]
        _save_history(history)
    return {"status": "deleted"}


@app.put("/api/conversations/{conversation_id}/title")
async def rename_conversation(conversation_id: str, request: Request):
    body = await request.json()
    new_title = body.get("title", "Untitled")
    history = _load_history()
    if conversation_id in history:
        history[conversation_id]["title"] = new_title
        _save_history(history)
    return {"status": "renamed"}


# ── SSE Query Endpoint ────────────────────────────────────────
@app.post("/api/query")
async def query_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    conversation_id = resolve_conversation_id(body)

    if not question:
        return JSONResponse(status_code=400, content={"error": "Empty question"})
    if not MODELS_LOADED:
        return JSONResponse(status_code=503, content={"error": "Models not loaded yet"})

    async def event_stream():
        def send_event(event_type: str, data: dict) -> str:
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"

        # ── Save user message ─────────────────────────────────
        now = datetime.now().isoformat()
        history = _load_history()
        if conversation_id not in history:
            history[conversation_id] = {
                "title": question[:60] + ("..." if len(question) > 60 else ""),
                "created": now,
                "updated": now,
                "messages": []
            }
        history[conversation_id]["messages"].append({
            "role": "user",
            "content": question,
            "timestamp": now,
        })
        history[conversation_id]["updated"] = now
        _save_history(history)

        yield send_event("conversation_id", {"conversation_id": conversation_id})

        # Runtime diagnostics are deterministic operational data, not corpus QA.
        # Answer these requests directly so retrieval or an LLM cannot replace live
        # values with sample-document claims or incorrectly report them unavailable.
        diagnostic_terms = (
            "runtime status", "runtime diagnostic", "live diagnostic",
            "application readiness",
        )
        normalized_question = question.lower()
        if any(term in normalized_question for term in diagnostic_terms):
            nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
            runtime_storage_path = os.path.abspath(config.QDRANT_DIR)
            runtime_document_count = len(build_document_catalog(nodes_store)) if nodes_store else 0
            runtime_chunk_count = len(nodes_store) if nodes_store else 0
            snapshot_loaded = nodes_store is not None
            snapshot_exists = os.path.isfile(nodes_path)
            answer = (
                "## Live Runtime Diagnostics\n\n"
                f"- **Application ready:** {MODELS_LOADED}\n"
                f"- **Indexed documents loaded:** {runtime_document_count}\n"
                f"- **Chunks loaded:** {runtime_chunk_count}\n"
                f"- **BM25 snapshot loaded in memory:** {snapshot_loaded}\n"
                f"- **BM25 snapshot file exists:** {snapshot_exists}\n"
                f"- **Qdrant storage directory:** `{runtime_storage_path}`\n"
                f"- **Embedding model:** `{config.EMBED_MODEL}`\n"
                f"- **Generation model:** `{config.OPENROUTER_MODEL}`\n\n"
                "These values are read directly from the running application, not inferred "
                "from indexed project documentation."
            )
            yield send_event("token", {"token": answer})
            yield send_event("sources", {"sources": []})
            yield send_event("meta", {
                "source_type": "runtime_diagnostics",
                "chunks_retrieved": 0,
                "chunks_after_rerank": 0,
                "sub_queries": [],
            })
            history = _load_history()
            if conversation_id in history:
                history[conversation_id]["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                    "sources": [],
                    "meta": {"source_type": "runtime_diagnostics"},
                })
                history[conversation_id]["updated"] = datetime.now().isoformat()
                _save_history(history)
            yield send_event("done", {"status": "complete"})
            return

        current_stage = "decompose"
        try:
            request_started = time.perf_counter()
            stage_started = request_started
            stage_timings = {}

            def finish_stage(name):
                nonlocal stage_started
                now = time.perf_counter()
                stage_timings[name] = round((now - stage_started) * 1000)
                stage_started = now
                print(f"[TIMING] {name}={stage_timings[name]}ms", flush=True)

            # ── STAGE 1: Decompose ────────────────────────────
            yield send_event("stage", {"stage": "decompose", "status": "active", "label": "Decomposing query..."})
            loop = asyncio.get_event_loop()
            simple_query = is_simple_query(question)
            if simple_query:
                sub_queries = [question]
                print("[FAST PATH] Simple single-intent question; skipping query decomposition.", flush=True)
            else:
                sub_queries = await loop.run_in_executor(None, decompose_query, question, fast_llm)
            yield send_event("stage", {"stage": "decompose", "status": "done", "label": "Using original query" if simple_query else f"Split into {len(sub_queries)} sub-queries"})
            yield send_event("sub_queries", {"sub_queries": sub_queries})
            finish_stage("decompose")

            current_stage = "hyde"
            # ── STAGE 2: HyDE ─────────────────────────────────
            yield send_event("stage", {"stage": "hyde", "status": "active", "label": "Expanding with HyDE..."})

            # Expand only the original question. Expanding every decomposed
            # sub-query concurrently can overwhelm free-tier LLM rate limits
            # and leave the request waiting for several provider calls.
            # Decomposed queries still participate in hybrid retrieval and RAG-Fusion.
            hyde_queries = {q: q for q in sub_queries}
            if simple_query:
                print("[FAST PATH] Skipping HyDE for a simple question.", flush=True)
            else:
                print("[HYDE] Expanding the original query once; sub-queries use their original wording.", flush=True)
                try:
                    expanded_query = await loop.run_in_executor(
                        None, hyde_expand, question, fast_llm
                    )
                    hyde_queries[question] = normalize_hyde_query(expanded_query, question)
                    print("[HYDE] Original-query expansion completed.", flush=True)
                except Exception as hyde_error:
                    hyde_queries[question] = question
                    print(
                        f"[HYDE] Expansion failed; continuing with original query: "
                        f"{type(hyde_error).__name__}: {hyde_error}",
                        flush=True,
                    )

            yield send_event("stage", {"stage": "hyde", "status": "done", "label": "HyDE expansion complete"})
            finish_stage("hyde")

            current_stage = "retrieve"
            # ── STAGE 3: Retrieve ─────────────────────────────
            yield send_event("stage", {"stage": "retrieve", "status": "active", "label": "Hybrid retrieval..."})
            all_result_lists = []
            # Reuse retrievers, but run expensive vector search only for the
            # original question and its single HyDE expansion. The decomposed
            # sub-queries still contribute keyword recall through BM25.
            vector_retriever, bm25_retriever = await loop.run_in_executor(
                None, build_hybrid_retrievers, index, nodes_store
            )
            print(
                f"[RETRIEVE] Reusing one vector retriever and one BM25 retriever "
                f"across {len(sub_queries)} sub-queries.",
                flush=True,
            )

            if simple_query:
                hyde_res, orig_res, original_bm25 = await loop.run_in_executor(
                    None,
                    lambda: hybrid_retrieve_single(
                        question, question, index, nodes_store,
                        vector_retriever, bm25_retriever
                    ),
                )
                all_result_lists.extend([orig_res, original_bm25])
            else:
                # One dense search per request: HyDE supplies the semantic query,
                # while the original question and decomposed sub-queries retain
                # keyword recall through BM25. This avoids a second expensive
                # CPU embedding + local-Qdrant round trip.
                vector_query = hyde_queries.get(question, question)
                vector_results = await loop.run_in_executor(
                    None, vector_retriever.retrieve, vector_query
                )
                all_result_lists.append(vector_results)
                print(
                    "[RETRIEVE] One HyDE vector search completed; original and "
                    "decomposed queries use BM25.",
                    flush=True,
                )

                for q in sub_queries:
                    bm25_raw = await loop.run_in_executor(
                        None, bm25_retriever.retrieve, q
                    )
                    bm25_res = [
                        node for node in bm25_raw
                        if (node.score or 0) > config.BM25_SCORE_THRESHOLD
                    ]
                    all_result_lists.append(bm25_res)

            print(
                f"[RETRIEVE] Completed hybrid retrieval: 1 vector search maximum "
                f"and BM25 across {len(sub_queries)} queries.",
                flush=True,
            )
            finish_stage("retrieve")

            current_stage = "fusion"
            # ── STAGE 4: Fusion ───────────────────────────────
            yield send_event("stage", {"stage": "retrieve", "status": "done", "label": "Retrieval complete"})
            yield send_event("stage", {"stage": "fusion", "status": "active", "label": "RAG-Fusion merging..."})
            fused_nodes = rag_fusion(all_result_lists)
            diverse_nodes = enforce_source_diversity(fused_nodes)
            yield send_event("stage", {"stage": "fusion", "status": "done", "label": f"Fused → {len(fused_nodes)} unique chunks"})
            finish_stage("fusion_and_diversity")

            current_stage = "rerank"
            # ── STAGE 5: Rerank ───────────────────────────────
            yield send_event("stage", {"stage": "rerank", "status": "active", "label": "Reranking chunks..."})
            if simple_query:
                # RRF provides a useful ranking for short factual questions;
                # avoid a second CPU-heavy cross-encoder pass.
                reranked = diverse_nodes[:config.RERANK_TOP_N]
                print(f"[FAST PATH] Using top {len(reranked)} fused chunks; skipping local reranking.", flush=True)
            else:
                # Bound local cross-encoder work on the small demo corpus.
                # Keep the best RRF candidates, then rerank them for precision.
                rerank_candidates = diverse_nodes[:12]
                print(
                    f"[RERANK] Limiting cross-encoder input to {len(rerank_candidates)} "
                    f"top fusion candidates.",
                    flush=True,
                )
                reranked = await loop.run_in_executor(
                    None, rerank, question, rerank_candidates
                )
            yield send_event("stage", {"stage": "rerank", "status": "done", "label": f"Top {len(reranked)} chunks selected"})
            finish_stage("rerank")

            current_stage = "compress"
            # ── STAGE 6: Compress ─────────────────────────────
            yield send_event("stage", {"stage": "compress", "status": "active", "label": "Compressing context..."})
            raw_context = "\n\n---\n\n".join([n.node.text for n in reranked])
            if simple_query:
                # Preserve full retrieved passages; skip expensive sentence scoring.
                compressed_context = raw_context
                print("[FAST PATH] Skipping cross-encoder context compression.", flush=True)
            else:
                compressed_context = await loop.run_in_executor(
                    None, compress_context, question, raw_context, fast_llm
                )
            yield send_event("stage", {"stage": "compress", "status": "done", "label": "Context compressed"})
            finish_stage("compress")

            current_stage = "crag"
            # ── STAGE 7: CRAG Gate ────────────────────────────
            yield send_event("stage", {"stage": "crag", "status": "active", "label": "CRAG sufficiency check..."})
            crag_ok = await loop.run_in_executor(
                None, crag_gate, question, compressed_context, fast_llm
            )
            if crag_ok:
                yield send_event("stage", {"stage": "crag", "status": "done", "label": "Context sufficient ✓"})
                source_type = "docs"
            else:
                yield send_event("stage", {"stage": "crag", "status": "done", "label": "Insufficient — web fallback triggered"})
                yield send_event("stage", {"stage": "web", "status": "active", "label": "Searching the web..."})
                web_ctx = await loop.run_in_executor(
                    None, web_fallback, question, compressed_context
                )
                if web_ctx:
                    compressed_context = web_ctx
                    source_type = "web+docs"
                else:
                    source_type = "docs_partial"
                yield send_event("stage", {"stage": "web", "status": "done", "label": "Web fallback complete"})
            finish_stage("crag_and_fallback")

            current_stage = "generate"
            # ── STAGE 8: Generate (STREAMING) ─────────────────
            yield send_event("stage", {"stage": "generate", "status": "active", "label": "Generating answer..."})

            runtime_document_count = len(build_document_catalog(nodes_store)) if nodes_store else 0
            runtime_chunk_count = len(nodes_store) if nodes_store else 0
            runtime_snapshot_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
            runtime_snapshot_exists = os.path.isfile(runtime_snapshot_path)
            runtime_storage_path = os.path.abspath(config.QDRANT_DIR)
            runtime_diagnostics = (
                f"Application ready = {MODELS_LOADED}; "
                f"documents loaded = {runtime_document_count}; "
                f"chunks loaded = {runtime_chunk_count}; "
                f"BM25 snapshot loaded in memory = {nodes_store is not None}; "
                f"BM25 snapshot file exists = {runtime_snapshot_exists}; "
                "vector database = local Qdrant; "
                f"configured Qdrant storage directory = {runtime_storage_path}; "
                "hosted Railway deployment uses a persistent volume for this directory."
            )

            gen_prompt = (
                "You are a precise, factual assistant. Answer in clear, readable Markdown, "
                "not JSON or another data wrapper. Use ONLY the provided context, except "
                "that the trusted runtime configuration and live diagnostics below are "
                "authoritative for details about this running application.\n"
                f"Trusted runtime configuration: embedding model = {config.EMBED_MODEL}; "
                f"generation model = {config.OPENROUTER_MODEL}.\n"
                f"Trusted live runtime diagnostics for this request: {runtime_diagnostics}\n"
                "Treat these live counts and status values as current for this request. "
                "Clearly distinguish live runtime facts from sample-document claims. "
                "Do not say that live counts are unavailable when they are present here.\n"
                "If a document conflicts with the runtime configuration, clearly distinguish "
                "what the document says from what this deployment actually uses.\n"
                "Implementation details for this deployment: simple questions use a fast path "
                "with the original query for vector and BM25 retrieval; complex questions are "
                "decomposed (the original question is retained), HyDE expands the original "
                "question once, one dense vector search uses that HyDE query, and BM25 runs "
                "for the original and decomposed queries. RAG-Fusion combines the result lists, "
                "then source-diversity rules reduce repeated sources. Complex questions use a "
                "reranking stage (configured Cohere reranker when available, otherwise a local "
                "cross-encoder); simple questions skip reranking. Both paths still apply "
                "RAG-Fusion and source-diversity rules. Simple questions preserve the retrieved "
                "passages without cross-encoder compression and skip self-critique; they still "
                "run the CRAG sufficiency check. Complex-query context compression uses local "
                "cross-encoder sentence scoring rather than an LLM summary, and complex queries "
                "run self-critique.\n"
                "Formatting rules: use concise headings and clean Markdown; use normal bullet "
                "lists or a single consistently numbered list; never emit empty bullets, "
                "standalone backslashes, or repeated duplicate sections.\n"
                "The application displays retrieved source documents and excerpts separately "
                "beneath your answer. Do not claim that supporting documents cannot be shown; "
                "do not invent filenames, pages, quotations, or citations. Answer from the "
                "provided context and let the separate source cards identify the retrieved files.\n"
                "If the answer cannot be found in the context or trusted configuration, reply: "
                "'The provided documents do not contain this information.'\n\n"
                f"Context:\n{compressed_context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )

            # Try streaming first, fall back to non-streaming
            draft_tokens = []
            try:
                stream_resp = final_llm.stream_complete(gen_prompt)
                for chunk in stream_resp:
                    token = chunk.delta
                    if token:
                        draft_tokens.append(token)
                        yield send_event("token", {"token": token})
                        await asyncio.sleep(0)  # yield control for SSE flush
            except Exception as stream_error:
                print(
                    f"⚠️ Streaming generation failed; using non-streaming fallback: "
                    f"{type(stream_error).__name__}: {stream_error}",
                    flush=True,
                )
                draft = await loop.run_in_executor(
                    None, final_generate_with_fallback, question, compressed_context, final_llm, fast_llm, stream_error
                )
                draft = normalize_generated_answer(draft)
                if not draft:
                    raise RuntimeError("Streaming and fallback generation both returned an empty answer.")
                draft_tokens = [draft]
                # Replace any partial streamed text instead of appending a duplicate answer.
                yield send_event("correction", {"corrected_answer": draft})

            draft_answer = normalize_generated_answer("".join(draft_tokens))
            if not draft_answer:
                print(
                    "⚠️ Streaming generation returned no usable text; retrying with non-streaming generation.",
                    flush=True,
                )
                draft = await loop.run_in_executor(
                    None, final_generate_with_fallback, question, compressed_context, final_llm, fast_llm
                )
                draft_answer = normalize_generated_answer(draft)
                if not draft_answer:
                    raise RuntimeError("Streaming and fallback generation both returned an empty answer.")
                draft_tokens = [draft_answer]
                yield send_event("correction", {"corrected_answer": draft_answer})

            yield send_event("stage", {"stage": "generate", "status": "done", "label": "Generation complete"})
            finish_stage("generate")

            current_stage = "critique"
            # ── STAGE 9: Self-Critique ────────────────────────
            yield send_event("stage", {"stage": "critique", "status": "active", "label": "Self-critique..."})
            if simple_query:
                final_answer = draft_answer
                print("[FAST PATH] Skipping self-critique for a simple factual question.", flush=True)
                critique_label = "Skipped for simple question"
            else:
                final_answer = await loop.run_in_executor(
                    None, self_critique, question, draft_answer, compressed_context, fast_llm
                )
                critique_label = "Self-critique passed"

            # If critique changed the answer, send the corrected version
            if final_answer != draft_answer:
                yield send_event("correction", {"corrected_answer": final_answer})

            yield send_event("stage", {"stage": "critique", "status": "done", "label": critique_label})
            finish_stage("critique")
            total_ms = round((time.perf_counter() - request_started) * 1000)
            print(f"[TIMING] total_pipeline={total_ms}ms stages={stage_timings}", flush=True)

            current_stage = "build_sources"
            # ── Build sources ─────────────────────────────────
            sources = []
            seen_source_pages = set()
            document_ids_by_name = {
                doc["name"]: doc["id"] for doc in build_document_catalog(nodes_store)
            }
            for n in reranked:
                meta = n.node.metadata
                file_name = meta.get("file_name") or meta.get("filename") or "unknown"
                page_label = str(meta.get("page_label", "?"))
                source_key = (file_name, page_label)
                if source_key in seen_source_pages:
                    continue
                seen_source_pages.add(source_key)
                excerpt = " ".join(str(getattr(n.node, "text", "") or "").split())
                safe_name = str(file_name).replace("\\", "/").split("/")[-1]
                sources.append({
                    "file": file_name,
                    "document_id": document_ids_by_name.get(safe_name),
                    "page": page_label,
                    "score": round(n.score, 3) if n.score else None,
                    "excerpt": excerpt[:360] + ("…" if len(excerpt) > 360 else "")
                })

            yield send_event("sources", {"sources": sources})
            yield send_event("meta", {
                "source_type": source_type,
                "chunks_retrieved": len(fused_nodes),
                "chunks_after_rerank": len(reranked),
                "sub_queries": sub_queries,
            })

            current_stage = "persist_assistant"
            # ── Save assistant message ────────────────────────
            history = _load_history()
            if conversation_id in history:
                history[conversation_id]["messages"].append({
                    "role": "assistant",
                    "content": final_answer,
                    "timestamp": datetime.now().isoformat(),
                    "sources": sources,
                    "meta": {
                        "source_type": source_type,
                        "chunks_retrieved": len(fused_nodes),
                        "chunks_after_rerank": len(reranked),
                    }
                })
                history[conversation_id]["updated"] = datetime.now().isoformat()
                _save_history(history)

            yield send_event("done", {"status": "complete"})

        except Exception as e:
            print(
                f"❌ Pipeline failed during '{current_stage}': "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            yield send_event("error", {
                "stage": current_stage,
                "message": str(e),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
