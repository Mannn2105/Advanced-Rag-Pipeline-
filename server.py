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
from retrieval_utils import normalize_hyde_query
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
    return {
        "ready": MODELS_LOADED,
        "chunks": len(nodes_store) if nodes_store else 0,
        "documents": len(build_document_catalog(nodes_store)) if nodes_store else 0,
        "snapshot_loaded": nodes_store is not None,
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

        current_stage = "decompose"
        try:
            # ── STAGE 1: Decompose ────────────────────────────
            yield send_event("stage", {"stage": "decompose", "status": "active", "label": "Decomposing query..."})
            loop = asyncio.get_event_loop()
            sub_queries = await loop.run_in_executor(None, decompose_query, question, fast_llm)
            yield send_event("stage", {"stage": "decompose", "status": "done", "label": f"Split into {len(sub_queries)} sub-queries"})
            yield send_event("sub_queries", {"sub_queries": sub_queries})

            current_stage = "hyde"
            # ── STAGE 2: HyDE ─────────────────────────────────
            yield send_event("stage", {"stage": "hyde", "status": "active", "label": "Expanding with HyDE..."})

            # Expand only the original question. Expanding every decomposed
            # sub-query concurrently can overwhelm free-tier LLM rate limits
            # and leave the request waiting for several provider calls.
            # Decomposed queries still participate in hybrid retrieval and RAG-Fusion.
            hyde_queries = {q: q for q in sub_queries}
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

            current_stage = "retrieve"
            # ── STAGE 3: Retrieve ─────────────────────────────
            yield send_event("stage", {"stage": "retrieve", "status": "active", "label": "Hybrid retrieval..."})
            all_result_lists = []
            # Build the vector and BM25 retrievers once and reuse them for every sub-query.
            vector_retriever, bm25_retriever = await loop.run_in_executor(
                None, build_hybrid_retrievers, index, nodes_store
            )
            print(
                f"[RETRIEVE] Reusing one vector retriever and one BM25 retriever "
                f"across {len(sub_queries)} sub-queries.",
                flush=True,
            )
            for q in sub_queries:
                hyde_q = hyde_queries.get(q, q)
                print(f"[RETRIEVE] Searching original and HyDE variants for: {q[:100]}", flush=True)
                hyde_res, orig_res, bm25_res = await loop.run_in_executor(
                    None,
                    lambda q=q, hyde_q=hyde_q: hybrid_retrieve_single(
                        q, hyde_q, index, nodes_store, vector_retriever, bm25_retriever
                    ),
                )
                all_result_lists.extend([hyde_res, orig_res, bm25_res])
            print(f"[RETRIEVE] Completed hybrid retrieval for {len(sub_queries)} queries.", flush=True)

            current_stage = "fusion"
            # ── STAGE 4: Fusion ───────────────────────────────
            yield send_event("stage", {"stage": "retrieve", "status": "done", "label": "Retrieval complete"})
            yield send_event("stage", {"stage": "fusion", "status": "active", "label": "RAG-Fusion merging..."})
            fused_nodes = rag_fusion(all_result_lists)
            diverse_nodes = enforce_source_diversity(fused_nodes)
            yield send_event("stage", {"stage": "fusion", "status": "done", "label": f"Fused → {len(fused_nodes)} unique chunks"})

            current_stage = "rerank"
            # ── STAGE 5: Rerank ───────────────────────────────
            yield send_event("stage", {"stage": "rerank", "status": "active", "label": "Reranking chunks..."})
            reranked = await loop.run_in_executor(None, rerank, question, diverse_nodes)
            yield send_event("stage", {"stage": "rerank", "status": "done", "label": f"Top {len(reranked)} chunks selected"})

            current_stage = "compress"
            # ── STAGE 6: Compress ─────────────────────────────
            yield send_event("stage", {"stage": "compress", "status": "active", "label": "Compressing context..."})
            raw_context = "\n\n---\n\n".join([n.node.text for n in reranked])
            compressed_context = await loop.run_in_executor(
                None, compress_context, question, raw_context, fast_llm
            )
            yield send_event("stage", {"stage": "compress", "status": "done", "label": "Context compressed"})

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

            current_stage = "generate"
            # ── STAGE 8: Generate (STREAMING) ─────────────────
            yield send_event("stage", {"stage": "generate", "status": "active", "label": "Generating answer..."})

            gen_prompt = (
                "You are a precise, factual assistant. Answer the question using ONLY "
                "the provided context.\n"
                "If the answer cannot be found in the context, reply: "
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

            current_stage = "critique"
            # ── STAGE 9: Self-Critique ────────────────────────
            yield send_event("stage", {"stage": "critique", "status": "active", "label": "Self-critique..."})
            final_answer = await loop.run_in_executor(
                None, self_critique, question, draft_answer, compressed_context, fast_llm
            )

            # If critique changed the answer, send the corrected version
            if final_answer != draft_answer:
                yield send_event("correction", {"corrected_answer": final_answer})

            yield send_event("stage", {"stage": "critique", "status": "done", "label": "Self-critique passed"})

            current_stage = "build_sources"
            # ── Build sources ─────────────────────────────────
            sources = []
            for n in reranked:
                meta = n.node.metadata
                sources.append({
                    "file": meta.get("file_name", "unknown"),
                    "page": str(meta.get("page_label", "?")),
                    "score": round(n.score, 3) if n.score else None
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
