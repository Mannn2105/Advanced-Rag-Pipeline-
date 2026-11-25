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
import uuid
import time
import pickle
import asyncio
import traceback
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from auth import basic_auth_valid
from request_validation import validate_conversation_id, validate_query_payload
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from query_limiter import MAX_CONCURRENT_QUERIES, QueryConcurrencyLimiter
import uvicorn

import config
from pipeline import init_models, load_index, run_pipeline, log
from pipeline import (
    decompose_query, hyde_expand, hybrid_retrieve_single,
    rag_fusion, enforce_source_diversity, rerank,
    compress_context, crag_gate, web_fallback,
    final_generate, self_critique, format_context
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(title="RAG Pipeline UI", version="2.0")
QUERY_LIMITER = QueryConcurrencyLimiter(MAX_CONCURRENT_QUERIES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single-user access gate ───────────────────────────────────
# Set APP_USERNAME and APP_PASSWORD in the deployment environment.
# HTTP Basic is used so the browser can prompt without putting a shared
# secret in frontend JavaScript. Always serve this app over HTTPS.
@app.middleware("http")
async def require_basic_auth(request: Request, call_next):
    if request.url.path == "/api/status":
        return await call_next(request)

    expected_user = os.getenv("APP_USERNAME", "")
    expected_password = os.getenv("APP_PASSWORD", "")
    if not expected_user or not expected_password:
        return JSONResponse(
            status_code=503,
            content={"error": "Application access is not configured."},
        )

    valid = basic_auth_valid(
        request.headers.get("authorization", ""),
        expected_user,
        expected_password,
    )
    if not valid:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required."},
            headers={"WWW-Authenticate": 'Basic realm="RAG Hub"'},
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
MODEL_INIT_TASK: Optional[asyncio.Task] = None


async def _initialize_models_and_index():
    global MODELS_LOADED, embed_model, fast_llm, final_llm, index, nodes_store

    nodes_path = Path(config.QDRANT_DIR) / "nodes.pkl"
    manifest_path = Path(config.QDRANT_DIR) / "embedding_config.json"
    expected_manifest = {
        "model": config.EMBED_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "collection": config.COLLECTION_NAME,
    }

    try:
        print("\nStarting background model and index initialization...")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None

        # Never rebuild a persistent vector index as a side effect of API startup.
        # Ingestion is an explicit maintenance operation so a deployment cannot
        # unexpectedly rewrite a mounted volume or exhaust its memory budget.
        if not nodes_path.is_file():
            raise RuntimeError(
                f"Persisted node catalogue is missing at {nodes_path}. "
                "Run python ingest.py explicitly after verifying the QDRANT_DIR mount."
            )
        if not config.manifest_matches(manifest):
            raise RuntimeError(
                "Persisted Qdrant index is missing or incompatible with the configured "
                "embedding model, dimension, or collection. Back up the volume and run "
                "python ingest.py explicitly after verifying the QDRANT_DIR mount."
            )

        embed_model, fast_llm, final_llm = await asyncio.to_thread(init_models)
        index = await asyncio.to_thread(load_index)

        with nodes_path.open("rb") as f:
            nodes_store = pickle.load(f)

        if not nodes_store:
            raise RuntimeError("The document index contains no chunks.")

        probe_vector = embed_model.get_query_embedding("RAG readiness check")
        if probe_vector is None or len(probe_vector) != config.EMBED_DIMENSION:
            actual_dimension = None if probe_vector is None else len(probe_vector)
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {config.EMBED_DIMENSION}, "
                f"received {actual_dimension}."
            )

        probe_results = index.as_retriever(similarity_top_k=1).retrieve("RAG readiness check")
        if not probe_results:
            raise RuntimeError("The vector index returned no results during readiness validation.")

        print(
            f"Retrieval readiness check passed: {len(nodes_store)} chunks, "
            f"{len(probe_vector)}-dimensional embeddings."
        )
        MODELS_LOADED = True
        print(f"Ready! Index loaded with {len(nodes_store)} chunks.")
    except Exception as e:
        MODELS_LOADED = False
        print(f"Model/index initialization failed: {type(e).__name__}: {e}")
        traceback.print_exc()


@app.on_event("startup")
async def startup_load_models():
    global MODEL_INIT_TASK
    print("Starting API server; model and index initialization will continue in the background.")
    MODEL_INIT_TASK = asyncio.create_task(_initialize_models_and_index())


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
        "model": config.OPENROUTER_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


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
    try:
        conversation_id = validate_conversation_id(conversation_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    history = _load_history()
    if conversation_id not in history:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return history[conversation_id]


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        conversation_id = validate_conversation_id(conversation_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    history = _load_history()
    if conversation_id in history:
        del history[conversation_id]
        _save_history(history)
    return {"status": "deleted"}


@app.put("/api/conversations/{conversation_id}/title")
async def rename_conversation(conversation_id: str, request: Request):
    try:
        conversation_id = validate_conversation_id(conversation_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
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
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Request body must be valid JSON"})

    try:
        question, conversation_id = validate_query_payload(body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if not MODELS_LOADED:
        return JSONResponse(status_code=503, content={"error": "Models not loaded yet"})
    if not QUERY_LIMITER.try_acquire():
        return JSONResponse(
            status_code=429,
            content={"error": "The query service is busy. Please retry shortly."},
            headers={"Retry-After": "2"},
        )

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

        try:
            # ── STAGE 1: Decompose ────────────────────────────
            yield send_event("stage", {"stage": "decompose", "status": "active", "label": "Decomposing query..."})
            loop = asyncio.get_event_loop()
            sub_queries = await loop.run_in_executor(None, decompose_query, question, fast_llm)
            yield send_event("stage", {"stage": "decompose", "status": "done", "label": f"Split into {len(sub_queries)} sub-queries"})
            yield send_event("sub_queries", {"sub_queries": sub_queries})

            # ── STAGE 2: HyDE ─────────────────────────────────
            yield send_event("stage", {"stage": "hyde", "status": "active", "label": "Expanding with HyDE..."})
            hyde_queries = {}
            with ThreadPoolExecutor(max_workers=len(sub_queries)) as ex:
                futures = {ex.submit(hyde_expand, q, fast_llm): q for q in sub_queries}
                for future in as_completed(futures):
                    q = futures[future]
                    try:
                        hyde_queries[q] = future.result()
                    except Exception:
                        hyde_queries[q] = q
            yield send_event("stage", {"stage": "hyde", "status": "done", "label": "HyDE expansion complete"})

            # ── STAGE 3: Retrieve ─────────────────────────────
            yield send_event("stage", {"stage": "retrieve", "status": "active", "label": "Hybrid retrieval..."})
            all_result_lists = []
            for q in sub_queries:
                hyde_q = hyde_queries[q]
                hyde_res, orig_res, bm25_res = await loop.run_in_executor(
                    None, hybrid_retrieve_single, q, hyde_q, index, nodes_store
                )
                all_result_lists.extend([hyde_res, orig_res, bm25_res])

            # ── STAGE 4: Fusion ───────────────────────────────
            yield send_event("stage", {"stage": "retrieve", "status": "done", "label": "Retrieval complete"})
            yield send_event("stage", {"stage": "fusion", "status": "active", "label": "RAG-Fusion merging..."})
            fused_nodes = rag_fusion(all_result_lists)
            diverse_nodes = enforce_source_diversity(fused_nodes)
            yield send_event("stage", {"stage": "fusion", "status": "done", "label": f"Fused → {len(fused_nodes)} unique chunks"})

            # ── STAGE 5: Rerank ───────────────────────────────
            yield send_event("stage", {"stage": "rerank", "status": "active", "label": "Reranking chunks..."})
            reranked = await loop.run_in_executor(None, rerank, question, diverse_nodes)
            yield send_event("stage", {"stage": "rerank", "status": "done", "label": f"Top {len(reranked)} chunks selected"})

            # ── STAGE 6: Compress ─────────────────────────────
            yield send_event("stage", {"stage": "compress", "status": "active", "label": "Compressing context..."})
            raw_context = format_context(reranked)
            compressed_context = await loop.run_in_executor(
                None, compress_context, question, raw_context, fast_llm
            )
            yield send_event("stage", {"stage": "compress", "status": "done", "label": "Context compressed"})

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
            except Exception:
                # Fallback: non-streaming generation
                draft = await loop.run_in_executor(
                    None, final_generate, question, compressed_context, final_llm
                )
                draft_tokens = [draft]
                # Send in small chunks to simulate streaming
                words = draft.split(" ")
                for i in range(0, len(words), 3):
                    chunk_text = " ".join(words[i:i+3])
                    if i > 0:
                        chunk_text = " " + chunk_text
                    yield send_event("token", {"token": chunk_text})
                    await asyncio.sleep(0.02)

            draft_answer = "".join(draft_tokens)
            yield send_event("stage", {"stage": "generate", "status": "done", "label": "Generation complete"})

            # ── STAGE 9: Self-Critique ────────────────────────
            yield send_event("stage", {"stage": "critique", "status": "active", "label": "Self-critique..."})
            final_answer = await loop.run_in_executor(
                None, self_critique, question, draft_answer, compressed_context, fast_llm
            )

            # If critique changed the answer, send the corrected version
            if final_answer != draft_answer:
                yield send_event("correction", {"corrected_answer": final_answer})

            yield send_event("stage", {"stage": "critique", "status": "done", "label": "Self-critique passed"})

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
            print(f"Query pipeline failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            yield send_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(QUERY_LIMITER.release),
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
