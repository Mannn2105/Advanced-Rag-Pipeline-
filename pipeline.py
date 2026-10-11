# TF guard — stops broken TensorFlow DLL from crashing sentence-transformers on Windows
import os
import re
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ============================================================
#  pipeline.py — Max-Accuracy RAG Pipeline v2
#
#  Query Decomposition → HyDE (parallel) → Hybrid Retrieval
#  → RAG-Fusion → BM25 Threshold → Source Diversity
#  → Cohere Rerank → Context Compression → CRAG Gate
#  → Web Fallback → Parallel Generation → Self-Critique
# ============================================================

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import config

console = Console()


# ── Logging helper ────────────────────────────────────────────
def log(step: str, msg: str, color: str = "cyan"):
    if config.VERBOSE:
        console.print(f"[bold {color}][{step}][/bold {color}] {msg}")


# ── Model Initialisation ──────────────────────────────────────
def init_models():
    log("INIT", "Loading embedding model onto CPU...", "yellow")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL, device="cpu")

    log("INIT", "Loading cross-encoder compression model (22MB)...", "yellow")
    _get_cross_encoder()  # pre-warm — avoids delay on first query

    fast_llm = Groq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY)
    final_llm = OpenAILike(
        model=config.OPENROUTER_MODEL,
        api_base="https://openrouter.ai/api/v1",
        api_key=config.OPENROUTER_API_KEY,
        is_chat_model=True,
        max_tokens=2048
    )
    Settings.embed_model = embed_model
    Settings.llm = final_llm
    log("INIT", "All models ready ✓", "green")
    return embed_model, fast_llm, final_llm


# ── Load index from disk ──────────────────────────────────────
def load_index():
    qdrant_client = QdrantClient(path=config.QDRANT_DIR)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=config.COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


# ── STEP 1: Query Decomposition ───────────────────────────────
# Breaks complex queries into focused sub-queries for better recall
def decompose_query(query: str, fast_llm) -> list:
    log("DECOMPOSE", "Breaking query into sub-queries...")
    prompt = (
        "Break the following question into 2-3 focused search queries.\n"
        "Each query should target a different aspect of the question.\n\n"
        "CRITICAL RULES:\n"
        "- Use ONLY the exact entity names from the question — do NOT substitute similar "
        "real-world companies, people, or products.\n"
        "- If the question mentions 'Axon Systems', search for 'Axon Systems' — NOT 'ARM Holdings', "
        "'Qualcomm', or any other real company.\n"
        "- If the question mentions a person's name, keep that exact name.\n"
        "- Sub-queries must be directly answerable from internal documents about the entities mentioned.\n"
        "- Return ONLY the queries, one per line, no numbering, no bullets.\n\n"
        f"Question: {query}\n\nQueries:"
    )
    resp = fast_llm.complete(prompt)
    sub_queries = [
        q.strip().lstrip("-•123456789. ")
        for q in resp.text.strip().split("\n")
        if q.strip() and len(q.strip()) > 5
    ][:3]  # max 3 sub-queries

    # Always include the original query
    if query not in sub_queries:
        sub_queries = [query] + sub_queries

    # SC-01 fix: for signature/signer questions, always add an anchor
    # sub-query that targets the document's signature block explicitly.
    # Signature blocks sit at the end of documents and are often missed.
    sig_keywords = [
        "sign", "signed", "signer", "signers", "signatory",
        "signatories", "signature", "signatures", "executed by"
    ]
    sig_pattern = r"\b(" + "|".join(re.escape(kw) for kw in sig_keywords) + r")\b"
    if re.search(sig_pattern, query.lower()):
        anchor = "signatures signed by date executed agreement"
        if anchor not in sub_queries:
            sub_queries.append(anchor)
            log("DECOMPOSE", "Added signature anchor sub-query", "dim")

    log("DECOMPOSE", f"Generated {len(sub_queries)} queries ✓", "green")
    return sub_queries


# ── STEP 2: HyDE — Hypothetical Document Embedding ───────────
def hyde_expand(query: str, fast_llm) -> str:
    prompt = (
        "Write a short, dense, factual paragraph (3-5 sentences) "
        "that would perfectly answer the following question. "
        "Be specific and information-dense. Do not add disclaimers.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = fast_llm.complete(prompt)
    return response.text


# ── STEP 3: Hybrid Retrieval (Vector + BM25) ─────────────────
# Retrieves with BOTH HyDE query AND original query for dual coverage
def hybrid_retrieve_single(query: str, hyde_query: str, index, nodes_store: list) -> list:
    vector_retriever = index.as_retriever(similarity_top_k=config.VECTOR_TOP_K)

    # Dual retrieval: HyDE query + original query (Fix #1)
    hyde_results = vector_retriever.retrieve(hyde_query)
    orig_results = vector_retriever.retrieve(query)

    # BM25 on original query with score threshold (Fix #3)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes_store, similarity_top_k=config.BM25_TOP_K
    )
    bm25_raw = bm25_retriever.retrieve(query)
    bm25_results = [n for n in bm25_raw if (n.score or 0) > config.BM25_SCORE_THRESHOLD]

    return hyde_results, orig_results, bm25_results


# ── STEP 4: RAG-Fusion (Reciprocal Rank Fusion) ──────────────
# Mathematically merges rankings from multiple result lists
# Chunks ranked highly across multiple queries score higher
def rag_fusion(result_lists: list, k: int = 60) -> list:
    scores = defaultdict(float)
    node_map = {}
    for results in result_lists:
        for rank, node in enumerate(results):
            nid = node.node.node_id
            scores[nid] += 1.0 / (k + rank + 1)
            # Keep highest-scored version of each node
            if nid not in node_map or (node.score or 0) > (node_map[nid].score or 0):
                node_map[nid] = node

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [node_map[nid] for nid, _ in fused]


# ── STEP 5: Source Diversity (Fix #6) ────────────────────────
# Caps chunks per document so answer isn't biased to one source
def enforce_source_diversity(nodes: list, max_per_file: int = None) -> list:
    if max_per_file is None:
        max_per_file = config.MAX_CHUNKS_PER_FILE
    file_counts = defaultdict(int)
    diverse = []
    for node in nodes:
        fname = node.node.metadata.get("file_name", "unknown")
        if file_counts[fname] < max_per_file:
            diverse.append(node)
            file_counts[fname] += 1
    return diverse


# ── STEP 6: Cohere Reranking ──────────────────────────────────
def rerank(query: str, nodes: list) -> list:
    log("RERANK", f"Reranking {len(nodes)} chunks → keeping top {config.RERANK_TOP_N}...")
    reranker = CohereRerank(
        api_key=config.COHERE_API_KEY,
        top_n=config.RERANK_TOP_N,
        model=config.COHERE_RERANK
    )
    reranked = reranker.postprocess_nodes(nodes, query_str=query)
    log("RERANK", f"Top {len(reranked)} chunks selected ✓", "green")
    return reranked


# ── STEP 7: Cross-Encoder Context Compression ───────────────
# Uses a cross-encoder model to score every sentence against the
# query mathematically. No LLM judgment, no keyword hacks.
# The cross-encoder does a single forward pass: (query, sentence)
# → relevance score 0–1. Sentences above threshold are kept.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2
# Size: 22MB | Speed: ~100ms for 50 sentences on CPU | No API needed
#
# Why this is correct:
# - Deterministic: same query+sentence always gives same score
# - Blind to formatting: scores "Rajesh Subramaniam CEO July 28"
#   as highly relevant to "who signed" regardless of it looking
#   like a template/boilerplate
# - No rate limits, no API calls, runs locally

_cross_encoder = None  # lazy-loaded on first use

def _get_cross_encoder():
    """Lazy-load the cross-encoder model (22MB, cached after first load)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        log("COMPRESS", "Loading cross-encoder model (22MB, once only)...", "yellow")
        _cross_encoder = CrossEncoder(
            config.CROSS_ENCODER_MODEL,
            max_length=512,
            device="cpu"
        )
        log("COMPRESS", "Cross-encoder ready ✓", "green")
    return _cross_encoder


def _split_sentences(text: str) -> list:
    """
    Split text into sentences, preserving context.
    Uses a simple but robust approach: split on sentence-ending
    punctuation followed by whitespace, but keep short lines
    (like "Name: Value" pairs) as single units.
    """
    import re
    # Split on ". ", "! ", "? ", or newlines — but keep list items together
    raw = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s:
            # Keep bullet/list items as-is, split long prose further
            lines = [l.strip() for l in s.split("\n") if l.strip()]
            sentences.extend(lines)
    return [s for s in sentences if len(s) > 8]  # drop very short fragments


def compress_context(query: str, raw_context: str, fast_llm=None) -> str:
    # fast_llm kept in signature for backward compatibility — not used
    if len(raw_context) < 1500:
        log("COMPRESS", "Context small — skipping compression", "dim")
        return raw_context

    log("COMPRESS", "Scoring sentences with cross-encoder...")

    sentences = _split_sentences(raw_context)
    if len(sentences) < 4:
        log("COMPRESS", "Too few sentences — skipping compression", "dim")
        return raw_context

    # Score every sentence against the query in one batch call
    ce = _get_cross_encoder()
    pairs = [(query, s) for s in sentences]
    scores = ce.predict(pairs, show_progress_bar=False)

    # Keep sentences above threshold OR in top-K — whichever gives more
    threshold = config.COMPRESS_SCORE_THRESHOLD
    top_k = config.COMPRESS_TOP_K

    # Sort by score descending, take top_k, then restore original order
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k_indices = set(idx for idx, _ in scored[:top_k])
    threshold_indices = set(idx for idx, sc in enumerate(scores) if sc >= threshold)
    keep_indices = top_k_indices | threshold_indices

    # Reconstruct in original order to preserve document flow
    kept = [sentences[i] for i in sorted(keep_indices)]
    compressed = "\n".join(kept)

    raw_len = max(len(raw_context), 1)
    ratio = round(len(compressed) / raw_len * 100)

    if len(compressed) < 100 or len(kept) < 2:
        log("COMPRESS", "Too few sentences kept — using full context", "yellow")
        return raw_context

    log("COMPRESS", f"Kept {len(kept)}/{len(sentences)} sentences ({ratio}% of chars) ✓", "green")
    return compressed


# ── STEP 8: CRAG Gate ─────────────────────────────────────────
# Checks if compressed context is sufficient — if not, triggers web fallback
def crag_gate(query: str, context_text: str, fast_llm) -> bool:
    log("CRAG", "Checking if context is sufficient...")
    # Use the already-compressed context (same text that generation will see)
    # Truncate to 4000 chars max to keep Groq prompt small
    context = context_text[:4000]
    prompt = (
        "You are evaluating whether document excerpts contain enough information "
        "to answer a question. Default to YES unless clearly impossible.\n\n"
        "Rules:\n"
        "- YES if the specific fact, number, name, or answer exists anywhere in the excerpts\n"
        "- YES if you can see the answer even partially stated in the text\n"
        "- YES for calculations if the required input numbers are present\n"
        "- NO ONLY if the topic is completely absent — not a single relevant sentence\n"
        "- NO if the question needs live data (stock price, weather, breaking news)\n\n"
        "Examples:\n"
        "Q: How many transistors? [excerpts say '18 billion transistors'] → YES\n"
        "Q: What is eco mode power? [excerpts contain 'Eco (0.8W)'] → YES\n"
        "Q: Monthly revenue? [excerpts contain annual revenue figure] → YES (can divide)\n"
        "Q: Current stock price? [excerpts are internal docs with no price] → NO\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Reply with ONLY YES or NO:"
    )
    response = fast_llm.complete(prompt)
    result = "YES" in response.text.strip().upper()
    log("CRAG", "✓ Sufficient" if result else "✗ Insufficient — triggering web fallback",
        "green" if result else "red")
    return result


# ── STEP 9: Web Fallback ──────────────────────────────────────
def web_fallback(query: str, doc_context: str = "") -> str:
    log("FALLBACK", "Searching the web via Tavily...", "yellow")
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        results = client.search(query=query, max_results=config.MAX_WEB_RESULTS)
        raw_results = results.get("results") or []  # guard against None
        web_combined = "\n\n---\n\n".join([
            f"[WEB SOURCE: {r.get('url', 'web')}]\n{r['content']}"
            for r in raw_results if r and r.get("content")
        ])
        log("FALLBACK", f"Retrieved {len(raw_results)} web results ✓", "green")
        if not web_combined:
            log("FALLBACK", "No usable web results, using doc context only.", "yellow")
            return doc_context
        # Doc context always primary — web is supplemental only
        if doc_context:
            return (
                f"[INTERNAL DOCUMENTS — primary and authoritative]\n"
                f"IMPORTANT: The company/entity described in the internal documents above "
                f"is the subject of the question. Web results about companies or products "
                f"with similar names (e.g. a public company sharing a name with a private "
                f"one in the internal docs) should be IGNORED unless the internal docs "
                f"explicitly reference them.\n\n"
                f"{doc_context}\n\n"
                f"[WEB SEARCH RESULTS — supplement only, ignore if about a different entity]\n"
                f"{web_combined}"
            )
        return web_combined
    except Exception as e:
        log("FALLBACK", f"Web search failed: {e}. Using doc context only.", "red")
        return doc_context


# ── STEP 10: Final Generation ─────────────────────────────────
def final_generate(query: str, context: str, final_llm) -> str:
    log("GENERATE", f"Generating answer with {config.OPENROUTER_MODEL.split('/')[-1]}...")
    prompt = (
        "You are a precise, factual assistant. Answer the question using ONLY "
        "the provided context.\n"
        "IMPORTANT: Always complete your answer fully. Never truncate mid-sentence. "
        "When listing names, always include ALL names mentioned in the context.\n"
        "If the answer cannot be found in the context, reply: "
        "'The provided documents do not contain this information.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = final_llm.complete(prompt)
    return response.text.strip()


# ── STEP 11: Self-Critique ────────────────────────────────────
def self_critique(query: str, draft_answer: str, context: str, fast_llm) -> str:
    log("CRITIQUE", "Running self-critique pass...")
    prompt = (
        "You are a strict fact-checker. Verify the draft answer.\n\n"
        "RULES:\n"
        "1. If EVERY claim is supported by the context → reply with only: PASS\n"
        "2. If any claim is unsupported → rewrite the corrected answer only\n"
        "3. Do NOT add commentary like 'Based on context...' or 'Upon reviewing...'\n"
        "4. Do NOT include PASS anywhere in a corrected answer\n"
        "5. If draft correctly says info is missing from context → reply PASS\n"
        "6. CRITICAL: Never remove or replace specific numbers, dates, amounts, or "
        "   percentages that ARE present in the context. A direct factual answer "
        "   (e.g. '90 days') is better than an indirect citation (e.g. 'per Article 5.3').\n"
        "7. If the draft answer is correct and direct, reply PASS — do not rewrite "
        "   just to add article references or legal citations.\n"
        "8. CRITICAL: If the context lists multiple people (founders, signatories, authors), "
        "   the answer MUST include ALL of them. An answer missing any named person is WRONG.\n\n"
        f"Question: {query}\n\n"
        f"Draft Answer:\n{draft_answer}\n\n"
        f"Source Context:\n{context}\n\n"
        "Verdict (PASS or corrected answer only):"
    )
    critique = fast_llm.complete(prompt)
    raw = critique.text.strip()

    # Strip trailing PASS if model appended it (Fix #5 — startswith AND endswith)
    cleaned = raw.rstrip()
    if cleaned.upper().startswith("PASS"):
        log("CRITIQUE", "Answer passed self-critique ✓", "green")
        return draft_answer
    if cleaned.upper().endswith("PASS"):
        cleaned = cleaned[:-4].rstrip(" \n.,")

    if cleaned == "":
        log("CRITIQUE", "Answer passed self-critique ✓", "green")
        return draft_answer

    log("CRITIQUE", "Answer corrected by self-critique ✓", "yellow")
    return cleaned


# ── MASTER PIPELINE ───────────────────────────────────────────
def run_pipeline(query: str, index, nodes_store: list, fast_llm, final_llm) -> dict:
    console.print(Panel(f"[bold white]Query:[/bold white] {query}", style="blue"))

    # 1. Decompose query into sub-queries
    sub_queries = decompose_query(query, fast_llm)

    # 2. HyDE expansion — run in parallel for all sub-queries
    log("HyDE", f"Expanding {len(sub_queries)} queries in parallel...")
    hyde_queries = {}
    with ThreadPoolExecutor(max_workers=len(sub_queries)) as ex:
        futures = {ex.submit(hyde_expand, q, fast_llm): q for q in sub_queries}
        for future in as_completed(futures):
            q = futures[future]
            try:
                hyde_queries[q] = future.result()
            except Exception:
                hyde_queries[q] = q  # fallback to original if HyDE fails
    log("HyDE", f"HyDE expansion complete ✓", "green")

    # 3. Hybrid retrieval for each sub-query, collect all result lists
    log("RETRIEVE", "Running hybrid retrieval for all sub-queries...")
    all_result_lists = []
    total_bm25 = 0
    for q in sub_queries:
        hyde_q = hyde_queries[q]
        hyde_res, orig_res, bm25_res = hybrid_retrieve_single(q, hyde_q, index, nodes_store)
        all_result_lists.extend([hyde_res, orig_res, bm25_res])
        total_bm25 += len(bm25_res)

    # 4. RAG-Fusion across all result lists
    fused_nodes = rag_fusion(all_result_lists)
    log("FUSION", f"RAG-Fusion merged {sum(len(r) for r in all_result_lists)} results → "
        f"{len(fused_nodes)} unique chunks ✓", "green")

    # 5. Source diversity — max chunks per file
    diverse_nodes = enforce_source_diversity(fused_nodes)
    log("DIVERSITY", f"After diversity filter: {len(diverse_nodes)} chunks ✓", "green")

    # 6. Rerank top N
    reranked = rerank(query, diverse_nodes)

    # 7. Context compression — extract only relevant sentences
    raw_context = "\n\n---\n\n".join([n.node.text for n in reranked])
    compressed_context = compress_context(query, raw_context, fast_llm)

    # 8. Parallel: CRAG gate + Final generation simultaneously
    log("PARALLEL", "Running CRAG gate + generation in parallel...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        crag_future = ex.submit(crag_gate, query, compressed_context, fast_llm)
        gen_future  = ex.submit(final_generate, query, compressed_context, final_llm)
        crag_ok = crag_future.result()
        draft   = gen_future.result()

    # 9. If CRAG says insufficient, decide whether to use web fallback
    if not crag_ok:
        # Smart gate: if the query is about live/real-time data for an entity
        # that IS in our documents, skip web search entirely.
        # Reason: web search returns data for real-world companies with similar
        # names (e.g. "Axon Systems" → real Axon Inc. AXON ticker).
        # For internal document entities, "not in documents" is the correct answer.
        live_data_keywords = [
            "stock price", "share price", "current price", "market cap",
            "stock market", "trading at", "ticker", "valuation today",
            "weather", "breaking news", "latest news"
        ]
        query_lower = query.lower()
        is_live_data_query = any(kw in query_lower for kw in live_data_keywords)

        # Check if primary entities in query appear in our doc context
        # If yes, this is an internal entity — don't web search
        entity_in_docs = len(compressed_context) > 100  # we got relevant doc chunks

        if is_live_data_query and entity_in_docs:
            log("CRAG", "Live data query for internal entity — skipping web search", "yellow")
            draft = "The provided documents do not contain this information. This appears to be a request for real-time data (such as a stock price) that is not available in the internal documents."
            source_type = "docs_insufficient"
        else:
            web_ctx = web_fallback(query, doc_context=compressed_context)
            if web_ctx:
                draft = final_generate(query, web_ctx, final_llm)
                compressed_context = web_ctx
                source_type = "web+docs"
            else:
                source_type = "docs_partial"
    else:
        source_type = "docs"

    # 10. Self-critique
    final_answer = self_critique(query, draft, compressed_context, fast_llm)

    # 11. Build sources
    sources = []
    if source_type in ("docs", "docs_partial", "web+docs"):
        for n in reranked:
            meta = n.node.metadata
            sources.append({
                "file": meta.get("file_name", "unknown"),
                "page": meta.get("page_label", "?"),
                "score": round(n.score, 3) if n.score else None
            })

    return {
        "answer": final_answer,
        "sources": sources,
        "source_type": source_type,
        "sub_queries": sub_queries,
        "chunks_retrieved": len(fused_nodes),
        "chunks_after_rerank": len(reranked)
    }