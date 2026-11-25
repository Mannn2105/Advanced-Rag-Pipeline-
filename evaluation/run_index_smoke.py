"""Run the labelled smoke questions against an existing local Qdrant index.

This evaluates retrieval provenance only. It does not call answer-generation LLMs
or assess whether generated answers are factually correct.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import config
import pipeline
from evaluation.retrieval_metrics import evaluate_records, load_jsonl


def run_smoke(index_dir: Path, questions_path: Path, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be at least 1")

    nodes_path = index_dir / "nodes.pkl"
    if not nodes_path.is_file():
        raise FileNotFoundError(
            f"BM25 node snapshot not found at {nodes_path}. Run ingestion first."
        )

    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Retrieval needs embeddings, but this script deliberately does not initialize
    # Groq/OpenRouter clients because it does not generate answers.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL,
        device="cpu",
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )

    with nodes_path.open("rb") as stream:
        nodes_store = pickle.load(stream)
    if not isinstance(nodes_store, list) or not nodes_store:
        raise ValueError(f"{nodes_path} does not contain a non-empty node list")

    records = load_jsonl(questions_path)
    if not records:
        raise ValueError(f"No smoke questions found in {questions_path}")

    # Respect the explicit CLI path; never silently evaluate the app's configured
    # (potentially production-mounted) Qdrant directory instead.
    index = pipeline.load_index(qdrant_dir=index_dir)
    metric_records = []
    per_query = []
    try:
        for question in records:
            query = question["question"]
            ranked_lists = pipeline.hybrid_retrieve_single(
                query, query, index, nodes_store
            )
            fused = pipeline.rag_fusion(list(ranked_lists))
            selected = pipeline.enforce_source_diversity(fused)
            retrieved_sources = []
            for hit in selected:
                metadata = getattr(hit.node, "metadata", {}) or {}
                source = metadata.get("file_name") or metadata.get("source")
                if source and source not in retrieved_sources:
                    retrieved_sources.append(source)

            expected = list(dict.fromkeys(question["expected_source_files"]))
            metric_records.append({
                "query_id": question["test_id"],
                "relevant_doc_ids": expected,
                "retrieved_doc_ids": retrieved_sources,
            })
            found = [source for source in expected if source in retrieved_sources[:k]]
            per_query.append({
                "test_id": question["test_id"],
                "question": query,
                "expected_sources": expected,
                "retrieved_sources": retrieved_sources[:k],
                "expected_sources_found": found,
                "expected_sources_missing": [
                    source for source in expected if source not in found
                ],
            })
    finally:
        client = pipeline._qdrant_client
        if client is not None:
            client.close()
            pipeline._qdrant_client = None

    report = evaluate_records(metric_records, k=k)
    return {
        "index_directory": str(index_dir),
        "question_count": len(records),
        "k": k,
        "retrieval_metrics": report["macro_average"],
        "per_query": per_query,
        "scope": (
            "Hybrid vector/BM25 retrieval and source-file recall only; "
            "no answer generation or answer-quality evaluation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate labelled smoke questions against an existing Qdrant index."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(config.QDRANT_DIR),
        help="Existing Qdrant directory (default: configured QDRANT_DIR).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("rag_smoke_questions.jsonl"),
        help="Smoke-question JSONL file.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    report = run_smoke(args.index_dir, args.questions, k=args.k)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
