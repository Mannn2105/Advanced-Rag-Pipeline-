"""Dependency-free retrieval metrics for labelled RAG evaluation sets.

Input JSONL records have this shape:
{"query_id":"q1","relevant_doc_ids":["doc-a"],"retrieved_doc_ids":["doc-a","doc-b"]}

This utility evaluates retrieval rankings only; it does not judge answer quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Document IDs must be non-empty strings.")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Fraction of the first k unique ranked results that are relevant."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    ranked = _unique_in_order(ranked)
    return sum(doc_id in relevant for doc_id in ranked[:k]) / k


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Fraction of all known relevant documents found in the first k."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not relevant:
        return 0.0
    ranked = _unique_in_order(ranked)
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    """Reciprocal rank of the first relevant unique result (zero if none is found)."""
    ranked = _unique_in_order(ranked)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Binary-relevance normalized discounted cumulative gain at k."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    ranked = _unique_in_order(ranked)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / ideal_dcg


def evaluate_records(records: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Compute macro-averaged retrieval metrics over query records."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not records:
        raise ValueError("At least one evaluation record is required.")

    query_ids: set[str] = set()
    per_query: list[dict[str, Any]] = []
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {row_number} must be a JSON object.")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Record {row_number} needs a non-empty query_id.")
        if query_id in query_ids:
            raise ValueError(f"Duplicate query_id: {query_id}")
        query_ids.add(query_id)

        relevant_values = record.get("relevant_doc_ids")
        retrieved_values = record.get("retrieved_doc_ids")
        if not isinstance(relevant_values, list) or not isinstance(retrieved_values, list):
            raise ValueError(
                f"Record {row_number} must contain relevant_doc_ids and retrieved_doc_ids lists."
            )
        relevant = set(_unique_in_order(relevant_values))
        ranked = _unique_in_order(retrieved_values)
        metrics = {
            "query_id": query_id,
            f"precision@{k}": precision_at_k(relevant, ranked, k),
            f"recall@{k}": recall_at_k(relevant, ranked, k),
            "mrr": reciprocal_rank(relevant, ranked),
            f"ndcg@{k}": ndcg_at_k(relevant, ranked, k),
        }
        per_query.append(metrics)

    metric_names = [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]
    averages = {
        name: sum(row[name] for row in per_query) / len(per_query)
        for name in metric_names
    }
    return {
        "query_count": len(per_query),
        "k": k,
        "macro_average": averages,
        "per_query": per_query,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate labelled RAG retrieval rankings.")
    parser.add_argument("dataset", type=Path, help="Path to a JSONL evaluation file.")
    parser.add_argument("--k", type=int, default=5, help="Ranking cutoff (default: 5).")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()

    try:
        report = evaluate_records(load_jsonl(args.dataset), k=args.k)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
