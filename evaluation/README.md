# Retrieval Evaluation

This directory contains an independently written, dependency-free evaluator for labelled retrieval rankings. It is intentionally separate from the upstream-derived RAG pipeline so the evaluation utility can be reviewed and tested on its own.

## What it measures

For each query, the evaluator calculates:

- **Precision@k:** relevant retrieved documents in the first k positions divided by k.
- **Recall@k:** known relevant documents found in the first k positions divided by all labelled relevant documents.
- **MRR:** reciprocal rank of the first relevant result.
- **nDCG@k:** ranking quality with logarithmic position discounts and binary relevance.

The report includes per-query scores and macro averages across queries. Repeated document IDs in a ranking are de-duplicated, preserving their first occurrence.

These are retrieval metrics only. They do not measure answer correctness, factual grounding, citation quality, latency, or cost.

## Dataset format

Use one JSON object per line:

```json
{"query_id":"q-001","question":"Where are embeddings stored?","relevant_doc_ids":["doc-7"],"retrieved_doc_ids":["doc-7","doc-2","doc-9"]}
```

- `query_id` must be unique.
- `relevant_doc_ids` is the human-labelled set of documents that contain evidence needed to answer the question.
- `retrieved_doc_ids` is the ordered list returned by the retrieval configuration being evaluated.
- Keep the same question set, corpus snapshot, and relevance labels when comparing configurations.

`sample_dataset.jsonl` is synthetic and exists only to demonstrate the format. It is not a benchmark of this application.

## Run it

From the repository root:

```bash
python evaluation/retrieval_metrics.py evaluation/sample_dataset.jsonl --k 5
python evaluation/retrieval_metrics.py evaluation/sample_dataset.jsonl --k 5 --output evaluation/report.json
python -m unittest discover -s evaluation -p "test_*.py" -v
```

No external packages or API keys are required for this evaluator.

## Recommended experiment protocol

1. Freeze and identify the document corpus; record a corpus version or checksum.
2. Create a representative question set and have a human label relevant document IDs.
3. Run each retrieval configuration against exactly the same questions and corpus.
4. Save each ordered result list as JSONL, preserving query IDs.
5. Compare vector-only, BM25-only, hybrid fusion, reranked, and full-pipeline results.
6. Record the Git commit, configuration, embedding model, reranker, question-set version, and corpus version with every run.
7. Repeat runs when any stochastic or externally hosted component can affect rankings.

Do not tune and report on the same small question set as if it were an unbiased test set. Keep a held-out set for final comparison.

## Interpretation cautions

- A query with no labelled relevant documents receives zero for recall, MRR, and nDCG; avoid including such queries unless they are intentionally part of the evaluation design.
- Recall depends on the completeness of human relevance labels.
- Macro averages weight each query equally.
- Retrieval scores do not establish that generated answers are correct or grounded.
- Never invent benchmark results. Publish measured reports together with their dataset and configuration provenance.

## End-to-end demo checks

For a separate set of ten questions grounded in the repository's four demo documents, see [`rag_smoke_questions.jsonl`](rag_smoke_questions.jsonl) and the [smoke-test runbook](SMOKE_TESTS.md). These checks validate the hosted ingestion, retrieval, and answer flow manually; they are not retrieval-metric benchmark data and should not be mixed with `sample_dataset.jsonl`.
