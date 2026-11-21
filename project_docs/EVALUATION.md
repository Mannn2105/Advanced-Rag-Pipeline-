# Evaluation Guide

## Goal

Measure whether each retrieval stage improves evidence retrieval and answer quality on a fixed question set. Do not rely on a single anecdotal query or a project-reported percentage.

## Independent retrieval metrics

The `evaluation/` directory contains a dependency-free evaluator for labelled retrieval rankings. It reports Precision@k, Recall@k, MRR, and nDCG@k, with per-query scores and macro averages. Its sample dataset is synthetic and is not a benchmark of this application. See [`evaluation/README.md`](../evaluation/README.md) for the data format and commands. A GitHub Actions workflow runs its unit tests and sample smoke test on relevant pushes and pull requests.

## Existing evaluation runner

The repository includes `tests.py`, a predefined question suite with expected keywords and grouped scenarios. It runs against the configured local index and model providers, so it is an integration-style evaluation rather than a fully isolated unit-test suite.

Before running it:

1. Install dependencies and configure the required provider keys.
2. Add the same evaluation documents to the local `docs/` directory.
3. Run `python ingest.py` to build the index.
4. Run `python tests.py` and preserve the generated report under the ignored local test-results directory.

The runner's keyword checks are useful regression signals, but they do not fully measure factual correctness, citation quality, or retrieval relevance. Review a sample of outputs manually.

## Recommended retrieval ablation

Run the same fixed question set against these configurations:

| Run | Retrieval configuration | Question set / corpus |
|---|---|---|
| A | Vector retrieval only | Keep fixed |
| B | BM25 only | Keep fixed |
| C | Hybrid retrieval + fusion | Keep fixed |
| D | Hybrid + fusion + reranking | Keep fixed |
| E | Full pipeline, including enrichment and compression | Keep fixed |

Change one major factor at a time where practical. Keep the corpus, questions, model versions, top-k values, and hardware consistent. Record any provider/model changes between runs.

## Metrics to record

- **Recall@k:** proportion of questions for which a relevant evidence chunk appears in the top k retrieved chunks.
- **MRR:** average reciprocal rank of the first relevant chunk.
- **Answer correctness:** human-judged correctness against the source documents.
- **Groundedness:** whether material answer claims are supported by retrieved sources.
- **Citation/source quality:** whether cited sources actually support the associated claims.
- **Latency:** end-to-end time, ideally with median and p95 over repeated runs.
- **Cost:** estimated model and external API cost per question.

Define relevance judgments before computing Recall@k or MRR. For answer quality, use a documented rubric and manually review disagreements.

## Results template

Copy this table into a run log and fill it only with measured values:

| Run | Recall@5 | MRR | Correct answers (%) | Grounded answers (%) | Median latency (s) | p95 latency (s) | Cost / query |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — Vector only | — | — | — | — | — | — | — |
| B — BM25 only | — | — | — | — | — | — | — |
| C — Hybrid + fusion | — | — | — | — | — | — | — |
| D — + reranking | — | — | — | — | — | — | — |
| E — Full pipeline | — | — | — | — | — | — | — |

Also record the date, commit SHA, corpus version, question-set version, hardware, embedding model, LLM model/provider, and configuration values for every experiment.

## Reporting results responsibly

The 35–49% contextual-retrieval improvement mentioned in the original project notes refers to an external research-reported result. It must not be presented as a measured result of this implementation without a reproducible experiment. Likewise, latency figures should include the hardware, corpus size, model/provider, network conditions, and measurement method.
