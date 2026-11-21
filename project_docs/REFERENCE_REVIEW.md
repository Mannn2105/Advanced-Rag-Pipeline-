# Reference Repository Review and Improvement Plan

## Scope

This review compares the public repository [Hitanshu-Varia/Orchestrated_RAG](https://github.com/Hitanshu-Varia/Orchestrated_RAG) with this project. It is a technical review, not a claim about authorship or intent.

## Findings

- The reference repository was created on 2026-03-14. This repository was created on 2026-10-11.
- At the time of review, neither repository exposed a `LICENSE` file in its root tree. A README badge is not a substitute for an actual license.
- The following files were byte-for-byte identical in the checked default-branch snapshots:
  - `config.py`
  - `pipeline.py`
  - `query.py`
  - `ingest.py`
  - `tests.py`
- The reference repository also contains `.agent/.skills/SKILL.md`, a frontend-building instruction for an AI coding agent. It requests a single-file HTML/CSS/JS mock RAG interface. It is not a ChatGPT system prompt and does not establish a ChatGPT integration.
- The reference project describes Groq, OpenRouter, LlamaIndex's OpenAI-compatible adapter, Cohere, Qdrant, BM25, and Tavily. The OpenAI-compatible adapter alone does not mean the OpenAI API or ChatGPT is being used.

## Responsible reuse

Because no explicit license was found in the reference repository, do not copy, redistribute, or publish its source code as reusable open-source code based only on its public availability. Seek written permission from the rightsholder or replace the overlapping implementation with independently written code. Preserve clear attribution for any reuse that is authorized.

The identical snapshots do not, by themselves, establish who authored the code first or how it entered either repository. This note deliberately makes no accusation about either project.

## Improvement plan for this project

1. **Provenance first:** establish which files are independently authored, which are authorized third-party material, and which need replacement.
2. **Correct licensing:** choose a license only for code the project owner has the right to license. Until then, make no MIT claim.
3. **Reliability:** add isolated tests for parsing, retrieval fusion, source diversity, context compression, and API validation.
4. **Retrieval quality:** measure Recall@k and MRR against a fixed, labelled corpus; compare vector-only, BM25-only, hybrid, reranked, and full-pipeline variants.
5. **Grounded answers:** require source identifiers for factual claims, test unsupported-question refusal, and separate web-derived evidence from indexed-document evidence.
6. **Security:** validate uploads and file types, enforce request size limits, treat retrieved content as untrusted, and restrict CORS/authentication before public use.
7. **Operations:** add structured logs, stage-level latency, provider error handling, health/readiness checks, and safe ingestion lifecycle management.

## Acceptance criteria

- No unlicensed third-party implementation is copied into the public project.
- The repository's license statement matches an actual license file and the project's provenance.
- Tests run in a documented environment and report real results.
- Evaluation results include the corpus, question set, model versions, configuration, commit SHA, and measured metrics.
- A deployed service is called ready only after its index and required providers are confirmed usable.
