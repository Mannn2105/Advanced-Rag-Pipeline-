# End-to-End RAG Smoke Tests

This file defines a small, human-reviewable question set for checking the deployed demo against its four-document sample corpus.

## Dataset

`rag_smoke_questions.jsonl` contains ten questions. Each record includes:

- `test_id`: stable test identifier.
- `question`: question to submit through the chat UI or API.
- `expected_source_files`: files expected to contain supporting evidence.
- `expected_facts`: concepts or facts a reviewer should be able to verify in the retrieved passages and answer.

## Run after deployment

1. Confirm Railway reports the deployment as `SUCCESS`.
2. Check `/api/status` reports `ready: true` and a non-zero chunk count.
3. Sign in to the hosted demo and submit each question.
4. Inspect the returned source passages, not just the generated answer.
5. Mark a test as passing only when at least one expected source is retrieved and the answer is supported by that source.
6. For test `smoke-007`, verify the answer clearly says the current demo is single-login and does not isolate conversations by user.
7. Submit one out-of-corpus question (for example, "What is the weather in Tokyo?"). The answer must not invent information from the sample corpus; if web fallback is enabled, distinguish web-sourced information from document-sourced information.

## Result recording

Record the deployment commit, date, chunk count, question ID, retrieved source filenames, pass/fail, and a short reviewer note. Keep the results in a separate run report; do not overwrite this question set with observed results.

## Limitations

- This is a smoke-test set for the sample demo corpus, not a representative benchmark.
- Expected facts are review cues, not exact-match answer strings. Equivalent wording is acceptable when supported by the cited passage.
- A successful smoke test does not establish production security, multi-user isolation, factual accuracy across arbitrary documents, or performance at scale.
- Do not claim measured retrieval or answer-quality improvements from these ten questions.
