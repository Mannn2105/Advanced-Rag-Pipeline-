"""Repeatable local regression checks for the RAG application.

Run `python tests.py` for deterministic unit checks. Pass `--live` to run
the question set against the configured LLMs and local index.
"""
import argparse
import json
import re
import unittest
from pathlib import Path

import config
from pipeline import (
    rag_fusion,
    enforce_source_diversity,
    _sentences,
)

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()

def auto_score(answer: str, test_case: dict) -> tuple[int, str]:
    candidate = normalize(answer)
    if not candidate:
        return 0, "No answer returned"
    failures = [word for word in test_case.get("keywords_fail", []) if normalize(word) in candidate]
    if failures:
        return 0, "Disallowed phrase(s): " + ", ".join(failures)
    expected = [word for word in test_case.get("keywords_pass", []) if normalize(word) in candidate]
    required = test_case.get("keywords_pass", [])
    if not required:
        return 3, "Manual review required"
    ratio = len(expected) / len(required)
    if ratio == 1:
        return 5, "All expected indicators found"
    if ratio >= 0.6:
        return 4, f"{len(expected)}/{len(required)} expected indicators found"
    if ratio > 0:
        return 2, f"{len(expected)}/{len(required)} expected indicators found"
    return 1, "No expected indicators found"

class _Node:
    def __init__(self, node_id, text, metadata=None):
        self.node_id = node_id
        self.text = text
        self.metadata = metadata or {}

class _Hit:
    def __init__(self, node, score=None):
        self.node = node
        self.score = score

class RetrievalUtilityTests(unittest.TestCase):
    def test_rank_fusion_deduplicates_and_rewards_consensus(self):
        shared = _Hit(_Node("shared", "A"))
        other = _Hit(_Node("other", "B"))
        result = rag_fusion([[shared, other], [shared]])
        self.assertEqual([item.node.node_id for item in result], ["shared", "other"])

    def test_diversity_limits_repeated_source(self):
        hits = [
            _Hit(_Node("a", "A", {"file_name": "one.pdf"})),
            _Hit(_Node("b", "B", {"file_name": "one.pdf"})),
            _Hit(_Node("c", "C", {"file_name": "two.pdf"})),
        ]
        selected = enforce_source_diversity(hits, max_per_file=1)
        self.assertEqual([item.node.node_id for item in selected], ["a", "c"])

    def test_sentence_extraction_ignores_empty_fragments(self):
        self.assertEqual(_sentences("First sentence.\n\nSecond sentence!"), ["First sentence.", "Second sentence!"])

    def test_config_has_valid_chunk_dimensions(self):
        self.assertEqual(config.validate_runtime(), [])

class AnswerScoringTests(unittest.TestCase):
    def test_score_requires_expected_indicators(self):
        score, _ = auto_score("Power use is 0.8 W.", {"keywords_pass": ["0.8"], "keywords_fail": []})
        self.assertEqual(score, 5)

    def test_score_rejects_disallowed_indicators(self):
        score, _ = auto_score("The answer is 12 TFLOPS.", {"keywords_pass": ["12"], "keywords_fail": ["tflops"]})
        self.assertEqual(score, 0)

SMOKE_CASES = [
    {"id": "R-01", "question": "What is the power consumption of the Cortex-X9 in Eco mode?", "keywords_pass": ["0.8"], "keywords_fail": []},
    {"id": "R-02", "question": "How many transistors does the Cortex-X9 integrate?", "keywords_pass": ["18 billion", "18billion"], "keywords_fail": []},
    {"id": "R-03", "question": "What is the NNT for remission in the Veranox trial?", "keywords_pass": ["4.5"], "keywords_fail": []},
    {"id": "R-04", "question": "What is the P1 response time SLA under Platinum Support?", "keywords_pass": ["30 minute", "30-minute"], "keywords_fail": ["4 hour"]},
]

def run_live() -> int:
    import pickle
    from pipeline import init_models, load_index, run_pipeline
    if not config.GROQ_API_KEY or not config.OPENROUTER_API_KEY:
        raise RuntimeError("Live tests require GROQ_API_KEY and OPENROUTER_API_KEY")
    _, fast_llm, answer_llm = init_models()
    index = load_index()
    with (Path(config.QDRANT_DIR) / "nodes.pkl").open("rb") as stream:
        nodes = pickle.load(stream)
    if not nodes:
        raise RuntimeError("The index catalogue is empty")
    failed = 0
    for case in SMOKE_CASES:
        result = run_pipeline(case["question"], index, nodes, fast_llm, answer_llm)
        score, reason = auto_score(result["answer"], case)
        print(f"{case['id']} | score={score}/5 | {reason} | {result['answer']}")
        failed += score < 4
    return 1 if failed else 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run model-backed smoke questions")
    args = parser.parse_args()
    if args.live:
        return run_live()
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())
