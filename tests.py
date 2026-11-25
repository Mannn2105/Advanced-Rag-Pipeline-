"""Repeatable local regression checks for the RAG application.

Run `python tests.py` for deterministic unit checks. Pass `--live` to run
the question set against the configured LLMs and local index.
"""
import argparse
import os
import json
import re
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import config
from pipeline import (
    rag_fusion,
    enforce_source_diversity,
    _sentences,
)
from ingest import promote_staged_index
from scripts.migrate_qdrant_volume import migrate_volume

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

    def test_legacy_skip_flag_disables_contextual_enrichment(self):
        self.assertFalse(config.contextual_enrichment_enabled({
            "RAG_SKIP_CONTEXTUAL_ENRICHMENT": "true"
        }))

    def test_new_enrichment_flag_overrides_legacy_skip_flag(self):
        self.assertTrue(config.contextual_enrichment_enabled({
            "RAG_SKIP_CONTEXTUAL_ENRICHMENT": "true",
            "RAG_ENABLE_CONTEXTUAL_ENRICHMENT": "yes",
        }))

    def test_contextual_enrichment_defaults_to_disabled(self):
        self.assertFalse(config.contextual_enrichment_enabled({}))

    def test_simple_chunking_defaults_to_enabled(self):
        self.assertTrue(config._env_bool(None, True))

    def test_manifest_accepts_ingestion_metadata(self):
        manifest = {
            "model": config.EMBED_MODEL,
            "dimension": config.EMBED_DIMENSION,
            "collection": config.COLLECTION_NAME,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "node_count": 29,
        }
        self.assertTrue(config.manifest_matches(manifest))

    def test_manifest_rejects_incompatible_embedding_setup(self):
        manifest = {
            "model": config.EMBED_MODEL,
            "dimension": config.EMBED_DIMENSION + 1,
            "collection": config.COLLECTION_NAME,
            "node_count": 29,
        }
        self.assertFalse(config.manifest_matches(manifest))

class IngestionPromotionTests(unittest.TestCase):
    def test_promotion_keeps_previous_index_as_rollback_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "qdrant_db"
            staging = root / "qdrant_db.staging"
            active.mkdir()
            staging.mkdir()
            (active / "index.txt").write_text("old index", encoding="utf-8")
            (staging / "index.txt").write_text("new index", encoding="utf-8")

            backup = promote_staged_index(staging, active)

            self.assertIsNotNone(backup)
            self.assertEqual((active / "index.txt").read_text(encoding="utf-8"), "new index")
            self.assertTrue(backup.is_dir())
            self.assertEqual((backup / "index.txt").read_text(encoding="utf-8"), "old index")
            self.assertFalse(staging.exists())

    def test_promotion_rejects_different_filesystems_without_touching_active_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "qdrant_db"
            staging = root / "qdrant_db.staging"
            active.mkdir()
            staging.mkdir()
            (active / "index.txt").write_text("known-good index", encoding="utf-8")
            (staging / "index.txt").write_text("candidate index", encoding="utf-8")
            real_stat = os.stat

            def different_devices(path, *args, **kwargs):
                result = real_stat(path, *args, **kwargs)
                resolved = Path(path)
                values = list(result)
                if resolved == staging:
                    values[2] = result.st_dev + 1
                    return os.stat_result(values)
                return result

            with mock.patch("ingest.os.stat", side_effect=different_devices):
                with self.assertRaisesRegex(OSError, "different filesystems"):
                    promote_staged_index(staging, active)

            self.assertEqual((active / "index.txt").read_text(encoding="utf-8"), "known-good index")
            self.assertEqual((staging / "index.txt").read_text(encoding="utf-8"), "candidate index")

    def test_promotion_rejects_missing_staging_without_touching_active_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "qdrant_db"
            active.mkdir()
            (active / "index.txt").write_text("known-good index", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                promote_staged_index(root / "missing-staging", active)

            self.assertEqual((active / "index.txt").read_text(encoding="utf-8"), "known-good index")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["qdrant_db"])

    def test_failed_promotion_restores_previous_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "qdrant_db"
            staging = root / "qdrant_db.staging"
            active.mkdir()
            staging.mkdir()
            (active / "index.txt").write_text("known-good index", encoding="utf-8")
            (staging / "index.txt").write_text("candidate index", encoding="utf-8")
            real_replace = os.replace

            with mock.patch("ingest.os.replace", side_effect=[
                real_replace,
                OSError("simulated activation failure"),
                real_replace,
            ]):
                with self.assertRaisesRegex(OSError, "simulated activation failure"):
                    promote_staged_index(staging, active)

            self.assertEqual((active / "index.txt").read_text(encoding="utf-8"), "known-good index")
            self.assertTrue(staging.is_dir())


class VolumeMigrationTests(unittest.TestCase):
    def _volume(self, root):
        (root / "nodes.pkl").write_bytes(b"node catalogue")
        (root / "embedding_config.json").write_text(
            json.dumps({"model": config.EMBED_MODEL, "dimension": config.EMBED_DIMENSION}),
            encoding="utf-8",
        )
        (root / "collection_data").mkdir()
        (root / "collection_data" / "segment.bin").write_bytes(b"qdrant data")

    def test_dry_run_does_not_move_volume_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._volume(root)
            planned = migrate_volume(root)
            self.assertEqual({path.name for path in planned}, {
                "nodes.pkl", "embedding_config.json", "collection_data"
            })
            self.assertTrue((root / "nodes.pkl").is_file())
            self.assertFalse((root / "active").exists())

    def test_migration_requires_explicit_stopped_service_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._volume(root)
            with self.assertRaisesRegex(RuntimeError, "Stop the application"):
                migrate_volume(root, apply=True)
            self.assertTrue((root / "nodes.pkl").is_file())

    def test_confirmed_migration_moves_all_index_files_into_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._volume(root)
            moved = migrate_volume(root, apply=True, service_stopped=True)
            active = root / "active"
            self.assertEqual(len(moved), 3)
            self.assertTrue((active / "nodes.pkl").is_file())
            self.assertTrue((active / "embedding_config.json").is_file())
            self.assertTrue((active / "collection_data" / "segment.bin").is_file())
            self.assertEqual(list(root.iterdir()), [active])


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
