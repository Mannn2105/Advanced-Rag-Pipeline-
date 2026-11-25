"""Safety tests for staged document-index promotion."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ingest


class BuildIndexGuardTests(unittest.TestCase):
    def test_empty_nodes_are_rejected_before_optional_dependencies_load(self):
        with patch.dict("sys.modules", {
            "llama_index": None,
            "llama_index.core": None,
        }):
            with self.assertRaisesRegex(ValueError, "No chunks were produced"):
                ingest.build_index([])


class StagedPromotionTests(unittest.TestCase):
    def test_promotes_staged_directory_and_keeps_previous_index_as_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "qdrant_db"
            staging = root / "qdrant_db.staging-test"
            active.mkdir()
            staging.mkdir()
            (active / "old-index.txt").write_text("old", encoding="utf-8")
            (staging / "new-index.txt").write_text("new", encoding="utf-8")

            backup = ingest.promote_staged_index(staging, active)

            self.assertIsNotNone(backup)
            self.assertTrue((active / "new-index.txt").is_file())
            self.assertFalse((active / "old-index.txt").exists())
            self.assertTrue((backup / "old-index.txt").is_file())
            self.assertFalse(staging.exists())

    def test_restores_previous_index_when_staged_rename_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "qdrant_db"
            staging = root / "qdrant_db.staging-test"
            active.mkdir()
            staging.mkdir()
            (active / "old-index.txt").write_text("old", encoding="utf-8")
            (staging / "new-index.txt").write_text("new", encoding="utf-8")

            real_replace = __import__("os").replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated promotion failure")
                return real_replace(source, destination)

            with patch("ingest.os.replace", side_effect=fail_second_replace):
                with self.assertRaisesRegex(OSError, "simulated promotion failure"):
                    ingest.promote_staged_index(staging, active)

            self.assertTrue((active / "old-index.txt").is_file())
            self.assertFalse((active / "new-index.txt").exists())
            self.assertTrue(staging.exists())

    def test_rejects_staging_directory_on_different_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active" / "qdrant_db"
            staging = root / "staging" / "qdrant_db"
            active.parent.mkdir()
            staging.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "share a parent filesystem"):
                ingest.promote_staged_index(staging, active)


if __name__ == "__main__":
    unittest.main()

class IngestionFlowTests(unittest.TestCase):
    def test_demo_corpus_ingests_all_four_markdown_sources_and_preserves_runtime_state(self):
        from types import SimpleNamespace
        import config

        repository_root = Path(__file__).resolve().parents[1]
        demo_docs = repository_root / "docs"
        source_files = sorted(demo_docs.glob("*.md"))
        self.assertEqual(len(source_files), 4, "The demo corpus should contain four Markdown sources")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "qdrant_db"
            active.mkdir()
            (active / "chat_history.json").write_text('{"conversation": []}', encoding="utf-8")
            (active / "enrichment_cache.json").write_text("{}", encoding="utf-8")
            (active / "old-index.marker").write_text("previous", encoding="utf-8")

            def extract(path):
                return [SimpleNamespace(text=path.read_text(encoding="utf-8"), metadata={})]

            def chunk(documents):
                return [
                    SimpleNamespace(
                        text=document.text[:200],
                        metadata=dict(document.metadata),
                    )
                    for document in documents
                ]

            def build(nodes, qdrant_dir):
                Path(qdrant_dir, "new-index.marker").write_text(
                    str(len(nodes)), encoding="utf-8"
                )

            with (
                patch.object(config, "DOCS_DIR", str(demo_docs)),
                patch.object(config, "QDRANT_DIR", str(active)),
                patch.object(config, "RAG_ENABLE_CONTEXTUAL_ENRICHMENT", False),
                patch.object(config, "COLLECTION_NAME", "test_collection"),
                patch.object(config, "validate_runtime", return_value=[]),
                patch.object(ingest, "_extract_document", side_effect=extract),
                patch.object(ingest, "_chunk_documents", side_effect=chunk),
                patch.object(ingest, "build_index", side_effect=build),
            ):
                result = ingest.ingest()

            self.assertEqual(result["files"], 4)
            self.assertEqual(result["documents"], 4)
            self.assertEqual(result["chunks"], 4)
            self.assertEqual(result["warnings"], [])
            self.assertTrue((active / "new-index.marker").is_file())
            self.assertEqual((active / "new-index.marker").read_text(), "4")
            self.assertEqual(
                (active / "chat_history.json").read_text(encoding="utf-8"),
                '{"conversation": []}',
            )
            self.assertEqual(
                (active / "enrichment_cache.json").read_text(encoding="utf-8"), "{}"
            )
            self.assertIsNotNone(result["backup_directory"])
            backup = Path(result["backup_directory"])
            self.assertTrue((backup / "old-index.marker").is_file())

    def test_smoke_questions_reference_only_existing_demo_sources(self):
        import json

        repository_root = Path(__file__).resolve().parents[1]
        docs_dir = repository_root / "docs"
        questions_path = repository_root / "evaluation" / "rag_smoke_questions.jsonl"
        available_sources = {path.name for path in docs_dir.glob("*.md")}
        records = [
            json.loads(line)
            for line in questions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(records), 10)
        self.assertEqual(len({record["test_id"] for record in records}), 10)
        for record in records:
            self.assertTrue(record["expected_source_files"])
            missing = set(record["expected_source_files"]) - available_sources
            self.assertFalse(
                missing,
                f"{record['test_id']} references missing demo sources: {sorted(missing)}",
            )

