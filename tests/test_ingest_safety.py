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
