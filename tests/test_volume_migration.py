"""Safety tests for the persistent Qdrant volume migration helper."""
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_qdrant_volume import migrate_volume, plan_migration


class VolumeMigrationTests(unittest.TestCase):
    def make_volume(self, root: Path) -> Path:
        root.mkdir()
        (root / "nodes.pkl").write_bytes(b"snapshot")
        (root / "embedding_config.json").write_text("{}", encoding="utf-8")
        (root / "chat_history.json").write_text("{}", encoding="utf-8")
        return root

    def test_dry_run_lists_files_without_moving_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_volume(Path(temporary) / "volume")
            planned = migrate_volume(root)
            self.assertEqual({path.name for path in planned}, {
                "nodes.pkl", "embedding_config.json", "chat_history.json"
            })
            self.assertTrue((root / "nodes.pkl").is_file())
            self.assertFalse((root / "active").exists())

    def test_apply_requires_explicit_stopped_service_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_volume(Path(temporary) / "volume")
            with self.assertRaisesRegex(RuntimeError, "Refusing to move a live"):
                migrate_volume(root, apply=True)
            self.assertTrue((root / "nodes.pkl").is_file())

    def test_apply_moves_volume_contents_into_child_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_volume(Path(temporary) / "volume")
            moved = migrate_volume(root, apply=True, service_stopped=True)
            active = root / "active"
            self.assertEqual(len(moved), 3)
            self.assertTrue((active / "nodes.pkl").is_file())
            self.assertTrue((active / "embedding_config.json").is_file())
            self.assertTrue((active / "chat_history.json").is_file())
            self.assertFalse((root / "nodes.pkl").exists())

    def test_rejects_target_names_that_escape_volume_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_volume(Path(temporary) / "volume")
            for name in ("../outside", "nested/active", "/tmp/outside", ".", "..", "nested\\active"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "single child-directory"):
                        plan_migration(root, name)

    def test_rejects_symlink_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self.make_volume(base / "volume")
            outside = base / "outside"
            outside.mkdir()
            (root / "active").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                plan_migration(root)


if __name__ == "__main__":
    unittest.main()
