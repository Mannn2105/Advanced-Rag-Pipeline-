"""Tests for the real-index retrieval smoke evaluator's safety and metrics."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pipeline
from evaluation.run_index_smoke import run_smoke


class IndexSmokeTests(unittest.TestCase):
    def test_missing_bm25_snapshot_fails_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "nodes.pkl"):
                run_smoke(Path(temporary), Path("questions.jsonl"))

    def test_invalid_question_schema_fails_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_dir = root / "index"
            index_dir.mkdir()
            with (index_dir / "nodes.pkl").open("wb") as stream:
                pickle.dump([{"placeholder": True}], stream)
            questions = root / "questions.jsonl"
            questions.write_text(
                json.dumps({"test_id": "q1", "question": "A question"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected_source_files"):
                run_smoke(index_dir, questions)


    def test_does_not_close_qdrant_client_owned_by_caller(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_dir = root / "index"
            index_dir.mkdir()
            with (index_dir / "nodes.pkl").open("wb") as stream:
                pickle.dump([{"placeholder": True}], stream)
            questions = root / "questions.jsonl"
            questions.write_text(
                json.dumps({
                    "test_id": "q1",
                    "question": "A question",
                    "expected_source_files": ["PROJECT_GUIDE.md"],
                }) + "\\n",
                encoding="utf-8",
            )

            core = ModuleType("llama_index.core")
            core.Settings = SimpleNamespace(embed_model=None)
            embeddings = ModuleType("llama_index.embeddings")
            huggingface = ModuleType("llama_index.embeddings.huggingface")
            huggingface.HuggingFaceEmbedding = lambda **kwargs: object()
            fake_llama = ModuleType("llama_index")
            fake_llama.__path__ = []
            embeddings.__path__ = []
            existing_client = SimpleNamespace(close=unittest.mock.Mock())

            with (
                patch.dict("sys.modules", {
                    "llama_index": fake_llama,
                    "llama_index.core": core,
                    "llama_index.embeddings": embeddings,
                    "llama_index.embeddings.huggingface": huggingface,
                }),
                patch.object(pipeline, "_qdrant_client", existing_client),
                patch.object(pipeline, "_qdrant_dir", str(root / "somewhere-else")),
                patch.object(
                    pipeline,
                    "load_index",
                    side_effect=RuntimeError("Different Qdrant directory already open"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Different Qdrant directory"):
                    run_smoke(index_dir, questions)

            existing_client.close.assert_not_called()
            self.assertIs(pipeline._qdrant_client, existing_client)

    def test_smoke_report_measures_retrieved_source_recall_without_answer_llms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_dir = root / "index"
            index_dir.mkdir()
            with (index_dir / "nodes.pkl").open("wb") as stream:
                pickle.dump([{"placeholder": True}], stream)
            questions = root / "questions.jsonl"
            questions.write_text(
                json.dumps({
                    "test_id": "q1",
                    "question": "Which framework serves the app?",
                    "expected_source_files": ["PROJECT_GUIDE.md"],
                    "expected_facts": ["FastAPI"],
                }) + "\n",
                encoding="utf-8",
            )

            core = ModuleType("llama_index.core")
            core.Settings = SimpleNamespace(embed_model=None)
            embeddings = ModuleType("llama_index.embeddings")
            huggingface = ModuleType("llama_index.embeddings.huggingface")
            huggingface.HuggingFaceEmbedding = lambda **kwargs: object()
            fake_llama = ModuleType("llama_index")
            fake_llama.__path__ = []
            embeddings.__path__ = []

            node = SimpleNamespace(
                node_id="node-1",
                metadata={"file_name": "PROJECT_GUIDE.md", "page_label": "1"},
                text="FastAPI serves the chat application.",
            )
            hit = SimpleNamespace(node=node, score=0.9)
            client = SimpleNamespace(close=lambda: None)

            with (
                patch.dict("sys.modules", {
                    "llama_index": fake_llama,
                    "llama_index.core": core,
                    "llama_index.embeddings": embeddings,
                    "llama_index.embeddings.huggingface": huggingface,
                }),
                patch.object(pipeline, "load_index", return_value=object()) as load_index,
                patch.object(
                    pipeline,
                    "hybrid_retrieve_single",
                    return_value=([hit], [], []),
                ),
                patch.object(pipeline, "_qdrant_client", client),
            ):
                report = run_smoke(index_dir, questions, k=5)

            load_index.assert_called_once_with(qdrant_dir=index_dir)
            self.assertEqual(report["question_count"], 1)
            self.assertEqual(report["retrieval_metrics"]["recall@5"], 1.0)
            self.assertEqual(
                report["per_query"][0]["expected_sources_missing"], []
            )
            self.assertIn("no answer generation", report["scope"].lower())


if __name__ == "__main__":
    unittest.main()
