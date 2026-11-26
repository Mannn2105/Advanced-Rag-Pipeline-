import unittest
from unittest.mock import patch

from pipeline import rerank


class FakeNode:
    def __init__(self, text):
        self.node = type("Node", (), {"text": text})()


class FakeEncoder:
    def predict(self, pairs, show_progress_bar=False):
        return [0.2, 0.9, 0.5]


class RerankFallbackTests(unittest.TestCase):
    @patch("pipeline.config.COHERE_API_KEY", "")
    @patch("pipeline.config.RERANK_TOP_N", 2)
    @patch("pipeline._get_cross_encoder", return_value=FakeEncoder())
    def test_missing_cohere_key_uses_local_cross_encoder(self, _encoder):
        nodes = [FakeNode("low"), FakeNode("high"), FakeNode("middle")]
        result = rerank("query", nodes)
        self.assertEqual([item.node.text for item in result], ["high", "middle"])

    @patch("pipeline.config.COHERE_API_KEY", "configured-key")
    @patch("pipeline.config.RERANK_TOP_N", 2)
    @patch("pipeline.CohereRerank", side_effect=RuntimeError("provider unavailable"))
    @patch("pipeline._get_cross_encoder", return_value=FakeEncoder())
    def test_cohere_error_falls_back_locally(self, _encoder, _cohere):
        nodes = [FakeNode("low"), FakeNode("high"), FakeNode("middle")]
        result = rerank("query", nodes)
        self.assertEqual([item.node.text for item in result], ["high", "middle"])

    def test_empty_nodes_returns_empty(self):
        self.assertEqual(rerank("query", []), [])


if __name__ == "__main__":
    unittest.main()
