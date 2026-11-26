import unittest

from rerank_utils import local_cross_encoder_rerank


class FakeNode:
    def __init__(self, text):
        self.node = type("Node", (), {"text": text})()


class FakeEncoder:
    def predict(self, pairs, show_progress_bar=False):
        return [0.2, 0.9, 0.5]


class LocalRerankTests(unittest.TestCase):
    def test_returns_highest_scoring_nodes(self):
        nodes = [FakeNode("low"), FakeNode("high"), FakeNode("middle")]
        result = local_cross_encoder_rerank(nodes, "query", FakeEncoder(), 2)
        self.assertEqual([item.node.text for item in result], ["high", "middle"])

    def test_empty_nodes_returns_empty(self):
        self.assertEqual(local_cross_encoder_rerank([], "query", FakeEncoder(), 2), [])

    def test_non_positive_top_n_returns_empty(self):
        nodes = [FakeNode("one")]
        self.assertEqual(local_cross_encoder_rerank(nodes, "query", FakeEncoder(), 0), [])

    def test_mismatched_score_count_raises(self):
        class BadEncoder:
            def predict(self, pairs, show_progress_bar=False):
                return [0.5]

        nodes = [FakeNode("one"), FakeNode("two")]
        with self.assertRaises(ValueError):
            local_cross_encoder_rerank(nodes, "query", BadEncoder(), 2)


if __name__ == "__main__":
    unittest.main()
