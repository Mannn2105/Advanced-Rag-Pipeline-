"""Unit tests for the dependency-free retrieval metric evaluator."""
import unittest

from retrieval_metrics import (
    evaluate_records,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_precision_recall_and_rr(self):
        relevant = {"a", "c"}
        ranked = ["b", "a", "c"]
        self.assertAlmostEqual(precision_at_k(relevant, ranked, 3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(relevant, ranked, 2), 1 / 2)
        self.assertAlmostEqual(reciprocal_rank(relevant, ranked), 1 / 2)

    def test_ndcg_perfect_ranking(self):
        self.assertAlmostEqual(ndcg_at_k({"a", "b"}, ["a", "b"], 2), 1.0)

    def test_ndcg_penalizes_lower_relevant_rank(self):
        perfect = ndcg_at_k({"a", "b"}, ["a", "b", "x"], 3)
        worse = ndcg_at_k({"a", "b"}, ["x", "a", "b"], 3)
        self.assertGreater(perfect, worse)

    def test_duplicate_retrieved_ids_count_once(self):
        report = evaluate_records([{
            "query_id": "q1",
            "relevant_doc_ids": ["a", "b"],
            "retrieved_doc_ids": ["a", "a", "b"],
        }], k=3)
        self.assertAlmostEqual(report["macro_average"]["recall@3"], 1.0)
        self.assertAlmostEqual(report["macro_average"]["precision@3"], 2 / 3)

    def test_metric_helpers_also_ignore_duplicate_ranked_documents(self):
        relevant = {"a"}
        ranked = ["a", "a", "b"]
        self.assertAlmostEqual(precision_at_k(relevant, ranked, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(relevant, ranked, 3), 1.0)
        self.assertAlmostEqual(reciprocal_rank(relevant, ranked), 1.0)
        self.assertAlmostEqual(ndcg_at_k(relevant, ranked, 3), 1.0)

    def test_empty_relevant_set_returns_zero(self):
        report = evaluate_records([{
            "query_id": "q1",
            "relevant_doc_ids": [],
            "retrieved_doc_ids": ["a"],
        }])
        self.assertEqual(report["macro_average"]["recall@5"], 0.0)
        self.assertEqual(report["macro_average"]["mrr"], 0.0)

    def test_invalid_k_and_duplicate_query_are_rejected(self):
        with self.assertRaises(ValueError):
            precision_at_k({"a"}, ["a"], 0)
        duplicate = [
            {"query_id": "q1", "relevant_doc_ids": ["a"], "retrieved_doc_ids": ["a"]},
            {"query_id": "q1", "relevant_doc_ids": ["a"], "retrieved_doc_ids": ["a"]},
        ]
        with self.assertRaises(ValueError):
            evaluate_records(duplicate)


if __name__ == "__main__":
    unittest.main()
