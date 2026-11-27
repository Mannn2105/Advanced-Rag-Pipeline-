"""Tests for query routing decisions in the hosted RAG pipeline."""
import unittest

from retrieval_utils import is_simple_query


class SimpleQueryRoutingTests(unittest.TestCase):
    def test_short_factual_question_uses_fast_path(self):
        self.assertTrue(is_simple_query("What is the purpose of this project?"))
        self.assertTrue(is_simple_query("How many documents are indexed?"))

    def test_multi_part_questions_keep_full_pipeline(self):
        self.assertFalse(is_simple_query("Compare vector search and BM25 retrieval."))
        self.assertFalse(is_simple_query("Explain how retrieval and reranking work together."))
        self.assertFalse(is_simple_query("What is the difference between the two approaches?"))

    def test_long_questions_keep_full_pipeline(self):
        question = "Explain the complete architecture and describe every stage in detail, including how the components interact."
        self.assertFalse(is_simple_query(question))

    def test_non_string_input_is_not_simple(self):
        self.assertFalse(is_simple_query(None))


if __name__ == "__main__":
    unittest.main()
