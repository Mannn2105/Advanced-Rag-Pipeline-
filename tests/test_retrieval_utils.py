import unittest

from retrieval_utils import normalize_hyde_query


class NormalizeHydeQueryTests(unittest.TestCase):
    def test_none_falls_back_to_original(self):
        self.assertEqual(normalize_hyde_query(None, "original query"), "original query")

    def test_blank_falls_back_to_original(self):
        self.assertEqual(normalize_hyde_query("  \n", "original query"), "original query")

    def test_non_string_falls_back_to_original(self):
        self.assertEqual(normalize_hyde_query([], "original query"), "original query")

    def test_valid_result_is_trimmed(self):
        self.assertEqual(normalize_hyde_query("  expanded query  ", "original query"), "expanded query")


if __name__ == "__main__":
    unittest.main()
