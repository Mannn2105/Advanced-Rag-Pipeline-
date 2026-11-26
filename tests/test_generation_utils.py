import unittest

from generation_utils import normalize_generated_answer


class NormalizeGeneratedAnswerTests(unittest.TestCase):
    def test_valid_answer_is_trimmed(self):
        self.assertEqual(normalize_generated_answer("  A valid answer.  "), "A valid answer.")

    def test_blank_answer_is_empty(self):
        self.assertEqual(normalize_generated_answer("  \n"), "")

    def test_none_is_empty(self):
        self.assertEqual(normalize_generated_answer(None), "")

    def test_non_string_is_empty(self):
        self.assertEqual(normalize_generated_answer({"answer": "text"}), "")


if __name__ == "__main__":
    unittest.main()
