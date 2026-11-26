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


class GenerateWithFallbackTests(unittest.TestCase):
    def test_primary_success_does_not_call_fallback(self):
        from generation_utils import generate_with_fallback
        calls = []
        result = generate_with_fallback(
            lambda: calls.append("primary") or "answer",
            lambda: calls.append("fallback") or "backup",
        )
        self.assertEqual(result, "answer")
        self.assertEqual(calls, ["primary"])

    def test_primary_failure_uses_fallback(self):
        from generation_utils import generate_with_fallback
        def fail():
            raise RuntimeError("provider unauthorized")
        result = generate_with_fallback(fail, lambda: "Groq answer")
        self.assertEqual(result, "Groq answer")

    def test_known_stream_failure_skips_primary_retry(self):
        from generation_utils import generate_with_fallback
        calls = []
        result = generate_with_fallback(
            lambda: calls.append("primary") or "primary answer",
            lambda: calls.append("fallback") or "fallback answer",
            primary_error=RuntimeError("stream unauthorized"),
        )
        self.assertEqual(result, "fallback answer")
        self.assertEqual(calls, ["fallback"])

    def test_both_providers_fail_raises_clear_error(self):
        from generation_utils import generate_with_fallback
        def fail_primary():
            raise RuntimeError("primary down")
        def fail_fallback():
            raise RuntimeError("fallback down")
        with self.assertRaisesRegex(RuntimeError, "Groq fallback also failed"):
            generate_with_fallback(fail_primary, fail_fallback)
