"""Tests for bounded and well-formed RAG query requests."""
import unittest
from request_validation import _positive_int, validate_query_payload


class QueryPayloadValidationTests(unittest.TestCase):
    def test_accepts_valid_question_and_generates_conversation_id(self):
        question, conversation_id = validate_query_payload({"question": "  Explain RAG  "})
        self.assertEqual(question, "Explain RAG")
        self.assertTrue(conversation_id)

    def test_rejects_non_object_payload(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            validate_query_payload(["question"])

    def test_rejects_missing_or_non_string_question(self):
        for payload in ({}, {"question": None}, {"question": 123}):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "Question"):
                    validate_query_payload(payload)

    def test_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_query_payload({"question": "   "})

    def test_rejects_oversized_question(self):
        with self.assertRaisesRegex(ValueError, "too long"):
            validate_query_payload({"question": "x" * 4001})

    def test_rejects_invalid_conversation_id(self):
        for value in ("", " ", 42, "x" * 129):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Conversation ID"):
                    validate_query_payload({"question": "hello", "conversation_id": value})

    def test_positive_integer_setting_uses_default_when_unset(self):
        self.assertEqual(_positive_int(None, "MAX_QUESTION_CHARS", 4000), 4000)

    def test_positive_integer_setting_accepts_positive_value(self):
        self.assertEqual(_positive_int("2500", "MAX_QUESTION_CHARS", 4000), 2500)

    def test_positive_integer_setting_rejects_invalid_or_non_positive_values(self):
        for raw in ("abc", "0", "-1"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _positive_int(raw, "MAX_QUESTION_CHARS", 4000)


if __name__ == "__main__":
    unittest.main()
