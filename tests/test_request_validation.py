"""Tests for bounded and well-formed RAG query requests."""
import unittest
from request_validation import validate_query_payload


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
        for value in ("", " " , 42, "x" * 129):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Conversation ID"):
                    validate_query_payload({"question": "hello", "conversation_id": value})


if __name__ == "__main__":
    unittest.main()
