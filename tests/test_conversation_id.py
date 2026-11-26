"""Regression tests for conversation ID creation on new chats."""
import unittest
from unittest.mock import patch

from server import _resolve_conversation_id


class ResolveConversationIdTests(unittest.TestCase):
    def test_missing_id_generates_uuid(self):
        with patch("server.uuid.uuid4", return_value="generated-id"):
            self.assertEqual(_resolve_conversation_id({}), "generated-id")

    def test_null_id_generates_uuid(self):
        with patch("server.uuid.uuid4", return_value="generated-id"):
            self.assertEqual(_resolve_conversation_id({"conversation_id": None}), "generated-id")

    def test_empty_id_generates_uuid(self):
        with patch("server.uuid.uuid4", return_value="generated-id"):
            self.assertEqual(_resolve_conversation_id({"conversation_id": ""}), "generated-id")

    def test_existing_id_is_preserved(self):
        self.assertEqual(
            _resolve_conversation_id({"conversation_id": "existing-id"}),
            "existing-id",
        )


if __name__ == "__main__":
    unittest.main()
