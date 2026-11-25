"""Tests for bounded upstream model request timeout configuration."""
import unittest

from config import _positive_float_env


class PositiveFloatEnvironmentTests(unittest.TestCase):
    def test_uses_default_when_unset(self):
        self.assertEqual(_positive_float_env("RAG_LLM_TIMEOUT_SECONDS", 60.0), 60.0)

    def test_accepts_positive_integer_or_decimal(self):
        self.assertEqual(_positive_float_env("RAG_LLM_TIMEOUT_SECONDS", 60.0), 60.0)

    def test_rejects_invalid_zero_and_negative_values(self):
        import os
        from unittest.mock import patch

        for raw in ("invalid", "0", "-2", "nan"):
            with self.subTest(raw=raw), patch.dict(os.environ, {"RAG_LLM_TIMEOUT_SECONDS": raw}):
                with self.assertRaisesRegex(ValueError, "positive number"):
                    _positive_float_env("RAG_LLM_TIMEOUT_SECONDS", 60.0)


if __name__ == "__main__":
    unittest.main()
