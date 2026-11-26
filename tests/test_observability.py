"""Tests for privacy-safe structured RAG observability events."""
import json
import unittest

import server


class ObservabilityTests(unittest.TestCase):
    def test_success_logs_stage_name_and_duration_without_arguments(self):
        secret_prompt = "do-not-log-this-question"
        with self.assertLogs("rag.observability", level="INFO") as captured:
            result, duration_ms = server._run_observed(
                "unit_test", "request-123", lambda value: value.upper(), secret_prompt
            )

        self.assertEqual(result, secret_prompt.upper())
        self.assertGreaterEqual(duration_ms, 0)
        event = json.loads(captured.records[0].getMessage())
        self.assertEqual(event["event"], "rag_stage_completed")
        self.assertEqual(event["request_id"], "request-123")
        self.assertEqual(event["stage"], "unit_test")
        self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertNotIn(secret_prompt, captured.records[0].getMessage())

    def test_failure_logs_error_type_and_reraises(self):
        def fail():
            raise RuntimeError("private failure detail")

        with self.assertLogs("rag.observability", level="INFO") as captured:
            with self.assertRaisesRegex(RuntimeError, "private failure detail"):
                server._run_observed("unit_test", "request-456", fail)

        event = json.loads(captured.records[0].getMessage())
        self.assertEqual(event["event"], "rag_stage_failed")
        self.assertEqual(event["request_id"], "request-456")
        self.assertEqual(event["stage"], "unit_test")
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertNotIn("private failure detail", captured.records[0].getMessage())


if __name__ == "__main__":
    unittest.main()
