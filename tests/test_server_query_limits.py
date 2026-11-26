"""Route-level checks for RAG query concurrency rejection."""
import unittest
from unittest.mock import patch

import server
from query_limiter import QueryConcurrencyLimiter


class _FakeRequest:
    async def json(self):
        return {"question": "Explain retrieval augmented generation"}


class QueryEndpointLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_releases_capacity_when_stream_is_cancelled(self):
        limiter = QueryConcurrencyLimiter(1)
        history = {}
        with patch.object(server, "MODELS_LOADED", True), patch.object(
            server, "QUERY_LIMITER", limiter
        ), patch.object(server, "_load_history", return_value=history), patch.object(
            server, "_save_history"
        ):
            response = await server.query_endpoint(_FakeRequest())
            self.assertEqual(limiter.active, 1)
            stream = response.body_iterator
            first_event = await stream.__anext__()
            self.assertIn("conversation_id", first_event)
            await stream.aclose()

        self.assertEqual(limiter.active, 0)

    async def test_returns_429_when_query_capacity_is_exhausted(self):
        limiter = QueryConcurrencyLimiter(1)
        self.assertTrue(limiter.try_acquire())

        with patch.object(server, "MODELS_LOADED", True), patch.object(
            server, "QUERY_LIMITER", limiter
        ):
            response = await server.query_endpoint(_FakeRequest())

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers.get("retry-after"), "2")
        self.assertEqual(limiter.active, 1)
        limiter.release()


if __name__ == "__main__":
    unittest.main()
