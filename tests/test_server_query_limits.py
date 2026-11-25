"""Route-level checks for RAG query concurrency rejection."""
import unittest
from unittest.mock import patch

import server
from query_limiter import QueryConcurrencyLimiter


class _FakeRequest:
    async def json(self):
        return {"question": "Explain retrieval augmented generation"}


class QueryEndpointLimitTests(unittest.IsolatedAsyncioTestCase):
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
