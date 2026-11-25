"""Tests for the process-local query concurrency guard."""
import unittest
from query_limiter import QueryConcurrencyLimiter, _positive_limit


class QueryConcurrencyLimiterTests(unittest.TestCase):
    def test_respects_configured_limit(self):
        limiter = QueryConcurrencyLimiter(2)
        self.assertTrue(limiter.try_acquire())
        self.assertTrue(limiter.try_acquire())
        self.assertFalse(limiter.try_acquire())
        self.assertEqual(limiter.active, 2)

    def test_release_allows_another_query(self):
        limiter = QueryConcurrencyLimiter(1)
        self.assertTrue(limiter.try_acquire())
        limiter.release()
        self.assertTrue(limiter.try_acquire())
        self.assertEqual(limiter.active, 1)

    def test_release_does_not_make_counter_negative(self):
        limiter = QueryConcurrencyLimiter(1)
        limiter.release()
        self.assertEqual(limiter.active, 0)

    def test_rejects_invalid_limit(self):
        for value in (0, -1, True, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    QueryConcurrencyLimiter(value)

    def test_environment_limit_defaults_and_validates(self):
        self.assertEqual(_positive_limit(None), 2)
        self.assertEqual(_positive_limit("3"), 3)
        for value in ("abc", "0", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "MAX_CONCURRENT_QUERIES"):
                    _positive_limit(value)


if __name__ == "__main__":
    unittest.main()
