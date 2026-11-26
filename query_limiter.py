"""Process-local concurrency guard for expensive RAG query streams."""
import os
import threading


def _positive_limit(raw_value: str | None, default: int = 2) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("MAX_CONCURRENT_QUERIES must be a positive integer") from exc
    if value < 1:
        raise ValueError("MAX_CONCURRENT_QUERIES must be a positive integer")
    return value


class QueryConcurrencyLimiter:
    """Thread-safe, process-local cap for simultaneous expensive query streams."""

    def __init__(self, limit: int):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        self.limit = limit
        self._active = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


MAX_CONCURRENT_QUERIES = _positive_limit(os.getenv("MAX_CONCURRENT_QUERIES"))
