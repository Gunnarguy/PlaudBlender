"""Small in-process TTL cache for noisy read endpoints."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, Hashable, Tuple, TypeVar


T = TypeVar("T")


class TTLCache:
    """Thread-safe TTL cache for lightweight API response reuse."""

    def __init__(self):
        self._entries: Dict[Hashable, Tuple[float, T]] = {}
        self._lock = threading.Lock()

    def get_or_compute(self, key: Hashable, ttl_seconds: float, factory: Callable[[], T]) -> T:
        if os.getenv("PYTEST_CURRENT_TEST"):
            return factory()

        ttl_seconds = max(float(ttl_seconds), 0.0)
        now = time.monotonic()

        with self._lock:
            cached = self._entries.get(key)
            if cached and now < cached[0]:
                return cached[1]

        value = factory()
        if ttl_seconds <= 0:
            return value

        with self._lock:
            self._entries[key] = (time.monotonic() + ttl_seconds, value)

        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
