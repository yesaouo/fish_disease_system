from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Generic, Hashable, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Minimal thread-safe TTL cache for expensive dataset scans."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: Dict[K, Tuple[float, V]] = {}
        self._lock = threading.Lock()

    def get_or_set(self, key: K, loader: Callable[[], V]) -> V:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is not None:
                ts, value = item
                if now - ts < self._ttl:
                    return value

        value = loader()
        with self._lock:
            self._store[key] = (now, value)
        return value

    def clear(self, key: K | None = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)

