"""In-memory sliding-window rate limiter (single-instance deploy)."""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, window_s: float = 60.0):
        self.window_s = window_s
        self._hits: dict = {}
        self._lock = threading.Lock()
        self._last_gc = time.monotonic()

    def _gc(self, now: float) -> None:
        """Once per window, drop keys whose every hit has left the window (one-off IPs/sandboxes must not pile up)."""
        if now - self._last_gc < self.window_s:
            return
        self._last_gc = now
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] > self.window_s]:
            del self._hits[key]

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        with self._lock:
            self._gc(now)
            q = self._hits.get(key)
            if q is None:
                q = self._hits[key] = deque()
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
