"""Resilience & control harness that wraps each agent step.

The agents themselves are untouched; the orchestrator routes every step through
`ResilienceHarness.run`, which adds, around the call:

* rate limiting  -- a shared token bucket caps model calls/minute (RPM)
* circuit breaking -- after N consecutive failures on a key, fast-fail for a
  cooldown instead of hammering a down provider
* timeout        -- a per-step wall-clock cap (a hung call is abandoned)
* step retries   -- retry the whole step with backoff (on top of the backend's
  own network retries)
* error isolation -- on total failure the step returns a caller-supplied safe
  fallback value, so one bad step never crashes the whole claim

`network=False` steps (the deterministic rule agents) get error isolation only.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


class CircuitBreaker:
    """Per-key consecutive-failure breaker with a half-open trial after cooldown."""

    def __init__(self, threshold: int, reset_timeout: float, time_func=time.monotonic) -> None:
        self.threshold = max(1, threshold)
        self.reset_timeout = reset_timeout
        self._now = time_func
        self._failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            opened = self._opened_at.get(key)
            if opened is None:
                return True
            # Open: allow a single trial once the cooldown has elapsed (half-open).
            return (self._now() - opened) >= self.reset_timeout

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures[key] = 0
            self._opened_at.pop(key, None)

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._failures[key] += 1
            if self._failures[key] >= self.threshold:
                self._opened_at[key] = self._now()


class RateLimiter:
    """Token bucket. rpm<=0 means unlimited (no-op)."""

    def __init__(self, rpm: int, burst: int | None = None,
                 time_func=time.monotonic, sleep_func=time.sleep) -> None:
        self.unlimited = rpm <= 0
        self.rate = rpm / 60.0 if rpm > 0 else 0.0  # tokens/sec
        self.capacity = float(burst if burst is not None else max(1, rpm))
        self.tokens = self.capacity
        self._now = time_func
        self._sleep = sleep_func
        self._last = self._now()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._now()
        elapsed = now - self._last
        self._last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    def try_acquire(self) -> bool:
        if self.unlimited:
            return True
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def acquire(self) -> None:
        if self.unlimited:
            return
        while not self.try_acquire():
            self._sleep(0.05)


@dataclass
class HarnessStats:
    attempts: int = 0
    retries: int = 0
    timeouts: int = 0
    failures: int = 0
    circuit_skips: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def as_dict(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "retries": self.retries,
            "timeouts": self.timeouts,
            "failures": self.failures,
            "circuit_skips": self.circuit_skips,
        }


class _StepFailed(Exception):
    pass


def _run_with_timeout(func, timeout: float):
    """Run func in a daemon thread, abandoning it if it exceeds `timeout`.

    (Python can't force-kill a thread; an abandoned call keeps running but its
    result is ignored. Acceptable for bounded batch runs on Windows where
    signal-based timeouts are unavailable.)
    """
    box: dict = {}

    def target():
        try:
            box["value"] = func()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"step exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class ResilienceHarness:
    def __init__(self, *, timeout: float, retries: int, rpm: int,
                 circuit_threshold: int, circuit_reset: float,
                 time_func=time.monotonic, sleep_func=time.sleep) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self._sleep = sleep_func
        self.limiter = RateLimiter(rpm, time_func=time_func, sleep_func=sleep_func)
        self.circuit = CircuitBreaker(circuit_threshold, circuit_reset, time_func=time_func)
        self.stats = HarnessStats()

    def run(self, name: str, func, fallback, *, key: str | None = None, network: bool = True):
        """Execute `func` with the full control stack; return its value or
        `fallback()` on failure. Never raises."""
        key = key or name

        if network and not self.circuit.allow(key):
            self.stats._inc("circuit_skips")
            return fallback()

        try:
            value = self._attempt(func, network)
        except _StepFailed:
            if network:
                self.circuit.record_failure(key)
            self.stats._inc("failures")
            return fallback()
        else:
            if network:
                self.circuit.record_success(key)
            return value

    def _attempt(self, func, network: bool):
        attempt = 0
        while True:
            if network:
                self.limiter.acquire()
            self.stats._inc("attempts")
            try:
                if network and self.timeout and self.timeout > 0:
                    return _run_with_timeout(func, self.timeout)
                return func()
            except TimeoutError:
                self.stats._inc("timeouts")
            except Exception:  # noqa: BLE001 - isolate every step error
                pass
            if attempt < self.retries:
                attempt += 1
                self.stats._inc("retries")
                self._sleep(min(0.1 * (2 ** attempt), 5.0))
                continue
            raise _StepFailed
