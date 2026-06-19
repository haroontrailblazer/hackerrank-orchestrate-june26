"""Resilience harness: circuit breaker, rate limiter, and the guarded runner.

All deterministic and offline -- the clock/sleep are injected so timing logic is
tested without real waits (except one short real-timeout case).
"""

import time

from argus.harness import CircuitBreaker, RateLimiter, ResilienceHarness


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, d):
        self.t += d


def boom():
    raise RuntimeError("provider error")


# --- CircuitBreaker --------------------------------------------------------
def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(threshold=2, reset_timeout=10, time_func=Clock())
    assert cb.allow("k")
    cb.record_failure("k")
    assert cb.allow("k")  # one failure, still closed
    cb.record_failure("k")
    assert not cb.allow("k")  # threshold reached -> open


def test_circuit_half_opens_after_reset_then_success_closes():
    clk = Clock()
    cb = CircuitBreaker(threshold=2, reset_timeout=10, time_func=clk)
    cb.record_failure("k")
    cb.record_failure("k")
    assert not cb.allow("k")
    clk.advance(11)
    assert cb.allow("k")  # cooldown elapsed -> trial allowed
    cb.record_success("k")
    assert cb.allow("k")


# --- RateLimiter -----------------------------------------------------------
def test_rate_limiter_unlimited_is_noop():
    rl = RateLimiter(rpm=0)
    assert all(rl.try_acquire() for _ in range(50))


def test_rate_limiter_token_bucket_refills_over_time():
    clk = Clock()
    rl = RateLimiter(rpm=60, burst=2, time_func=clk)  # 1 token/sec, bucket 2
    assert rl.try_acquire() and rl.try_acquire()
    assert not rl.try_acquire()  # bucket empty
    clk.advance(1.0)
    assert rl.try_acquire()  # one token refilled
    assert not rl.try_acquire()


# --- ResilienceHarness -----------------------------------------------------
def _harness(**kw):
    cfg = dict(timeout=0, retries=0, rpm=0, circuit_threshold=5,
               circuit_reset=10, sleep_func=lambda s: None)
    cfg.update(kw)
    return ResilienceHarness(**cfg)


def test_run_returns_value_on_success():
    assert _harness().run("step", lambda: 42, lambda: -1) == 42


def test_run_retries_then_succeeds():
    h = _harness(retries=2)
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert h.run("step", f, lambda: "fb") == "ok"
    assert calls["n"] == 3
    assert h.stats.retries >= 2


def test_run_returns_fallback_after_exhausting_retries():
    h = _harness(retries=1)
    assert h.run("step", boom, lambda: "fb") == "fb"
    assert h.stats.failures >= 1


def test_run_times_out_and_returns_fallback():
    h = _harness(timeout=0.05, retries=0)

    def slow():
        time.sleep(0.3)
        return "late"

    assert h.run("step", slow, lambda: "fb") == "fb"
    assert h.stats.timeouts >= 1


def test_open_circuit_fast_fails_without_calling_func():
    h = _harness(retries=0, circuit_threshold=1, circuit_reset=100)
    h.run("k", boom, lambda: "fb", key="k")  # 1 failure -> opens
    called = {"n": 0}

    def f():
        called["n"] += 1
        return "ran"

    out = h.run("k", f, lambda: "fb2", key="k")
    assert out == "fb2"
    assert called["n"] == 0  # never called -- circuit open
    assert h.stats.circuit_skips >= 1


def test_local_step_isolates_errors_without_circuit_or_timeout():
    # network=False: still error-isolated, but no circuit/limiter/timeout.
    h = _harness()
    assert h.run("local", boom, lambda: "safe", network=False) == "safe"
