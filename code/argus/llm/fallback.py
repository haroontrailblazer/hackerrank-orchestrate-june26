"""FallbackBackend: try backends in order until one succeeds.

Wraps an ordered chain of already-constructed backends (sharing one Usage).
Each `complete()` tries them in turn; on any exception it logs a visible
warning and moves to the next. The chain always ends in the offline mock
backend, which never raises -- so a run completes even if every real provider
is down or unkeyed. Fallbacks are logged (never silent) so a degraded run is
obvious in stderr and the usage summary (`provider` shows the chain).
"""

from __future__ import annotations

import sys

from argus.llm.base import LLMBackend, Usage


class FallbackBackend(LLMBackend):
    def __init__(self, backends: list[LLMBackend], usage: Usage) -> None:
        super().__init__(usage)
        if not backends:
            raise ValueError("FallbackBackend needs at least one backend")
        self.backends = backends
        self.name = ">".join(b.name for b in backends)

    def complete(self, **kwargs):
        last_exc: Exception | None = None
        for i, backend in enumerate(self.backends):
            try:
                return backend.complete(**kwargs)
            except Exception as exc:  # noqa: BLE001 - resilience is the whole point
                last_exc = exc
                nxt = self.backends[i + 1].name if i + 1 < len(self.backends) else None
                if nxt is not None:
                    print(
                        f"[argus] backend '{backend.name}' failed "
                        f"({type(exc).__name__}); falling back to '{nxt}'",
                        file=sys.stderr,
                    )
        # Exhausted every backend (should be unreachable: mock never raises).
        raise last_exc  # type: ignore[misc]
