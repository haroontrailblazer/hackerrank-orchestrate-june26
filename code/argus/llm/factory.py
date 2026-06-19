"""Build the configured backend (with provider fallback), sharing one Usage.

If the selected provider has no API key, Argus falls back to the next available
provider, ending in the offline `mock` backend so a run always completes. The
fallback order is `ARGUS_FALLBACK` (default "openai,anthropic,mock"); set it to
the primary provider alone (e.g. ARGUS_FALLBACK=openai) to disable the mock
safety net, or to "" to fail loudly when the primary is unavailable.
"""

from __future__ import annotations

import os
import sys

from argus.config import Settings, _default_reasoning_model, _default_vision_model
from argus.llm.base import LLMBackend, Usage

_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "nvidia": "NVIDIA_API_KEY"}


def is_available(provider: str) -> bool:
    """A provider is usable if it's the offline mock or its API key is set."""
    p = provider.lower()
    if p == "mock":
        return True
    env = _KEY_ENV.get(p)
    return bool(env and os.getenv(env))


def _construct(provider: str, vision_model: str, reasoning_model: str,
               settings: Settings, usage: Usage) -> LLMBackend:
    p = provider.lower()
    if p == "mock":
        from argus.llm.mock_backend import MockBackend

        return MockBackend(usage)
    if p == "anthropic":
        from argus.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            vision_model=vision_model, reasoning_model=reasoning_model,
            use_prompt_cache=settings.use_prompt_cache, max_retries=settings.max_retries,
            timeout=settings.request_timeout, usage=usage,
        )
    if p == "openai":
        from argus.llm.openai_backend import OpenAIBackend

        return OpenAIBackend(
            vision_model=vision_model, reasoning_model=reasoning_model,
            max_retries=settings.max_retries, timeout=settings.request_timeout, usage=usage,
        )
    if p == "nvidia":
        from argus.llm.nvidia_backend import NvidiaBackend

        return NvidiaBackend(
            vision_model=vision_model, reasoning_model=reasoning_model,
            base_url=settings.nvidia_base_url, max_retries=settings.max_retries,
            timeout=settings.request_timeout, usage=usage,
        )
    raise ValueError(f"Unknown provider: {provider!r} (use mock|anthropic|openai|nvidia)")


def _resolve_chain(settings: Settings) -> list[str]:
    """Ordered, de-duplicated providers: primary (if keyed) then available
    fallbacks, with mock appended as the offline guarantee."""
    primary = settings.provider.lower()
    fb_env = os.getenv("ARGUS_FALLBACK", "openai,anthropic,mock")
    fb_list = [p.strip().lower() for p in fb_env.split(",") if p.strip()]

    chain: list[str] = []

    def add(p: str) -> None:
        if p not in chain:
            chain.append(p)

    if is_available(primary):
        add(primary)
    elif primary != "mock":
        print(
            f"[argus] provider '{primary}' is unavailable (missing "
            f"{_KEY_ENV.get(primary, 'key')}); using fallback chain",
            file=sys.stderr,
        )

    for p in fb_list:
        if is_available(p):
            add(p)

    if not chain or (fb_env.strip() and "mock" in fb_list):
        add("mock")  # offline guarantee unless the user explicitly removed mock
    if not chain:  # ARGUS_FALLBACK="" and primary unavailable -> still need something
        add(primary if is_available(primary) else "mock")

    # mock never fails, so any provider listed after it is unreachable -- drop it.
    if "mock" in chain:
        chain = chain[: chain.index("mock") + 1]
    return chain


def build_backend(settings: Settings, usage: Usage | None = None) -> LLMBackend:
    usage = usage or Usage()
    primary = settings.provider.lower()
    chain = _resolve_chain(settings)

    backends: list[LLMBackend] = []
    for p in chain:
        if p == primary:
            vm, rm = settings.vision_model, settings.reasoning_model
        else:  # fallbacks use their own provider's appropriate model ids
            vm, rm = _default_vision_model(p), _default_reasoning_model(p)
        backends.append(_construct(p, vm, rm, settings, usage))

    if len(backends) == 1:
        return backends[0]

    from argus.llm.fallback import FallbackBackend

    return FallbackBackend(backends, usage)
