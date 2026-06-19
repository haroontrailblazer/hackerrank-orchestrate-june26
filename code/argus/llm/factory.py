"""Build the configured backend, sharing one Usage record across the run."""

from __future__ import annotations

from argus.config import Settings
from argus.llm.base import LLMBackend, Usage


def build_backend(settings: Settings, usage: Usage | None = None) -> LLMBackend:
    usage = usage or Usage()
    provider = settings.provider.lower()

    if provider == "mock":
        from argus.llm.mock_backend import MockBackend

        return MockBackend(usage)

    if provider == "anthropic":
        from argus.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            vision_model=settings.vision_model,
            reasoning_model=settings.reasoning_model,
            use_prompt_cache=settings.use_prompt_cache,
            max_retries=settings.max_retries,
            timeout=settings.request_timeout,
            usage=usage,
        )

    if provider == "openai":
        from argus.llm.openai_backend import OpenAIBackend

        return OpenAIBackend(
            vision_model=settings.vision_model,
            reasoning_model=settings.reasoning_model,
            max_retries=settings.max_retries,
            timeout=settings.request_timeout,
            usage=usage,
        )

    raise ValueError(f"Unknown provider: {settings.provider!r} (use mock|anthropic|openai)")
