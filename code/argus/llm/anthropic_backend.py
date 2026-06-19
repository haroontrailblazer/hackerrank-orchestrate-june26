"""Anthropic (Claude) backend.

Uses the official SDK's structured-output helper (`messages.parse`) so the model
is forced to return our pydantic schema. The frozen system prompt is sent as a
cache_control prefix so it is written once and read at ~0.1x cost on every
subsequent image/claim (see evaluation/evaluation_report.md).
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from argus.imaging import ImagePayload
from argus.llm.base import LLMBackend

M = TypeVar("M", bound=BaseModel)


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, *, vision_model: str, reasoning_model: str, use_prompt_cache: bool,
                 max_retries: int, timeout: float, usage=None) -> None:
        super().__init__(usage)
        import anthropic  # imported lazily so 'mock' provider needs no SDK

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=0, timeout=timeout)
        self.vision_model = vision_model
        self.reasoning_model = reasoning_model
        self.use_prompt_cache = use_prompt_cache
        self._max_retries = max_retries

    def complete(
        self,
        *,
        system: str,
        user_text: str,
        response_model: Type[M],
        task: str,
        images: list[ImagePayload] | None = None,
        max_tokens: int = 1024,
    ) -> M:
        images = images or []
        model = self.vision_model if (images or task == "vision") else self.reasoning_model

        content: list[dict] = [{"type": "text", "text": user_text}]
        for img in images:
            if img.ok and img.b64:
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": img.media_type, "data": img.b64},
                    }
                )

        system_param = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if self.use_prompt_cache
            else system
        )

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(self._retryable()),
        )
        def _call():
            return self._client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system_param,
                messages=[{"role": "user", "content": content}],
                output_format=response_model,
            )

        resp = _call()
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage.record(
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                images=len(images),
            )
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:  # refusal / unparseable -> safe default
            return response_model()  # type: ignore[call-arg]
        return parsed

    def _retryable(self):
        a = self._anthropic
        return (a.RateLimitError, a.APIConnectionError, a.InternalServerError)
