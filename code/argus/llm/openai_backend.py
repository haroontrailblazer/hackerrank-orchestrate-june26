"""OpenAI backend (kept feature-parallel to the Anthropic one).

Exists primarily so the evaluation can compare a second model family on the same
pipeline. Uses the SDK's structured-output parse helper with a pydantic schema.
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


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, *, vision_model: str, reasoning_model: str, max_retries: int,
                 timeout: float, usage=None) -> None:
        super().__init__(usage)
        import openai  # lazy import

        self._openai = openai
        self._client = openai.OpenAI(max_retries=0, timeout=timeout)
        self.vision_model = vision_model
        self.reasoning_model = reasoning_model
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
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.media_type};base64,{img.b64}"},
                    }
                )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(self._retryable()),
        )
        def _call():
            return self._client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
                max_tokens=max_tokens,
                temperature=0,  # deterministic classification + reproducible eval
            )

        comp = _call()
        u = getattr(comp, "usage", None)
        if u is not None:
            cached = 0
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            self.usage.record(
                input_tokens=(getattr(u, "prompt_tokens", 0) or 0) - cached,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                cache_read_tokens=cached,
                images=len(images),
            )
        parsed = comp.choices[0].message.parsed
        if parsed is None:
            return response_model()  # type: ignore[call-arg]
        return parsed

    def _retryable(self):
        o = self._openai
        return (o.RateLimitError, o.APIConnectionError, o.InternalServerError)
