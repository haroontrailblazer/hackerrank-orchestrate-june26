"""NVIDIA backend (NVIDIA NIM / API Catalog vision-language models).

NVIDIA's hosted VLMs (build.nvidia.com -> https://integrate.api.nvidia.com/v1)
and self-hosted NIM containers both expose an OpenAI-compatible
/v1/chat/completions endpoint, so we drive them with the official `openai` SDK
pointed at NVIDIA's base_url -- this is exactly the integration NVIDIA documents.

Unlike OpenAI, NIM/vLLM models do not reliably support the strict json_schema
`.parse()` helper, so we request a JSON object and validate it ourselves
(`extract_json`), which is portable across the catalog's models.

Auth:   NVIDIA_API_KEY  (nvapi-...)
Base:   NVIDIA_BASE_URL (defaults to the hosted API Catalog; point at your NIM
        container, e.g. http://localhost:8000/v1, to self-host)
Model:  any catalog VLM slug, e.g. meta/llama-3.2-90b-vision-instruct or a newer
        qwen / kimi / minimax vision model (set ARGUS_VISION_MODEL).
"""

from __future__ import annotations

import json
import os
import re
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

NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _safe_default(model_cls: Type[M]) -> M:
    try:
        return model_cls()  # all fields have defaults
    except Exception:
        return model_cls.model_construct()  # last resort, no validation


def _loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json(model_cls: Type[M], content: str) -> M:
    """Parse a (possibly messy) model response into `model_cls`, never raising.

    Tolerates code fences and surrounding prose, fills missing fields from
    schema defaults, and falls back to a safe default on garbage.
    """
    if not content or not content.strip():
        return _safe_default(model_cls)

    obj = _loads(content)
    if obj is None:
        # Grab the first balanced-looking { ... } block from prose / code fence.
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            obj = _loads(m.group(0))
    if not isinstance(obj, dict):
        return _safe_default(model_cls)

    try:
        return model_cls.model_validate(obj)
    except Exception:
        # Keep only keys the model knows, drop the rest, and retry.
        known = {k: v for k, v in obj.items() if k in model_cls.model_fields}
        try:
            return model_cls.model_validate(known)
        except Exception:
            try:
                return model_cls.model_construct(**known)
            except Exception:
                return _safe_default(model_cls)


class NvidiaBackend(LLMBackend):
    name = "nvidia"

    def __init__(self, *, vision_model: str, reasoning_model: str, base_url: str,
                 max_retries: int, timeout: float, usage=None) -> None:
        super().__init__(usage)
        import openai  # NVIDIA documents the OpenAI Python SDK as the client

        self._openai = openai
        self._client = openai.OpenAI(
            base_url=base_url or NVIDIA_DEFAULT_BASE_URL,
            api_key=os.getenv("NVIDIA_API_KEY") or "missing-NVIDIA_API_KEY",
            max_retries=0,
            timeout=timeout,
        )
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

        # Ask for JSON with the exact field names (portable structured output).
        keys = ", ".join(response_model.model_fields.keys())
        json_directive = (
            f"\n\nRespond with ONLY a single JSON object (no prose, no code fence) "
            f"containing exactly these keys: {keys}. Use the allowed values described above."
        )
        content: list[dict] = [{"type": "text", "text": user_text + json_directive}]
        for img in images:
            if img.ok and img.b64:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{img.media_type};base64,{img.b64}"}}
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
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )

        try:
            resp = _call()
        except self._openai.BadRequestError:
            # Some NIM models reject response_format; retry once without it.
            resp = self._client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0
            )

        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage.record(
                input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                images=len(images),
            )
        text = resp.choices[0].message.content or ""
        return extract_json(response_model, text)

    def _retryable(self):
        o = self._openai
        return (o.RateLimitError, o.APIConnectionError, o.InternalServerError)
