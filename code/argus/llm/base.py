"""LLM/VLM backend interface + usage accounting.

A backend turns (system prompt, user text, optional images, target schema) into
a validated pydantic object plus a Usage record. Three implementations exist:
mock (offline/deterministic), anthropic, and openai. Agents are written against
this interface and never import a provider SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Type, TypeVar

from pydantic import BaseModel

from argus.imaging import ImagePayload

M = TypeVar("M", bound=BaseModel)


@dataclass
class Usage:
    """Token/call accounting, aggregated across the run (thread-safe add)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    images: int = 0
    cache_hits: int = 0  # vision-cache short-circuits (no API call)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        images: int = 0,
        calls: int = 1,
    ) -> None:
        with self._lock:
            self.calls += calls
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_read_tokens += cache_read_tokens
            self.cache_write_tokens += cache_write_tokens
            self.images += images

    def note_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "images": self.images,
            "cache_hits": self.cache_hits,
        }


class LLMBackend(ABC):
    """Provider-agnostic structured-completion interface."""

    name: str = "base"

    def __init__(self, usage: Usage | None = None) -> None:
        self.usage = usage or Usage()

    @abstractmethod
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
        """Return a populated instance of `response_model`.

        `task` is a hint ("conversation" | "vision" | "adjudication") used by the
        mock backend; real backends rely on the schema and ignore it.
        """
        raise NotImplementedError
