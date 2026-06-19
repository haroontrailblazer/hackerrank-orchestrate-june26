"""Provider fallback chain: degrade gracefully when a provider is unavailable
or fails, ending in the offline mock so a run always completes."""

import pytest

from argus.config import Settings
from argus.llm.base import LLMBackend, Usage
from argus.llm.factory import build_backend, is_available
from argus.llm.fallback import FallbackBackend
from argus.schemas import ConversationAnalysis

KEYS = ("NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


class _Boom(LLMBackend):
    name = "boom"

    def complete(self, **kwargs):
        raise RuntimeError("provider down")


class _Ok(LLMBackend):
    name = "ok"

    def complete(self, *, response_model, **kwargs):
        return response_model(asserted_issue_type="dent")


# --- FallbackBackend unit behaviour (no network) ---------------------------
def test_fallback_skips_failing_primary():
    u = Usage()
    fb = FallbackBackend([_Boom(u), _Ok(u)], u)
    out = fb.complete(system="", user_text="", response_model=ConversationAnalysis, task="conversation")
    assert out.asserted_issue_type == "dent"


def test_fallback_uses_primary_when_it_succeeds():
    u = Usage()
    primary = _Ok(u)
    primary.name = "primary"
    fb = FallbackBackend([primary, _Boom(u)], u)  # if it reached _Boom it would raise
    out = fb.complete(system="", user_text="", response_model=ConversationAnalysis, task="conversation")
    assert out.asserted_issue_type == "dent"


def test_fallback_name_lists_the_chain():
    u = Usage()
    fb = FallbackBackend([_Ok(u), _Boom(u)], u)
    assert ">" in fb.name


# --- availability + factory wiring -----------------------------------------
def test_is_available_reflects_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_available("mock") is True
    assert is_available("openai") is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert is_available("openai") is True


def test_nvidia_without_key_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = build_backend(Settings(provider="nvidia"))
    # nvidia skipped (no key); openai is the head; mock terminal guarantee
    assert backend.name.split(">")[0] == "openai"
    assert backend.name.endswith("mock")


def test_no_keys_falls_back_to_mock(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    backend = build_backend(Settings(provider="nvidia"))
    assert backend.name == "mock"
