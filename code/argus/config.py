"""Runtime configuration for Argus.

All knobs are environment-overridable so the same code runs offline (mock
backend, no key) and online (real VLM) without edits. Secrets are read from the
environment only -- never hardcoded (see AGENTS.md section 6.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    # config.py lives at <repo>/code/argus/config.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # --- provider / models ---------------------------------------------------
    # provider: "mock" (offline, deterministic), "anthropic", or "openai".
    provider: str = field(default_factory=lambda: os.getenv("ARGUS_PROVIDER", "mock"))
    # Vision model does the heavy per-image work; reasoning model handles text.
    vision_model: str = field(default_factory=lambda: os.getenv("ARGUS_VISION_MODEL", ""))
    reasoning_model: str = field(default_factory=lambda: os.getenv("ARGUS_REASONING_MODEL", ""))
    # Adjudication strategy: "rules" (deterministic, default) or "llm".
    adjudicator: str = field(default_factory=lambda: os.getenv("ARGUS_ADJUDICATOR", "rules"))

    # --- paths ---------------------------------------------------------------
    repo_root: Path = field(default_factory=_repo_root)
    dataset_dir: Path = field(default=None)  # type: ignore[assignment]
    cache_dir: Path = field(default=None)  # type: ignore[assignment]

    # --- image handling ------------------------------------------------------
    # Downscale long edge before sending to the VLM to control image-token cost.
    max_image_edge: int = field(default_factory=lambda: int(os.getenv("ARGUS_MAX_IMAGE_EDGE", "1024")))
    jpeg_quality: int = field(default_factory=lambda: int(os.getenv("ARGUS_JPEG_QUALITY", "82")))

    # --- concurrency / reliability ------------------------------------------
    max_workers: int = field(default_factory=lambda: int(os.getenv("ARGUS_MAX_WORKERS", "4")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ARGUS_MAX_RETRIES", "4")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("ARGUS_TIMEOUT", "120")))

    # --- caching -------------------------------------------------------------
    use_vision_cache: bool = field(default_factory=lambda: os.getenv("ARGUS_VISION_CACHE", "1") != "0")
    use_prompt_cache: bool = field(default_factory=lambda: os.getenv("ARGUS_PROMPT_CACHE", "1") != "0")

    def __post_init__(self) -> None:
        if self.dataset_dir is None:
            self.dataset_dir = self.repo_root / "dataset"
        if self.cache_dir is None:
            self.cache_dir = self.repo_root / "code" / ".argus_cache"
        # Resolve sensible default model ids per provider.
        if not self.vision_model:
            self.vision_model = _default_vision_model(self.provider)
        if not self.reasoning_model:
            self.reasoning_model = _default_reasoning_model(self.provider)

    # Convenience paths -------------------------------------------------------
    @property
    def claims_csv(self) -> Path:
        return self.dataset_dir / "claims.csv"

    @property
    def sample_csv(self) -> Path:
        return self.dataset_dir / "sample_claims.csv"

    @property
    def user_history_csv(self) -> Path:
        return self.dataset_dir / "user_history.csv"

    @property
    def evidence_csv(self) -> Path:
        return self.dataset_dir / "evidence_requirements.csv"

    @property
    def images_root(self) -> Path:
        return self.dataset_dir


def _default_vision_model(provider: str) -> str:
    if provider == "anthropic":
        # Strong vision + reasoning. Switch to claude-haiku-4-5 to cut cost ~5x
        # (see evaluation/evaluation_report.md for the model comparison).
        return "claude-opus-4-8"
    if provider == "openai":
        return "gpt-4o"
    return "mock-vision"


def _default_reasoning_model(provider: str) -> str:
    if provider == "anthropic":
        return "claude-haiku-4-5"  # cheap text model for claim extraction
    if provider == "openai":
        return "gpt-4o-mini"
    return "mock-reasoning"
