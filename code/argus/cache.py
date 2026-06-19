"""Disk-backed cache for per-image vision findings.

Keyed by (image content hash, model, claim_object). The same photo referenced by
several claims, or a re-run after a crash, costs zero extra VLM calls. This is a
real cost lever, so cache hits are tracked in the operational analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class VisionCache:
    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = Lock()
        self._data: dict[str, dict] = {}
        if enabled and path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    @staticmethod
    def make_key(sha: str, model: str, claim_object: str) -> str:
        return f"{sha}|{model}|{claim_object}"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=0), encoding="utf-8")
