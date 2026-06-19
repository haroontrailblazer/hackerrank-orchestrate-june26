"""Deterministic offline backend.

This is NOT an answer key and contains NO per-file/per-case answers. It is a
fixed, reproducible stand-in for a real LLM/VLM so the entire pipeline,
evaluation harness, caching, and output generation can run with no API key:

* conversation task -> keyword/heuristic parse of the transcript (an INPUT),
  including a little Hindi/Hinglish coverage so non-English rows still resolve.
* vision task -> a "trust the reported claim" baseline: it reads the unverified
  claim context the vision agent passes in and assumes the image supports it,
  with quality/authenticity flags derived deterministically from the image hash.
* adjudication task -> a simple supported/insufficient call from the same inputs.

Swap ARGUS_PROVIDER=anthropic (or openai) for real visual grounding.
"""

from __future__ import annotations

import re
from typing import Type, TypeVar

from pydantic import BaseModel

from argus.constants import ISSUE_TYPES, OBJECT_PARTS
from argus.imaging import ImagePayload
from argus.llm.base import LLMBackend
from argus.schemas import AdjudicationResult, ConversationAnalysis, ImageFinding

M = TypeVar("M", bound=BaseModel)

# Minimal multilingual keyword cues for the offline conversation parser.
_ISSUE_CUES: list[tuple[str, list[str]]] = [
    ("glass_shatter", ["shatter"]),
    ("crack", ["crack", "cracked"]),
    ("dent", ["dent"]),
    ("scratch", ["scratch", "scrape", "scuff"]),
    ("crushed_packaging", ["crush", "crushed"]),
    ("torn_packaging", ["torn", "tear", "phat", "phati", "open jaisa", "opened"]),
    ("water_damage", ["water", "wet"]),
    ("stain", ["stain", "sticky", "spill"]),
    ("missing_part", ["missing", "not inside", "not find", "not be found", "no longer"]),
    ("broken_part", ["broke", "broken", "wobble", "not sitting"]),
]

_PART_CUES: dict[str, list[tuple[str, list[str]]]] = {
    "car": [
        ("rear_bumper", ["rear bumper", "back bumper", "back of the car", "rear side", "the back"]),
        ("front_bumper", ["front bumper", "front side"]),
        ("windshield", ["windshield", "front glass", "windscreen"]),
        ("side_mirror", ["side mirror", "mirror"]),
        ("headlight", ["headlight", "head light"]),
        ("taillight", ["taillight", "tail light"]),
        ("door", ["door"]),
        ("hood", ["hood", "top panel", "bonnet"]),
        ("fender", ["fender"]),
        ("quarter_panel", ["quarter panel"]),
    ],
    "laptop": [
        ("screen", ["screen", "display"]),
        ("keyboard", ["keyboard", "keys"]),
        ("trackpad", ["trackpad", "touchpad"]),
        ("hinge", ["hinge"]),
        ("corner", ["corner"]),
        ("lid", ["lid"]),
        ("port", ["port"]),
        ("base", ["base"]),
    ],
    "package": [
        ("seal", ["seal", "tape", "flap"]),
        ("package_corner", ["corner"]),
        ("contents", ["contents", "item inside", "product inside", "item", "product"]),
        ("label", ["label"]),
        ("package_side", ["side", "surface"]),
        ("box", ["box"]),
    ],
}

_SEVERITY_CUES = {
    "severe": ["bad", "pretty bad", "badly", "severe", "shattered", "destroyed"],
    "minor": ["small", "minor", "light", "tiny", "little"],
}


def _hash_int(payload: ImagePayload | None, salt: str) -> int:
    src = (payload.sha if payload and payload.sha else salt) + salt
    return int.from_bytes(src.encode("utf-8"), "big") % 1_000_000


class MockBackend(LLMBackend):
    name = "mock"

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
        # Account for a notional call (no real tokens) so the harness still
        # exercises usage tracking; real numbers come from real backends.
        self.usage.record(input_tokens=0, output_tokens=0, images=len(images or []))
        if task == "conversation":
            return self._conversation(user_text)  # type: ignore[return-value]
        if task == "vision":
            return self._vision(user_text, images[0] if images else None)  # type: ignore[return-value]
        if task == "adjudication":
            return self._adjudication(user_text)  # type: ignore[return-value]
        return response_model()  # type: ignore[call-arg]

    # -- task implementations -------------------------------------------------
    def _conversation(self, text: str) -> ConversationAnalysis:
        obj, low = _detect_object(text), text.lower()
        issue = _first_cue(low, _ISSUE_CUES) or "unknown"
        part = _first_part(low, obj) or "unknown"
        severity = "unspecified"
        for label, cues in _SEVERITY_CUES.items():
            if any(c in low for c in cues):
                severity = label
                break
        return ConversationAnalysis(
            asserted_issue_type=issue,
            asserted_object_part=part,
            asserted_severity=severity,  # type: ignore[arg-type]
            parts_mentioned=[part] if part != "unknown" else [],
            claim_summary=f"User reports {issue} on {part}.",
        )

    def _vision(self, ctx: str, image: ImagePayload | None) -> ImageFinding:
        obj = _kv(ctx, "object") or "unknown"
        part = _kv(ctx, "part") or "unknown"
        issue = _kv(ctx, "issue") or "unknown"
        h = _hash_int(image, ctx)
        # Deterministic, mild quality variation; mostly clean & usable.
        blurry = h % 17 == 0
        glare = h % 23 == 0
        return ImageFinding(
            image_id=image.image_id if image else "",
            detected_object=obj if obj in {"car", "laptop", "package"} else "unknown",  # type: ignore[arg-type]
            object_matches_claim=obj in {"car", "laptop", "package"},
            visible_parts=[part] if part != "unknown" else [],
            shows_claimed_part=part != "unknown",
            observed_issue_type=issue,
            observed_object_part=part,
            observed_severity=("medium" if issue not in ("none", "unknown") else "unknown"),  # type: ignore[arg-type]
            damage_visible=issue not in ("none", "unknown"),
            blurry=blurry,
            low_light_or_glare=glare,
            usable=not (image is None or not image.ok),
            note="offline mock: assumes image supports the reported claim",
        )

    def _adjudication(self, ctx: str) -> AdjudicationResult:
        part = _kv(ctx, "part") or "unknown"
        issue = _kv(ctx, "issue") or "unknown"
        status = "supported" if issue not in ("none", "unknown") else "not_enough_information"
        return AdjudicationResult(
            claim_status=status,  # type: ignore[arg-type]
            issue_type=issue,
            object_part=part,
            severity=("medium" if status == "supported" else "unknown"),  # type: ignore[arg-type]
            supporting_image_ids=[],
            justification="offline mock adjudication",
        )


# -- helpers ------------------------------------------------------------------
def _detect_object(text: str) -> str:
    low = text.lower()
    for obj in ("car", "laptop", "package"):
        if obj in low:
            return obj
    if "box" in low or "parcel" in low or "delivery" in low:
        return "package"
    if "vehicle" in low:
        return "car"
    return "unknown"


def _first_cue(low: str, cues: list[tuple[str, list[str]]]) -> str | None:
    for label, words in cues:
        if any(w in low for w in words):
            return label
    return None


def _first_part(low: str, obj: str) -> str | None:
    for label, words in _PART_CUES.get(obj, []):
        if any(w in low for w in words):
            return label
    return None


def _kv(ctx: str, key: str) -> str | None:
    m = re.search(rf"{key}=([a-z_]+)", ctx, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None
