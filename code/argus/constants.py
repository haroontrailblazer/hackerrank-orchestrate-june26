"""Allowed vocabularies and mappings for the Multi-Modal Evidence Review task.

Every value the system emits must come from these lists (see problem_statement.md).
Centralising them here lets the agents validate/repair their own output and keeps
the prompts and the post-processing in lock-step.
"""

from __future__ import annotations

# ---- Output column order (exact, required by the grader) --------------------
OUTPUT_COLUMNS: list[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

INPUT_COLUMNS: list[str] = ["user_id", "image_paths", "user_claim", "claim_object"]

# ---- Allowed enums ----------------------------------------------------------
CLAIM_OBJECTS = {"car", "laptop", "package"}

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}

SEVERITIES = {"none", "low", "medium", "high", "unknown"}

OBJECT_PARTS: dict[str, set[str]] = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

# Stable display order for risk_flags so output is deterministic regardless of
# the order in which the rule layer discovers them.
RISK_FLAG_ORDER = [
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

# ---- Issue -> evidence-requirement "family" --------------------------------
# Maps a concrete issue_type to the `applies_to` family used in
# evidence_requirements.csv, per object. Used to pick the right minimum-evidence
# rule for a claim.
_CAR_PANEL = "dent or scratch"
_CAR_GLASS = "crack, broken, or missing part"
_LAPTOP_SCREEN = "screen, keyboard, or trackpad"
_LAPTOP_BODY = "hinge, lid, corner, body, or port"
_PKG_EXTERIOR = "crushed, torn, or seal damage"
_PKG_STAIN = "water, stain, or label damage"
_PKG_CONTENTS = "contents or inner item"

ISSUE_FAMILY: dict[str, dict[str, str]] = {
    "car": {
        "dent": _CAR_PANEL,
        "scratch": _CAR_PANEL,
        "crack": _CAR_GLASS,
        "glass_shatter": _CAR_GLASS,
        "broken_part": _CAR_GLASS,
        "missing_part": _CAR_GLASS,
    },
    "laptop": {
        "crack": _LAPTOP_SCREEN,
        "glass_shatter": _LAPTOP_SCREEN,
        "stain": _LAPTOP_SCREEN,
        "missing_part": _LAPTOP_SCREEN,
        "scratch": _LAPTOP_BODY,
        "dent": _LAPTOP_BODY,
        "broken_part": _LAPTOP_BODY,
    },
    "package": {
        "crushed_packaging": _PKG_EXTERIOR,
        "torn_packaging": _PKG_EXTERIOR,
        "broken_part": _PKG_EXTERIOR,
        "water_damage": _PKG_STAIN,
        "stain": _PKG_STAIN,
        "missing_part": _PKG_CONTENTS,
    },
}

# Parts that, for a given object, indicate the claim is about contents/inner item
# (so a different minimum-evidence rule applies).
PART_FAMILY_OVERRIDE: dict[str, dict[str, str]] = {
    "package": {
        "contents": _PKG_CONTENTS,
        "item": _PKG_CONTENTS,
        "label": _PKG_STAIN,
        "seal": _PKG_EXTERIOR,
        "box": _PKG_EXTERIOR,
        "package_corner": _PKG_EXTERIOR,
        "package_side": _PKG_EXTERIOR,
    },
    "car": {
        "windshield": _CAR_GLASS,
        "headlight": _CAR_GLASS,
        "taillight": _CAR_GLASS,
        "side_mirror": _CAR_GLASS,
    },
    "laptop": {
        "screen": _LAPTOP_SCREEN,
        "keyboard": _LAPTOP_SCREEN,
        "trackpad": _LAPTOP_SCREEN,
        "hinge": _LAPTOP_BODY,
        "lid": _LAPTOP_BODY,
        "corner": _LAPTOP_BODY,
        "port": _LAPTOP_BODY,
        "base": _LAPTOP_BODY,
    },
}

DEFAULT_FAMILY = "general claim review"

SEVERITY_RANK = {"none": 0, "unknown": 0, "low": 1, "medium": 2, "high": 3}


def coerce_part(claim_object: str, part: str | None) -> str:
    """Return a valid object_part for the object, falling back to 'unknown'."""
    if not part:
        return "unknown"
    part = part.strip().lower()
    valid = OBJECT_PARTS.get(claim_object, set())
    return part if part in valid else "unknown"


def coerce_issue(issue: str | None) -> str:
    if not issue:
        return "unknown"
    issue = issue.strip().lower()
    return issue if issue in ISSUE_TYPES else "unknown"


def coerce_severity(sev: str | None) -> str:
    if not sev:
        return "unknown"
    sev = sev.strip().lower()
    return sev if sev in SEVERITIES else "unknown"


def evidence_family(claim_object: str, issue_type: str, object_part: str) -> str:
    """Resolve the evidence-requirement family for an (object, issue, part)."""
    part_map = PART_FAMILY_OVERRIDE.get(claim_object, {})
    if object_part in part_map:
        return part_map[object_part]
    issue_map = ISSUE_FAMILY.get(claim_object, {})
    if issue_type in issue_map:
        return issue_map[issue_type]
    return DEFAULT_FAMILY
