"""Vocabulary coercion + evidence-family mapping (pure functions)."""

from argus.constants import (
    coerce_issue,
    coerce_part,
    coerce_severity,
    evidence_family,
)


def test_coerce_part_invalid_becomes_unknown():
    assert coerce_part("car", "left_wing") == "unknown"


def test_coerce_part_valid_passthrough_and_lowercased():
    assert coerce_part("car", "Rear_Bumper") == "rear_bumper"


def test_coerce_part_wrong_object_vocab_is_unknown():
    # 'screen' is a laptop part, not a car part
    assert coerce_part("car", "screen") == "unknown"


def test_coerce_issue_and_severity_invalid_become_unknown():
    assert coerce_issue("explosion") == "unknown"
    assert coerce_severity("catastrophic") == "unknown"


def test_evidence_family_car_panel():
    assert evidence_family("car", "dent", "door") == "dent or scratch"


def test_evidence_family_car_glass_via_part_override():
    # windshield forces the glass/light family regardless of issue
    assert evidence_family("car", "crack", "windshield") == "crack, broken, or missing part"


def test_evidence_family_package_contents():
    assert evidence_family("package", "missing_part", "contents") == "contents or inner item"


def test_evidence_family_unknown_falls_back_to_general():
    assert evidence_family("car", "unknown", "unknown") == "general claim review"
