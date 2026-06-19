"""Adjudicator (rules strategy) decision tree."""

from argus.agents.adjudicator import adjudicate_rules
from argus.agents.evidence_agent import EvidenceDecision
from argus.schemas import ConversationAnalysis, ImageFinding


def conv(issue="dent", part="rear_bumper", severity="unspecified"):
    return ConversationAnalysis(
        asserted_issue_type=issue, asserted_object_part=part, asserted_severity=severity
    )


def finding(**over):
    base = dict(
        image_id="img_1",
        detected_object="car",
        object_matches_claim=True,
        shows_claimed_part=True,
        observed_issue_type="dent",
        observed_object_part="rear_bumper",
        observed_severity="medium",
        damage_visible=True,
        usable=True,
    )
    base.update(over)
    return ImageFinding(**base)


def ev(met=True, valid=True):
    return EvidenceDecision(met=met, reason="", valid_image=valid)


def test_supported_when_damage_visible_on_claimed_part():
    v = adjudicate_rules(conv(), [finding()], ev(), {}, "car")
    assert v.claim_status == "supported"
    assert v.issue_type == "dent" and v.object_part == "rear_bumper"
    assert v.severity == "medium" and v.supporting_image_ids == "img_1"


def test_contradicted_when_part_shown_but_no_damage():
    f = finding(damage_visible=False, observed_issue_type="none", observed_severity="none")
    v = adjudicate_rules(conv(), [f], ev(met=True), {}, "car")
    assert v.claim_status == "contradicted"
    assert v.issue_type == "none" and v.severity == "none"


def test_contradicted_on_wrong_object():
    f = finding(detected_object="package", object_matches_claim=False, observed_severity="low")
    v = adjudicate_rules(conv(), [f], ev(valid=False), {"wrong_object": True}, "car")
    assert v.claim_status == "contradicted"
    assert v.issue_type == "unknown" and v.object_part == "unknown"


def test_contradicted_on_claim_mismatch_reports_actual_visible_damage():
    f = finding(observed_issue_type="scratch", observed_object_part="rear_bumper", observed_severity="low")
    v = adjudicate_rules(conv(issue="dent", severity="severe"), [f], ev(), {"claim_mismatch": True}, "car")
    assert v.claim_status == "contradicted"
    assert v.issue_type == "scratch"  # the actually-visible issue, not the claimed one


def test_not_enough_information_when_part_not_shown():
    f = finding(shows_claimed_part=False, damage_visible=False)
    v = adjudicate_rules(conv(part="headlight"), [f], ev(met=False), {}, "car")
    assert v.claim_status == "not_enough_information"
    assert v.supporting_image_ids == "none"
    assert v.object_part == "headlight"  # falls back to the claimed part
    assert v.issue_type == "unknown" and v.severity == "unknown"


def test_missing_contents_with_intact_exterior_is_not_enough_information():
    # case_018 shape: a 'missing contents' claim cannot be verified or refuted
    # from an undamaged box exterior -> not_enough_information, not contradicted.
    f = ImageFinding(
        image_id="img_1", detected_object="package", object_matches_claim=True,
        shows_claimed_part=True, observed_object_part="box", observed_issue_type="none",
        damage_visible=False, usable=True,
    )
    v = adjudicate_rules(
        conv(issue="missing_part", part="contents"), [f], ev(met=True), {}, "package"
    )
    assert v.claim_status == "not_enough_information"


def test_no_images_is_not_enough_information():
    v = adjudicate_rules(conv(), [], ev(met=False, valid=False), {}, "car")
    assert v.claim_status == "not_enough_information"
    assert v.supporting_image_ids == "none"
