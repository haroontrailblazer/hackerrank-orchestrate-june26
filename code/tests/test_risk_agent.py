"""RiskAgent intended contract (derived from problem_statement.md + sample labels)."""

from argus.agents.risk_agent import RiskAgent
from argus.schemas import ConversationAnalysis, ImageFinding, UserHistory


def conv(issue="dent", part="rear_bumper", severity="unspecified"):
    return ConversationAnalysis(
        asserted_issue_type=issue, asserted_object_part=part, asserted_severity=severity
    )


def clean_finding(**over):
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


AGENT = RiskAgent()


def test_clean_supported_case_has_no_flags():
    risk, _ = AGENT.assess(conv(), [clean_finding()], None, "car")
    assert risk == "none"


def test_blurry_image_flag():
    risk, b = AGENT.assess(conv(), [clean_finding(blurry=True)], None, "car")
    assert b["blurry_image"] and "blurry_image" in risk


def test_user_history_risk_also_triggers_manual_review():
    hist = UserHistory(user_id="u", history_flags="user_history_risk")
    _, b = AGENT.assess(conv(), [clean_finding()], hist, "car")
    assert b["user_history_risk"] and b["manual_review_required"]


def test_history_manual_review_without_history_risk():
    hist = UserHistory(user_id="u", history_flags="manual_review_required")
    _, b = AGENT.assess(conv(), [clean_finding()], hist, "car")
    assert b["manual_review_required"] and not b["user_history_risk"]


def test_non_original_image_triggers_authenticity_and_manual_review():
    _, b = AGENT.assess(conv(), [clean_finding(non_original_image=True)], None, "car")
    assert b["non_original_image"] and b["manual_review_required"]


def test_wrong_object_when_detected_object_differs():
    f = clean_finding(detected_object="package", object_matches_claim=False)
    _, b = AGENT.assess(conv(), [f], None, "car")
    assert b["wrong_object"] and b["manual_review_required"]


def test_damage_not_visible_when_part_shown_but_no_damage():
    f = clean_finding(damage_visible=False, observed_issue_type="none", observed_severity="none")
    _, b = AGENT.assess(conv(issue="dent"), [f], None, "car")
    assert b["damage_not_visible"]


def test_no_claim_mismatch_for_same_part_cosmetic_difference():
    # Claimed dent, VLM reads scratch, but it's on the SAME claimed part and not
    # exaggerated -> damage to that part IS present, so this is NOT a contradiction.
    f = clean_finding(observed_issue_type="scratch", observed_severity="low")
    _, b = AGENT.assess(conv(issue="dent"), [f], None, "car")
    assert not b["claim_mismatch"]


def test_claim_mismatch_when_damage_on_different_part():
    # Claimed hood, but the visible damage is on the front bumper; the claimed
    # part itself shows nothing -> mismatch (case_008 shape).
    f = clean_finding(
        shows_claimed_part=False,
        observed_object_part="front_bumper",
        observed_issue_type="broken_part",
        observed_severity="high",
    )
    _, b = AGENT.assess(conv(issue="scratch", part="hood"), [f], None, "car")
    assert b["claim_mismatch"]


def test_claim_mismatch_on_severity_exaggeration():
    f = clean_finding(observed_issue_type="scratch", observed_severity="low")
    _, b = AGENT.assess(conv(issue="scratch", severity="severe"), [f], None, "car")
    assert b["claim_mismatch"]


def test_severity_mismatch_when_severe_claim_but_only_medium_observed():
    # "pretty bad" (severe) but the visible damage is a medium dent -> exaggeration.
    f = clean_finding(observed_issue_type="dent", observed_severity="medium")
    _, b = AGENT.assess(conv(issue="dent", severity="severe"), [f], None, "car")
    assert b["claim_mismatch"]


def test_no_severity_mismatch_when_severe_claim_matches_high_damage():
    f = clean_finding(observed_issue_type="broken_part", observed_severity="high")
    _, b = AGENT.assess(conv(issue="broken_part", severity="severe"), [f], None, "car")
    assert not b["claim_mismatch"]


def test_text_instruction_present_flag():
    _, b = AGENT.assess(conv(), [clean_finding(contains_instruction_text=True)], None, "car")
    assert b["text_instruction_present"]


def test_flags_emitted_in_canonical_order():
    f = clean_finding(blurry=True, cropped_or_obstructed=True)
    risk, _ = AGENT.assess(conv(), [f], None, "car")
    # blurry_image precedes cropped_or_obstructed in RISK_FLAG_ORDER
    assert risk.index("blurry_image") < risk.index("cropped_or_obstructed")
