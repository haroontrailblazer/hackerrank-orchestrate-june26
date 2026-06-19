"""EvidenceAgent intended contract: when is the image set sufficient / usable?"""

from argus.agents.evidence_agent import EvidenceAgent
from argus.schemas import ConversationAnalysis, ImageFinding


def conv(part="rear_bumper", issue="dent"):
    return ConversationAnalysis(asserted_issue_type=issue, asserted_object_part=part)


def finding(**over):
    base = dict(
        image_id="img_1",
        detected_object="car",
        object_matches_claim=True,
        shows_claimed_part=True,
        observed_object_part="rear_bumper",
        damage_visible=True,
        usable=True,
    )
    base.update(over)
    return ImageFinding(**base)


AGENT = EvidenceAgent(evidence_lookup={})


def test_clear_part_shown_meets_standard_and_is_valid():
    d = AGENT.assess(conv(), [finding()], "car")
    assert d.met is True and d.valid_image is True


def test_non_original_image_is_not_valid_and_not_met():
    d = AGENT.assess(conv(), [finding(non_original_image=True)], "car")
    assert d.met is False and d.valid_image is False


def test_object_shown_but_claimed_part_missing_is_valid_but_not_met():
    # case_006 shape: right object visible, claimed part not shown -> can't verify
    d = AGENT.assess(conv(part="headlight"), [finding(shows_claimed_part=False)], "car")
    assert d.valid_image is True and d.met is False


def test_cropped_image_of_right_object_is_not_valid():
    d = AGENT.assess(conv(), [finding(cropped_or_obstructed=True)], "car")
    assert d.valid_image is False


def test_no_usable_images_not_met_not_valid():
    d = AGENT.assess(conv(), [finding(usable=False)], "car")
    assert d.met is False and d.valid_image is False
    assert "no usable image" in d.reason.lower()
