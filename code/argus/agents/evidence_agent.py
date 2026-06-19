"""EvidenceAgent (deterministic): decide whether the image set is sufficient.

Rule-based on purpose -- the visual reading already came from the VLM; this layer
just applies the minimum-evidence policy (evidence_requirements.csv) to those
findings so the decision is reproducible and auditable.
"""

from __future__ import annotations

from argus.constants import evidence_family
from argus.schemas import ConversationAnalysis, ImageFinding


class EvidenceDecision:
    def __init__(self, met: bool, reason: str, valid_image: bool) -> None:
        self.met = met
        self.reason = reason
        self.valid_image = valid_image


class EvidenceAgent:
    def __init__(self, evidence_lookup: dict[tuple[str, str], str]) -> None:
        # (claim_object, applies_to) -> minimum_image_evidence text
        self.evidence_lookup = evidence_lookup

    def assess(
        self,
        conversation: ConversationAnalysis,
        findings: list[ImageFinding],
        claim_object: str,
    ) -> EvidenceDecision:
        part = conversation.asserted_object_part or "unknown"
        usable = [f for f in findings if f.usable]
        authentic = [f for f in usable if not f.non_original_image and not f.possible_manipulation]

        valid_image = any(
            f.object_matches_claim and not f.cropped_or_obstructed for f in authentic
        )
        met = any(
            f.shows_claimed_part and not f.blurry and not f.cropped_or_obstructed
            for f in authentic
        )

        family = evidence_family(claim_object, conversation.asserted_issue_type, part)
        min_text = self.evidence_lookup.get((claim_object, family)) or self.evidence_lookup.get(
            (claim_object, "general claim review"), ""
        )

        if met:
            reason = f"The {part} is visible clearly enough to evaluate the claim."
        elif not usable:
            reason = "No usable image was submitted, so the claim cannot be evaluated."
        elif not authentic:
            reason = "The submitted image is not a usable original photo for automated review."
        elif not valid_image and not any(f.object_matches_claim for f in authentic):
            reason = (
                f"The image does not clearly show the claimed {claim_object}, "
                "so the claim cannot be verified."
            )
        else:
            cite = f" Minimum evidence: {min_text}" if min_text else ""
            reason = (
                f"The {part} is not shown clearly enough to verify the claim."
                + cite
            )
        return EvidenceDecision(met=met, reason=reason[:300], valid_image=valid_image)
