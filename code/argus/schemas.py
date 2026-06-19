"""Pydantic models for structured agent I/O.

These models are the contract between the LLM/VLM backends and the rule layers.
Real backends force the model to fill exactly these fields (structured outputs);
the mock backend constructs them deterministically. Keeping them flat and
enum-light keeps them compatible with Anthropic/OpenAI structured-output schemas.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClaimInput(BaseModel):
    """One row of claims.csv (the test input) plus resolved image paths."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    image_list: list[str] = Field(default_factory=list)  # split on ';'


class ConversationAnalysis(BaseModel):
    """What the user is actually claiming, extracted from the chat transcript.

    Note: the transcript may be in English, Hindi, or Hinglish — the extractor
    is responsible for normalising it to the allowed vocabulary.
    """

    asserted_issue_type: str = Field(default="unknown", description="Closest issue_type the user claims, or 'unknown'.")
    asserted_object_part: str = Field(default="unknown", description="Object part the user wants reviewed, or 'unknown'.")
    asserted_severity: Literal["minor", "moderate", "severe", "unspecified"] = "unspecified"
    parts_mentioned: list[str] = Field(default_factory=list)
    claim_summary: str = Field(default="", description="One short sentence: what is being claimed.")


class ImageFinding(BaseModel):
    """Objective inspection of a single image. Describes ONLY what is visible."""

    image_id: str = ""
    detected_object: Literal["car", "laptop", "package", "other", "unknown"] = "unknown"
    object_matches_claim: bool = False
    visible_parts: list[str] = Field(default_factory=list)
    shows_claimed_part: bool = False
    observed_issue_type: str = "unknown"
    observed_object_part: str = "unknown"
    observed_severity: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    damage_visible: bool = False
    # quality / authenticity / safety signals
    blurry: bool = False
    low_light_or_glare: bool = False
    cropped_or_obstructed: bool = False
    wrong_angle: bool = False
    possible_manipulation: bool = False
    non_original_image: bool = False
    contains_instruction_text: bool = False
    usable: bool = True
    note: str = Field(default="", description="<=20 words grounding the finding.")


class AdjudicationResult(BaseModel):
    """The decision fields an LLM adjudicator returns (Strategy B)."""

    claim_status: Literal["supported", "contradicted", "not_enough_information"] = "not_enough_information"
    issue_type: str = "unknown"
    object_part: str = "unknown"
    severity: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    supporting_image_ids: list[str] = Field(default_factory=list)
    justification: str = ""


class ClaimVerdict(BaseModel):
    """Final per-claim output. Serialises to one row of output.csv."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str

    def to_row(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": "true" if self.evidence_standard_met else "false",
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": "true" if self.valid_image else "false",
            "severity": self.severity,
        }


class UserHistory(BaseModel):
    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: str = "none"
    history_summary: str = ""


class EvidenceRule(BaseModel):
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str
