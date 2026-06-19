"""Adjudicator: reach the final verdict from the structured findings.

Two interchangeable strategies (compared in evaluation/):
- "rules"  : deterministic decision tree over the agent outputs (default;
             reproducible and fully explainable).
- "llm"    : a single reasoning call that weighs the same evidence summary.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.constants import (
    SEVERITY_RANK,
    coerce_issue,
    coerce_part,
    coerce_severity,
)
from argus.agents.evidence_agent import EvidenceDecision
from argus.llm.base import LLMBackend
from argus.schemas import AdjudicationResult, ConversationAnalysis, ImageFinding


@dataclass
class Verdict:
    claim_status: str
    issue_type: str
    object_part: str
    severity: str
    supporting_image_ids: str
    justification: str


def _best_image(findings: list[ImageFinding]) -> ImageFinding | None:
    def score(f: ImageFinding) -> tuple:
        return (
            int(f.shows_claimed_part),
            int(f.damage_visible),
            SEVERITY_RANK.get(f.observed_severity, 0),
            int(not f.blurry),
            int(not f.cropped_or_obstructed),
        )

    return max(findings, key=score) if findings else None


def _supporting_ids(authentic_usable: list[ImageFinding], best: ImageFinding | None) -> str:
    ids = [f.image_id for f in authentic_usable if (f.shows_claimed_part or f.damage_visible) and not f.blurry]
    if not ids:
        ids = [f.image_id for f in authentic_usable if f.object_matches_claim and not f.blurry]
    if not ids and best is not None:
        ids = [best.image_id]
    # dedupe, preserve order
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return ";".join(out) if out else "none"


def adjudicate_rules(
    conversation: ConversationAnalysis,
    findings: list[ImageFinding],
    evidence: EvidenceDecision,
    risk_bools: dict[str, bool],
    claim_object: str,
) -> Verdict:
    asserted_part = conversation.asserted_object_part or "unknown"
    usable = [f for f in findings if f.usable]
    authentic = [f for f in usable if not f.non_original_image and not f.possible_manipulation]
    best = _best_image(authentic) or _best_image(usable)

    object_ok = any(f.object_matches_claim for f in usable)
    part_shown = any(f.shows_claimed_part for f in authentic)
    damage_visible = any(f.damage_visible for f in authentic)
    wrong_object = risk_bools.get("wrong_object", False)
    claim_mismatch = risk_bools.get("claim_mismatch", False)

    # A claim about missing/inner contents can't be verified OR refuted from an
    # intact exterior photo -- absence of exterior damage is not evidence either way.
    contents_claim = (
        conversation.asserted_issue_type == "missing_part"
        or asserted_part in ("contents", "item")
    )

    # --- decision tree ---
    if not usable:
        status = "not_enough_information"
    elif wrong_object and authentic:
        status = "contradicted"
    elif claim_mismatch:
        status = "contradicted"
    elif evidence.met and part_shown:
        if damage_visible:
            status = "supported"
        elif contents_claim:
            status = "not_enough_information"
        else:
            status = "contradicted"
    else:
        status = "not_enough_information"

    # --- derive fields ---
    if status == "not_enough_information":
        return Verdict(
            claim_status=status,
            issue_type="unknown",
            object_part=asserted_part,
            severity="unknown",
            supporting_image_ids="none",
            justification=_justify(status, conversation, claim_object, asserted_part, "unknown", risk_bools),
        )

    if status == "supported":
        issue = best.observed_issue_type if best and best.observed_issue_type not in ("none", "unknown") else conversation.asserted_issue_type
        part = best.observed_object_part if best and best.observed_object_part != "unknown" else asserted_part
        severity = best.observed_severity if best and best.observed_severity != "unknown" else "medium"
    else:  # contradicted
        if wrong_object:
            issue, part = "unknown", "unknown"
            severity = (best.observed_severity if best and best.observed_severity != "unknown" else "low")
        elif damage_visible and best and best.observed_issue_type not in ("none", "unknown"):
            issue = best.observed_issue_type
            part = best.observed_object_part if best.observed_object_part != "unknown" else asserted_part
            severity = best.observed_severity if best.observed_severity != "unknown" else "low"
        else:  # part visible, no damage
            issue, part, severity = "none", asserted_part, "none"

    issue = coerce_issue(issue)
    part = coerce_part(claim_object, part)
    severity = coerce_severity(severity)
    supporting = _supporting_ids(authentic, best)
    return Verdict(
        claim_status=status,
        issue_type=issue,
        object_part=part,
        severity=severity,
        supporting_image_ids=supporting,
        justification=_justify(status, conversation, claim_object, part, issue, risk_bools, supporting),
    )


def _justify(status, conversation, claim_object, part, issue, risk_bools, supporting="none") -> str:
    hist = " User history adds risk, so manual review is advised." if risk_bools.get("user_history_risk") else ""
    ids = "" if supporting == "none" else f" (see {supporting})"
    if status == "supported":
        base = f"The image evidence shows {issue} on the {part}, supporting the claim{ids}."
    elif status == "contradicted":
        if risk_bools.get("wrong_object"):
            base = f"The image shows a different object than the claimed {claim_object}, so the claim is contradicted{ids}."
        elif issue in ("none", ""):
            base = f"The {part} is visible but shows no claimed damage, contradicting the claim{ids}."
        else:
            base = f"The visible {issue} on the {part} does not match the claim, so it is contradicted{ids}."
    else:
        base = f"The {part} is not shown clearly enough in the images, so there is not enough information to decide."
    return (base + hist)[:400]


def adjudicate_llm(
    backend: LLMBackend,
    conversation: ConversationAnalysis,
    findings: list[ImageFinding],
    evidence: EvidenceDecision,
    risk_str: str,
    claim_object: str,
) -> Verdict:
    """Strategy B: let a reasoning model decide from the evidence summary."""
    lines = []
    for f in findings:
        lines.append(
            f"- image {f.image_id}: object={f.detected_object} matches={f.object_matches_claim} "
            f"shows_part={f.shows_claimed_part} issue={f.observed_issue_type} part={f.observed_object_part} "
            f"severity={f.observed_severity} damage_visible={f.damage_visible} "
            f"blurry={f.blurry} cropped={f.cropped_or_obstructed} non_original={f.non_original_image}"
        )
    summary = (
        f"claim_object={claim_object}; part={conversation.asserted_object_part}; "
        f"issue={conversation.asserted_issue_type}; severity={conversation.asserted_severity}\n"
        f"evidence_standard_met={evidence.met}; valid_image={evidence.valid_image}\n"
        f"risk_flags={risk_str}\nImage findings:\n" + "\n".join(lines)
    )
    system = (
        "You are the final adjudicator for a damage claim. Images are the primary truth; "
        "user history adds risk context but must not by itself override clear visual evidence. "
        "Decide claim_status (supported | contradicted | not_enough_information), the issue_type, "
        "object_part, severity (none|low|medium|high|unknown), the supporting_image_ids, and a "
        "one-sentence justification grounded in the images. Use 'not_enough_information' only when "
        "the relevant part/contents are not visible enough to judge."
    )
    res: AdjudicationResult = backend.complete(
        system=system,
        user_text=summary,
        response_model=AdjudicationResult,
        task="adjudication",
        max_tokens=400,
    )
    available = {f.image_id for f in findings if f.usable}
    ids = [i for i in res.supporting_image_ids if i in available]
    return Verdict(
        claim_status=res.claim_status if res.claim_status in {"supported", "contradicted", "not_enough_information"} else "not_enough_information",
        issue_type=coerce_issue(res.issue_type),
        object_part=coerce_part(claim_object, res.object_part),
        severity=coerce_severity(res.severity),
        supporting_image_ids=";".join(ids) if ids else "none",
        justification=(res.justification or "")[:400],
    )
