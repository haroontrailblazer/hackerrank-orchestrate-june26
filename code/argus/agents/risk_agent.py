"""RiskAgent (deterministic): combine image-quality, authenticity, claim-mismatch,
and user-history signals into the risk_flags column.

User history adds context but, per the spec, must not by itself override clear
visual evidence -- so history only contributes the user_history_risk /
manual_review_required flags; it never changes issue_type, object_part, or the
visual side of the decision.
"""

from __future__ import annotations

from argus.constants import RISK_FLAG_ORDER, SEVERITY_RANK
from argus.schemas import ConversationAnalysis, ImageFinding, UserHistory


class RiskAgent:
    def assess(
        self,
        conversation: ConversationAnalysis,
        findings: list[ImageFinding],
        history: UserHistory | None,
        claim_object: str,
    ) -> tuple[str, dict[str, bool]]:
        flags: set[str] = set()
        usable = [f for f in findings if f.usable]
        authentic = [f for f in usable if not f.non_original_image and not f.possible_manipulation]

        # --- image quality (from the VLM) ---
        if any(f.blurry for f in usable):
            flags.add("blurry_image")
        if any(f.cropped_or_obstructed for f in usable):
            flags.add("cropped_or_obstructed")
        if any(f.low_light_or_glare for f in usable):
            flags.add("low_light_or_glare")
        if any(f.wrong_angle for f in usable):
            flags.add("wrong_angle")

        # --- object / authenticity ---
        if usable and not any(f.object_matches_claim for f in usable):
            flags.add("wrong_object")
        if any(f.possible_manipulation for f in findings):
            flags.add("possible_manipulation")
        if any(f.non_original_image for f in findings):
            flags.add("non_original_image")
        if any(f.contains_instruction_text for f in findings):
            flags.add("text_instruction_present")

        # --- claimed damage visibility ---
        asserts_damage = conversation.asserted_issue_type not in ("none", "unknown")
        damage_visible_any = any(f.damage_visible for f in authentic)
        if asserts_damage and not damage_visible_any:
            flags.add("damage_not_visible")

        # --- claim mismatch (asserted vs visible) ---
        # A mere issue-name difference (e.g. dent vs scratch) on the claimed part
        # is NOT a contradiction -- damage to that part is still present, and the
        # VLM and customer often use different words for the same cosmetic mark.
        # A claim is contradicted only when the customer exaggerated severity, or
        # the visible damage is on a DIFFERENT part than claimed while the claimed
        # part itself shows nothing.
        max_obs_sev = max(
            (SEVERITY_RANK.get(f.observed_severity, 0) for f in authentic), default=0
        )
        claimed_part = conversation.asserted_object_part
        damage_on_claimed_part = any(f.damage_visible and f.shows_claimed_part for f in authentic)
        damage_on_other_part = any(f.damage_visible and not f.shows_claimed_part for f in authentic)
        # Severe claim but the visible damage is medium-or-lower = exaggeration.
        severity_mismatch = (
            conversation.asserted_severity == "severe"
            and damage_visible_any
            and max_obs_sev <= 2
        )
        part_mismatch = (
            claimed_part not in ("", "unknown")
            and damage_visible_any
            and not damage_on_claimed_part
            and damage_on_other_part
        )
        if severity_mismatch or part_mismatch:
            flags.add("claim_mismatch")

        # --- user history (context only) ---
        hist_flags = (history.history_flags if history else "none") or "none"
        user_history_risk = "user_history_risk" in hist_flags
        if user_history_risk:
            flags.add("user_history_risk")

        base_manual = "manual_review_required" in hist_flags
        manual_review = (
            base_manual
            or user_history_risk
            or "possible_manipulation" in flags
            or "non_original_image" in flags
            or "wrong_object" in flags
        )
        if manual_review:
            flags.add("manual_review_required")

        ordered = [f for f in RISK_FLAG_ORDER if f in flags]
        risk_str = ";".join(ordered) if ordered else "none"
        # Booleans some downstream logic (adjudicator) reuses.
        bools = {name: (name in flags) for name in RISK_FLAG_ORDER}
        return risk_str, bools
