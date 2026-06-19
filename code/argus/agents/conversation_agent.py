"""ConversationAgent: extract the actual damage claim from the chat transcript.

The transcript may be English, Hindi, or Hinglish (see sample_claims.csv), so an
LLM is the robust extractor; the mock backend falls back to keyword parsing.
Output is coerced to the allowed vocabulary before it leaves the agent.
"""

from __future__ import annotations

from argus.constants import OBJECT_PARTS, coerce_issue, coerce_part
from argus.llm.base import LLMBackend
from argus.schemas import ConversationAnalysis

_SYSTEM = """You extract the damage claim a customer is making from a short support \
chat. The chat may be in English, Hindi, or Hinglish; normalise it.

Return, using ONLY these allowed values:
- asserted_issue_type: one of [dent, scratch, crack, glass_shatter, broken_part, \
missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown]
- asserted_object_part: the single part the customer wants reviewed (object-specific); \
use 'unknown' if unclear
- asserted_severity: minor | moderate | severe | unspecified (how bad the CUSTOMER says it is)
- parts_mentioned: every part referenced
- claim_summary: one short sentence

Report only what the customer asserts. Do not judge whether it is true."""


class ConversationAgent:
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def analyze(self, user_claim: str, claim_object: str) -> ConversationAnalysis:
        allowed_parts = sorted(OBJECT_PARTS.get(claim_object, set()))
        user_text = (
            f"claim_object={claim_object}\n"
            f"Allowed parts for this object: {', '.join(allowed_parts)}\n\n"
            f"Chat transcript:\n{user_claim}"
        )
        result = self.backend.complete(
            system=_SYSTEM,
            user_text=user_text,
            response_model=ConversationAnalysis,
            task="conversation",
            max_tokens=400,
        )
        # Coerce to valid vocab.
        result.asserted_issue_type = coerce_issue(result.asserted_issue_type)
        result.asserted_object_part = coerce_part(claim_object, result.asserted_object_part)
        result.parts_mentioned = [coerce_part(claim_object, p) for p in result.parts_mentioned]
        return result
