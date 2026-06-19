"""Orchestrator: run the multi-agent pipeline over a set of claims.

Per claim, in order:
  ConversationAgent -> VisionAgent (per image, cached) -> EvidenceAgent
  -> RiskAgent -> Adjudicator (rules | llm) -> ClaimVerdict

Claims run concurrently (ThreadPoolExecutor) for throughput; per-image vision
results are cached by content hash; usage is accumulated for the cost report.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from argus import data_io
from argus.agents import ConversationAgent, EvidenceAgent, RiskAgent, VisionAgent, adjudicator
from argus.cache import VisionCache
from argus.config import Settings
from argus.imaging import load_image_payload
from argus.llm import Usage, build_backend
from argus.schemas import ClaimInput, ClaimVerdict


class Orchestrator:
    def __init__(self, settings: Settings, verbose: bool = True) -> None:
        self.settings = settings
        self.verbose = verbose
        self.usage = Usage()
        self.backend = build_backend(settings, self.usage)

        self.cache = VisionCache(
            settings.cache_dir / "vision_cache.json", enabled=settings.use_vision_cache
        )
        self.history = data_io.load_user_history(settings.user_history_csv)
        self.evidence_rules = data_io.load_evidence_rules(settings.evidence_csv)

        self.conversation_agent = ConversationAgent(self.backend)
        self.vision_agent = VisionAgent(self.backend, self.cache, settings.vision_model)
        self.evidence_agent = EvidenceAgent(self.evidence_rules)
        self.risk_agent = RiskAgent()

    # -- single claim ---------------------------------------------------------
    def process(self, claim: ClaimInput) -> ClaimVerdict:
        conversation = self.conversation_agent.analyze(claim.user_claim, claim.claim_object)

        findings = []
        for rel in claim.image_list:
            payload = load_image_payload(
                data_io.resolve_image(self.settings.images_root, rel),
                max_edge=self.settings.max_image_edge,
                quality=self.settings.jpeg_quality,
            )
            findings.append(self.vision_agent.inspect(payload, claim.claim_object, conversation))

        evidence = self.evidence_agent.assess(conversation, findings, claim.claim_object)
        risk_str, risk_bools = self.risk_agent.assess(
            conversation, findings, self.history.get(claim.user_id), claim.claim_object
        )

        if self.settings.adjudicator == "llm":
            v = adjudicator.adjudicate_llm(
                self.backend, conversation, findings, evidence, risk_str, claim.claim_object
            )
        else:
            v = adjudicator.adjudicate_rules(
                conversation, findings, evidence, risk_bools, claim.claim_object
            )

        return ClaimVerdict(
            user_id=claim.user_id,
            image_paths=claim.image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met=evidence.met,
            evidence_standard_met_reason=evidence.reason,
            risk_flags=risk_str,
            issue_type=v.issue_type,
            object_part=v.object_part,
            claim_status=v.claim_status,
            claim_status_justification=v.justification,
            supporting_image_ids=v.supporting_image_ids,
            valid_image=evidence.valid_image,
            severity=v.severity,
        )

    # -- batch ----------------------------------------------------------------
    def run(self, claims: list[ClaimInput]) -> list[ClaimVerdict]:
        results: list[ClaimVerdict | None] = [None] * len(claims)
        workers = max(1, self.settings.max_workers)

        if workers == 1 or self.settings.provider == "mock":
            for i, claim in enumerate(claims):
                results[i] = self._safe(claim)
                self._tick(i + 1, len(claims))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._safe, c): i for i, c in enumerate(claims)}
                done = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
                    done += 1
                    self._tick(done, len(claims))

        self.cache.save()
        return [r for r in results if r is not None]

    def _safe(self, claim: ClaimInput) -> ClaimVerdict:
        try:
            return self.process(claim)
        except Exception as exc:  # never let one claim sink the batch
            return ClaimVerdict(
                user_id=claim.user_id,
                image_paths=claim.image_paths,
                user_claim=claim.user_claim,
                claim_object=claim.claim_object,
                evidence_standard_met=False,
                evidence_standard_met_reason=f"processing_error: {type(exc).__name__}",
                risk_flags="manual_review_required",
                issue_type="unknown",
                object_part="unknown",
                claim_status="not_enough_information",
                claim_status_justification="The claim could not be processed automatically.",
                supporting_image_ids="none",
                valid_image=False,
                severity="unknown",
            )

    def _tick(self, done: int, total: int) -> None:
        if self.verbose:
            print(f"\r  processed {done}/{total} claims", end="", file=sys.stderr, flush=True)
            if done == total:
                print("", file=sys.stderr)

    def usage_summary(self) -> dict:
        u = self.usage.as_dict()
        u["provider"] = self.backend.name
        u["vision_model"] = self.settings.vision_model
        u["reasoning_model"] = self.settings.reasoning_model
        u["adjudicator"] = self.settings.adjudicator
        return u
