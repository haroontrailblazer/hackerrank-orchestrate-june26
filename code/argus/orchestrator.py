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
from argus.agents.evidence_agent import EvidenceDecision
from argus.cache import VisionCache
from argus.config import Settings
from argus.constants import RISK_FLAG_ORDER
from argus.harness import ResilienceHarness
from argus.imaging import load_image_payload
from argus.llm import Usage, build_backend
from argus.schemas import ClaimInput, ClaimVerdict, ConversationAnalysis, ImageFinding


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

        # Resilience/control layer wrapping every agent step.
        self.harness = ResilienceHarness(
            timeout=settings.step_timeout,
            retries=settings.step_retries,
            rpm=settings.rpm,
            circuit_threshold=settings.circuit_threshold,
            circuit_reset=settings.circuit_reset,
        )

    # -- single claim ---------------------------------------------------------
    def process(self, claim: ClaimInput) -> ClaimVerdict:
        h = self.harness

        # 1) Claim extraction (LLM/network) -> fallback to an empty analysis.
        conversation = h.run(
            "conversation",
            lambda: self.conversation_agent.analyze(claim.user_claim, claim.claim_object),
            lambda: ConversationAnalysis(),
            key="conversation",
        )

        # 2) Per-image inspection (VLM/network) -> failed image becomes unusable.
        findings = []
        for rel in claim.image_list:
            payload = load_image_payload(
                data_io.resolve_image(self.settings.images_root, rel),
                max_edge=self.settings.max_image_edge,
                quality=self.settings.jpeg_quality,
            )
            findings.append(
                h.run(
                    "vision",
                    lambda p=payload: self.vision_agent.inspect(p, claim.claim_object, conversation),
                    lambda p=payload: ImageFinding(
                        image_id=p.image_id, usable=False, detected_object="unknown",
                        note="vision step failed/timeout",
                    ),
                    key="vision",
                )
            )

        # 3) Evidence + 4) Risk (deterministic) -> error-isolated, no timeout/circuit.
        evidence = h.run(
            "evidence",
            lambda: self.evidence_agent.assess(conversation, findings, claim.claim_object),
            lambda: EvidenceDecision(False, "evidence step error; routed to manual review.", False),
            network=False,
        )
        risk_str, risk_bools = h.run(
            "risk",
            lambda: self.risk_agent.assess(
                conversation, findings, self.history.get(claim.user_id), claim.claim_object
            ),
            lambda: (
                "manual_review_required",
                {n: (n == "manual_review_required") for n in RISK_FLAG_ORDER},
            ),
            network=False,
        )

        # 5) Adjudication -> fallback verdict keeps the claimed part, defers decision.
        def _verdict_fallback():
            return adjudicator.Verdict(
                claim_status="not_enough_information",
                issue_type="unknown",
                object_part=conversation.asserted_object_part or "unknown",
                severity="unknown",
                supporting_image_ids="none",
                justification="Adjudication step failed; routed to manual review.",
            )

        if self.settings.adjudicator == "llm":
            v = h.run(
                "adjudicator",
                lambda: adjudicator.adjudicate_llm(
                    self.backend, conversation, findings, evidence, risk_str, claim.claim_object
                ),
                _verdict_fallback,
                key="adjudicator",
            )
        else:
            v = h.run(
                "adjudicator",
                lambda: adjudicator.adjudicate_rules(
                    conversation, findings, evidence, risk_bools, claim.claim_object
                ),
                _verdict_fallback,
                network=False,
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
        u["harness"] = self.harness.stats.as_dict()
        return u
