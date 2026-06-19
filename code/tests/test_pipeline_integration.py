"""End-to-end pipeline checks against the real dataset + the allowed-value spec.

Uses the offline mock provider, but exercises the full orchestration: CSV load,
real image decode, vision (mock), evidence/risk/adjudication rules, and output
serialisation. The vocab invariant comes straight from problem_statement.md.
"""

import pytest

from argus.config import Settings
from argus.constants import (
    CLAIM_STATUS,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    SEVERITIES,
)
from argus.data_io import load_sample, load_user_history
from argus.orchestrator import Orchestrator
from argus.schemas import ClaimInput


@pytest.fixture(scope="module")
def sample_run():
    settings = Settings()  # provider defaults to mock
    inputs, expected = load_sample(settings.sample_csv)
    orch = Orchestrator(settings, verbose=False)
    verdicts = orch.run(inputs)
    rows = [v.to_row() for v in verdicts]
    return inputs, expected, rows


def test_one_row_per_input(sample_run):
    inputs, _, rows = sample_run
    assert len(rows) == len(inputs) == 20


def test_every_field_uses_allowed_vocabulary(sample_run):
    inputs, _, rows = sample_run
    for claim, row in zip(inputs, rows):
        obj = claim.claim_object
        assert row["claim_status"] in CLAIM_STATUS
        assert row["issue_type"] in ISSUE_TYPES
        assert row["object_part"] in OBJECT_PARTS.get(obj, {"unknown"})
        assert row["severity"] in SEVERITIES
        assert row["evidence_standard_met"] in ("true", "false")
        assert row["valid_image"] in ("true", "false")
        for tok in row["risk_flags"].split(";"):
            assert tok in RISK_FLAGS, f"bad risk flag {tok!r}"
        # supporting_image_ids is 'none' or a ';'-list of non-empty ids
        sids = row["supporting_image_ids"]
        assert sids == "none" or all(t.strip() for t in sids.split(";"))


def test_history_derived_flags_match_user_history(sample_run):
    inputs, _, rows = sample_run
    hist = load_user_history(Settings().user_history_csv)
    for claim, row in zip(inputs, rows):
        flags = set(row["risk_flags"].split(";"))
        h = hist.get(claim.user_id)
        expect_uhr = bool(h and "user_history_risk" in h.history_flags)
        assert ("user_history_risk" in flags) == expect_uhr
        if "user_history_risk" in flags:
            # spec/sample invariant: history risk always escalates to manual review
            assert "manual_review_required" in flags


def test_no_image_claim_is_not_enough_information():
    settings = Settings()
    orch = Orchestrator(settings, verbose=False)
    claim = ClaimInput(
        user_id="ghost", image_paths="", user_claim="Customer: my car door has a dent.",
        claim_object="car", image_list=[],
    )
    v = orch.process(claim)
    assert v.claim_status == "not_enough_information"
    assert v.valid_image is False
    assert v.supporting_image_ids == "none"
