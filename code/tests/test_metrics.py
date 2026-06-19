"""Scoring math in evaluation/metrics.py."""

from evaluation.metrics import compute_metrics


def _row(status, risk="none", issue="dent", part="rear_bumper", sev="medium",
         valid="true", met="true", support="img_1"):
    return {
        "claim_status": status,
        "risk_flags": risk,
        "issue_type": issue,
        "object_part": part,
        "severity": sev,
        "valid_image": valid,
        "evidence_standard_met": met,
        "supporting_image_ids": support,
    }


def test_perfect_match_scores_one():
    pred = [_row("supported"), _row("contradicted")]
    exp = [_row("supported"), _row("contradicted")]
    m = compute_metrics(pred, exp)
    assert m["field_accuracy"]["claim_status"] == 1.0
    assert m["claim_status_macro_f1"] == 1.0


def test_claim_status_accuracy_half():
    pred = [_row("supported"), _row("supported")]
    exp = [_row("supported"), _row("contradicted")]
    m = compute_metrics(pred, exp)
    assert m["field_accuracy"]["claim_status"] == 0.5


def test_risk_flag_set_semantics_ignore_order_and_none():
    pred = [_row("supported", risk="user_history_risk;manual_review_required")]
    exp = [_row("supported", risk="manual_review_required;user_history_risk")]
    m = compute_metrics(pred, exp)
    assert m["risk_flags"]["f1"] == 1.0
    assert m["risk_flags"]["exact_set_match"] == 1.0


def test_risk_flag_partial_overlap():
    pred = [_row("supported", risk="blurry_image")]
    exp = [_row("supported", risk="blurry_image;user_history_risk")]
    m = compute_metrics(pred, exp)
    # 1 true positive, 0 fp, 1 fn -> precision 1.0, recall 0.5
    assert m["risk_flags"]["precision"] == 1.0
    assert m["risk_flags"]["recall"] == 0.5


def test_supporting_ids_set_f1():
    pred = [_row("supported", support="img_1;img_2")]
    exp = [_row("supported", support="img_1")]
    m = compute_metrics(pred, exp)
    # tp=1, fp=1, fn=0 -> precision .5, recall 1 -> f1 ~0.667
    assert m["supporting_image_ids_f1"] == round(2 * 0.5 * 1 / 1.5, 3)
