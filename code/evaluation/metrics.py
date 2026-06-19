"""Scoring of predicted output rows against the labeled sample_claims.csv."""

from __future__ import annotations

from collections import defaultdict

EXACT_FIELDS = [
    "evidence_standard_met",
    "valid_image",
    "issue_type",
    "object_part",
    "claim_status",
    "severity",
]
STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]


def _norm(v: str) -> str:
    return (v or "").strip().lower()


def _flag_set(v: str) -> set[str]:
    items = {p.strip().lower() for p in (v or "").split(";") if p.strip()}
    items.discard("none")
    return items


def _macro_f1(pred: list[str], exp: list[str], labels: list[str]) -> tuple[float, dict]:
    per = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for p, e in zip(pred, exp) if p == lab and e == lab)
        fp = sum(1 for p, e in zip(pred, exp) if p == lab and e != lab)
        fn = sum(1 for p, e in zip(pred, exp) if p != lab and e == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lab] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": tp + fn}
        f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per


def compute_metrics(predicted: list[dict], expected: list[dict]) -> dict:
    n = min(len(predicted), len(expected))
    predicted, expected = predicted[:n], expected[:n]

    field_acc = {}
    for f in EXACT_FIELDS:
        correct = sum(1 for p, e in zip(predicted, expected) if _norm(p.get(f)) == _norm(e.get(f)))
        field_acc[f] = round(correct / n, 3) if n else 0.0

    pred_status = [_norm(p.get("claim_status")) for p in predicted]
    exp_status = [_norm(e.get("claim_status")) for e in expected]
    # Macro-averages only over labels actually present in predictions or
    # expectations (sklearn 'macro' semantics): a label absent from both would
    # otherwise contribute F1=0 and make a perfect prediction score < 1.0.
    present = sorted(set(pred_status) | set(exp_status)) or STATUS_LABELS
    status_macro_f1, status_per = _macro_f1(pred_status, exp_status, present)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, e in zip(pred_status, exp_status):
        confusion[e][p] += 1

    # risk_flags micro F1 (token level)
    tp = fp = fn = exact = 0
    for p, e in zip(predicted, expected):
        ps, es = _flag_set(p.get("risk_flags")), _flag_set(e.get("risk_flags"))
        tp += len(ps & es)
        fp += len(ps - es)
        fn += len(es - ps)
        exact += int(ps == es)
    rprec = tp / (tp + fp) if (tp + fp) else 0.0
    rrec = tp / (tp + fn) if (tp + fn) else 0.0
    rf1 = 2 * rprec * rrec / (rprec + rrec) if (rprec + rrec) else 0.0

    # supporting_image_ids set F1
    stp = sfp = sfn = 0
    for p, e in zip(predicted, expected):
        ps, es = _flag_set(p.get("supporting_image_ids")), _flag_set(e.get("supporting_image_ids"))
        stp += len(ps & es)
        sfp += len(ps - es)
        sfn += len(es - ps)
    sprec = stp / (stp + sfp) if (stp + sfp) else 0.0
    srec = stp / (stp + sfn) if (stp + sfn) else 0.0
    sf1 = 2 * sprec * srec / (sprec + srec) if (sprec + srec) else 0.0

    return {
        "n": n,
        "field_accuracy": field_acc,
        "claim_status_macro_f1": round(status_macro_f1, 3),
        "claim_status_per_label": status_per,
        "claim_status_confusion": {k: dict(v) for k, v in confusion.items()},
        "risk_flags": {"precision": round(rprec, 3), "recall": round(rrec, 3), "f1": round(rf1, 3), "exact_set_match": round(exact / n, 3) if n else 0.0},
        "supporting_image_ids_f1": round(sf1, 3),
    }


def render_markdown(label_to_metrics: dict[str, dict], final_label: str) -> str:
    lines = ["# Argus evaluation metrics (sample_claims.csv)", ""]
    labels = list(label_to_metrics)
    # Field-accuracy comparison table
    lines.append("## Per-field accuracy")
    lines.append("")
    header = "| metric | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines += [header, sep]
    rows = EXACT_FIELDS + ["claim_status_macro_f1", "risk_flags_f1", "supporting_image_ids_f1"]
    for metric in rows:
        cells = []
        for lab in labels:
            m = label_to_metrics[lab]
            if metric in EXACT_FIELDS:
                cells.append(f"{m['field_accuracy'][metric]:.3f}")
            elif metric == "claim_status_macro_f1":
                cells.append(f"{m['claim_status_macro_f1']:.3f}")
            elif metric == "risk_flags_f1":
                cells.append(f"{m['risk_flags']['f1']:.3f}")
            else:
                cells.append(f"{m['supporting_image_ids_f1']:.3f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"**Final strategy chosen for output.csv:** `{final_label}`")
    lines.append("")
    lines.append("Confusion matrix (rows = expected, cols = predicted) for the final strategy:")
    lines.append("")
    conf = label_to_metrics[final_label]["claim_status_confusion"]
    lines.append("| expected \\ predicted | " + " | ".join(STATUS_LABELS) + " |")
    lines.append("|" + "---|" * (len(STATUS_LABELS) + 1))
    for exp_lab in STATUS_LABELS:
        row = conf.get(exp_lab, {})
        cells = [str(row.get(pl, 0)) for pl in STATUS_LABELS]
        lines.append(f"| {exp_lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
