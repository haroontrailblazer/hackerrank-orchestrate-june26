"""CSV + image-path I/O. Pure stdlib csv so quoting matches the dataset exactly."""

from __future__ import annotations

import csv
from pathlib import Path

from argus.constants import OUTPUT_COLUMNS
from argus.schemas import ClaimInput, EvidenceRule, UserHistory


def _split_images(image_paths: str) -> list[str]:
    return [p.strip() for p in (image_paths or "").split(";") if p.strip()]


def load_claims(path: Path) -> list[ClaimInput]:
    rows: list[ClaimInput] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                ClaimInput(
                    user_id=r.get("user_id", ""),
                    image_paths=r.get("image_paths", ""),
                    user_claim=r.get("user_claim", ""),
                    claim_object=(r.get("claim_object", "") or "").strip().lower(),
                    image_list=_split_images(r.get("image_paths", "")),
                )
            )
    return rows


def load_sample(path: Path) -> tuple[list[ClaimInput], list[dict]]:
    """sample_claims.csv has inputs AND expected outputs. Returns (inputs, expected_rows)."""
    inputs: list[ClaimInput] = []
    expected: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            inputs.append(
                ClaimInput(
                    user_id=r.get("user_id", ""),
                    image_paths=r.get("image_paths", ""),
                    user_claim=r.get("user_claim", ""),
                    claim_object=(r.get("claim_object", "") or "").strip().lower(),
                    image_list=_split_images(r.get("image_paths", "")),
                )
            )
            expected.append({c: (r.get(c, "") or "") for c in OUTPUT_COLUMNS})
    return inputs, expected


def load_user_history(path: Path) -> dict[str, UserHistory]:
    out: dict[str, UserHistory] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):

            def _int(key: str) -> int:
                try:
                    return int(float(r.get(key, "0") or 0))
                except ValueError:
                    return 0

            uid = r.get("user_id", "")
            out[uid] = UserHistory(
                user_id=uid,
                past_claim_count=_int("past_claim_count"),
                accept_claim=_int("accept_claim"),
                manual_review_claim=_int("manual_review_claim"),
                rejected_claim=_int("rejected_claim"),
                last_90_days_claim_count=_int("last_90_days_claim_count"),
                history_flags=r.get("history_flags", "none") or "none",
                history_summary=r.get("history_summary", "") or "",
            )
    return out


def load_evidence_rules(path: Path) -> dict[tuple[str, str], str]:
    """(claim_object, applies_to) -> minimum_image_evidence. 'all' is expanded
    to every concrete object so lookups never miss the generic rules."""
    lookup: dict[tuple[str, str], str] = {}
    if not path.exists():
        return lookup
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rule = EvidenceRule(
                requirement_id=r.get("requirement_id", ""),
                claim_object=(r.get("claim_object", "") or "").strip().lower(),
                applies_to=(r.get("applies_to", "") or "").strip().lower(),
                minimum_image_evidence=r.get("minimum_image_evidence", "") or "",
            )
            objs = ["car", "laptop", "package"] if rule.claim_object == "all" else [rule.claim_object]
            for obj in objs:
                lookup[(obj, rule.applies_to)] = rule.minimum_image_evidence
    return lookup


def resolve_image(images_root: Path, rel: str) -> Path:
    return images_root / rel


def write_output(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in OUTPUT_COLUMNS})
