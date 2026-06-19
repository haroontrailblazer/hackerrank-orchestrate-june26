"""Evaluation entry point.

Runs Argus on the labeled dataset/sample_claims.csv under (at least) two
strategies, scores each against the expected outputs, prints a comparison, and
writes evaluation/evaluation_metrics.md. Use this to choose the strategy before
producing predictions for dataset/claims.csv.

  python code/evaluation/main.py                 # compare rules vs llm adjudicator (mock ok)
  ARGUS_PROVIDER=anthropic python code/evaluation/main.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# code/evaluation/main.py -> add code/ to import path.
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(CODE_DIR / ".env")
except Exception:
    pass

from argus.config import Settings
from argus.data_io import load_sample
from argus.orchestrator import Orchestrator
from evaluation.metrics import compute_metrics, render_markdown


def run_strategy(label: str, adjudicator: str, provider: str | None, claims) -> tuple[list[dict], dict]:
    settings = Settings()
    if provider:
        settings.provider = provider
        settings.vision_model = ""
        settings.reasoning_model = ""
        settings.__post_init__()
    settings.adjudicator = adjudicator
    orch = Orchestrator(settings, verbose=False)
    verdicts = orch.run(claims)
    rows = [v.to_row() for v in verdicts]
    return rows, orch.usage_summary()


def main() -> int:
    p = argparse.ArgumentParser(description="Argus evaluation / strategy comparison")
    p.add_argument("--provider", choices=["mock", "anthropic", "openai", "nvidia"])
    p.add_argument("--final", default="rules", help="strategy label to mark as final (default: rules)")
    p.add_argument(
        "--strategies",
        default="rules,llm",
        help="comma list of adjudicators to compare (default: rules,llm). "
        "Use 'rules' alone during the improvement loop to halve model calls.",
    )
    args = p.parse_args()

    settings = Settings()
    claims, expected = load_sample(settings.sample_csv)
    print(f"Evaluating on {len(claims)} labeled sample claims...", file=sys.stderr)

    # >= 2 strategies compared on identical inputs (configurable for cost control).
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = [(s, s) for s in wanted]

    label_to_metrics: dict[str, dict] = {}
    usages: dict[str, dict] = {}
    for label, adj in strategies:
        rows, usage = run_strategy(label, adj, args.provider, claims)
        label_to_metrics[label] = compute_metrics(rows, expected)
        usages[label] = usage
        m = label_to_metrics[label]
        print(
            f"  [{label}] status_macro_f1={m['claim_status_macro_f1']:.3f} "
            f"status_acc={m['field_accuracy']['claim_status']:.3f} "
            f"risk_f1={m['risk_flags']['f1']:.3f}",
            file=sys.stderr,
        )

    final = args.final if args.final in label_to_metrics else "rules"
    md = render_markdown(label_to_metrics, final)
    out = Path(__file__).resolve().parent / "evaluation_metrics.md"
    out.write_text(md, encoding="utf-8")
    print(f"\nWrote metrics -> {out}", file=sys.stderr)
    print(json.dumps({"usages": usages}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
