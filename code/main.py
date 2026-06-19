"""Argus -- terminal entry point (the evaluable contract entry point).

Reads dataset/claims.csv (or --sample for the labeled set) and writes output.csv
with the exact required schema. Runs offline with the deterministic mock provider
by default; set ARGUS_PROVIDER=anthropic (or openai) + a key for real predictions.

Examples
--------
  python code/main.py                         # all test claims -> output.csv (mock)
  python code/main.py --sample --limit 5      # quick smoke test on 5 sample rows
  ARGUS_PROVIDER=anthropic python code/main.py --output output.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python code/main.py` (add code/ to path) and load .env.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

from argus.config import Settings
from argus.data_io import load_claims, load_sample, write_output
from argus.orchestrator import Orchestrator


def build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if args.provider:
        settings.provider = args.provider
        # Clear any auto-resolved model ids so the new provider's defaults apply
        # (unless the user passed explicit model ids below).
        settings.vision_model = ""
        settings.reasoning_model = ""
    if args.adjudicator:
        settings.adjudicator = args.adjudicator
    if args.workers is not None:
        settings.max_workers = args.workers
    settings.__post_init__()  # re-resolve default models for the chosen provider
    if args.vision_model:
        settings.vision_model = args.vision_model
    if args.reasoning_model:
        settings.reasoning_model = args.reasoning_model
    return settings


def main() -> int:
    p = argparse.ArgumentParser(description="Argus multi-modal evidence review")
    p.add_argument("--provider", choices=["mock", "anthropic", "openai", "nvidia"])
    p.add_argument("--vision-model")
    p.add_argument("--reasoning-model")
    p.add_argument("--adjudicator", choices=["rules", "llm"])
    p.add_argument("--claims", help="path to input CSV (default: dataset/claims.csv)")
    p.add_argument("--output", help="output CSV path (default: <repo>/output.csv)")
    p.add_argument("--sample", action="store_true", help="run on dataset/sample_claims.csv")
    p.add_argument("--limit", type=int, help="process only the first N claims")
    p.add_argument("--workers", type=int)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    settings = build_settings(args)

    if args.sample:
        claims, _ = load_sample(settings.sample_csv)
        default_out = settings.repo_root / "sample_output.csv"
    else:
        claims = load_claims(Path(args.claims) if args.claims else settings.claims_csv)
        default_out = settings.repo_root / "output.csv"

    if args.limit:
        claims = claims[: args.limit]

    out_path = Path(args.output) if args.output else default_out

    print(
        f"Argus: {len(claims)} claims | provider={settings.provider} | "
        f"vision={settings.vision_model} | adjudicator={settings.adjudicator}",
        file=sys.stderr,
    )

    orch = Orchestrator(settings, verbose=not args.quiet)
    verdicts = orch.run(claims)
    write_output(out_path, [v.to_row() for v in verdicts])

    print(f"Wrote {len(verdicts)} rows -> {out_path}", file=sys.stderr)
    print("Usage: " + json.dumps(orch.usage_summary()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
