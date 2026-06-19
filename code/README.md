# Argus — Multi-Modal Evidence Review

> *Argus Panoptes, the all-seeing hundred-eyed guardian of Greek myth — many eyes
> (many images, many specialist agents) cooperating to reach one verdict.*

Argus verifies damage claims (`car`, `laptop`, `package`) by orchestrating a
team of specialist agents over the submitted **images** (the primary source of
truth), the **claim conversation**, the user's **claim history**, and the
**minimum evidence requirements**. For each row in `dataset/claims.csv` it emits
one row of `output.csv` with the exact schema in `problem_statement.md`.

It runs **offline with no API key** (deterministic `mock` provider) so the whole
pipeline, caching, and evaluation are reproducible; flip one env var for real
VLM grounding.

---

## Architecture — orchestrated agents

```
                 ┌──────────────────────────────────────────────┐
   claim row ───▶│                Orchestrator                   │
 (csv + images)  │  (concurrency, per-image cache, usage stats)  │
                 └───┬───────────┬───────────┬──────────┬────────┘
                     ▼           ▼           ▼          ▼
            ConversationAgent  VisionAgent  EvidenceAgent  RiskAgent
              (LLM, text)      (VLM, /image)  (rules)       (rules)
                     │           │           │          │
                     └───────────┴─────┬─────┴──────────┘
                                       ▼
                                  Adjudicator           ──▶  ClaimVerdict ──▶ output.csv
                              (rules | llm strategy)
```

- **ConversationAgent** — extracts the *asserted* claim (issue, part, severity)
  from the chat. Handles English / Hindi / Hinglish. (LLM; keyword fallback offline.)
- **VisionAgent** — inspects each image **objectively** (object, parts, visible
  damage, quality, authenticity, embedded-instruction text). Results are cached
  by image content hash. (VLM.)
- **EvidenceAgent** — applies `evidence_requirements.csv` to the findings to set
  `evidence_standard_met` / `valid_image`. (Deterministic.)
- **RiskAgent** — fuses image-quality, authenticity, claim-mismatch and
  user-history signals into `risk_flags`. History adds risk context but never
  overrides clear visual evidence on its own. (Deterministic.)
- **Adjudicator** — final `claim_status`, `issue_type`, `object_part`,
  `severity`, `supporting_image_ids`, justification. Two interchangeable
  strategies: `rules` (default, reproducible) and `llm`.

Why this split: the *visual reading* is the part only a VLM can do, so it is
isolated in one agent and cached; everything downstream is deterministic policy
that can be audited, tested, and reproduced exactly.

---

## Quickstart

```bash
pip install -r code/requirements.txt
cp code/.env.example code/.env      # optional; defaults to the offline mock provider

# Offline (no key) — produces output.csv for all of dataset/claims.csv:
python code/main.py

# Quick smoke test on 5 labeled rows:
python code/main.py --sample --limit 5

# Real VLM grounding:
export ANTHROPIC_API_KEY=sk-ant-...     # never commit this
ARGUS_PROVIDER=anthropic python code/main.py --output output.csv
```

`python code/main.py` writes `<repo>/output.csv` (override with `--output`).
`--sample` runs `dataset/sample_claims.csv` and writes `sample_output.csv`.

### Evaluation

```bash
python code/evaluation/main.py          # compares strategies, writes evaluation_metrics.md
```

See `code/evaluation/evaluation_report.md` for metrics, the strategy/model
comparison, and the operational analysis (calls, tokens, cost, latency, rate
limits, batching/caching/retry).

### Tests

```bash
pip install pytest
python -m pytest code/tests -q        # 47 tests, offline (no API key)
```

The deterministic core (vocab coercion, evidence/risk rules, the adjudication
decision tree, metrics, CSV/image I/O) is unit-tested, plus an end-to-end test
that runs the mock pipeline on the real samples and asserts every emitted value
is in the allowed vocabulary and that history-derived flags match
`user_history.csv`. Behaviour changes from here are made test-first.

---

## Configuration (all env-overridable; see `.env.example`)

| var | default | meaning |
|---|---|---|
| `ARGUS_PROVIDER` | `mock` | `mock` \| `anthropic` \| `openai` |
| `ARGUS_VISION_MODEL` | provider default | per-image VLM (`claude-opus-4-8`, `claude-haiku-4-5`, `gpt-4o`, …) |
| `ARGUS_REASONING_MODEL` | provider default | cheap text model for claim extraction |
| `ARGUS_ADJUDICATOR` | `rules` | `rules` \| `llm` |
| `ARGUS_MAX_IMAGE_EDGE` | `1024` | downscale long edge (image-token control) |
| `ARGUS_MAX_WORKERS` | `4` | concurrent claims (RPM control) |
| `ARGUS_VISION_CACHE` / `ARGUS_PROMPT_CACHE` | on | content-hash + prompt caching |

Secrets are read from the environment only — never hardcoded.

---

## Layout

```
code/
├── main.py                 # entry point: claims.csv -> output.csv
├── argus/
│   ├── config.py           # env-driven settings
│   ├── constants.py        # allowed vocab + issue->evidence-family mapping
│   ├── schemas.py          # pydantic models (structured agent I/O)
│   ├── data_io.py          # CSV + image-path I/O
│   ├── imaging.py          # load + downscale + hash images
│   ├── cache.py            # disk-backed per-image vision cache
│   ├── orchestrator.py     # wires the agents, concurrency, usage
│   ├── llm/                # provider-agnostic backends (mock|anthropic|openai)
│   └── agents/             # conversation, vision, evidence, risk, adjudicator
└── evaluation/
    ├── main.py             # strategy comparison on the labeled set
    ├── metrics.py          # scoring
    ├── evaluation_report.md# metrics + operational/cost analysis
    └── evaluation_metrics.md (generated)
```

No hardcoded per-case answers: the `mock` provider is a deterministic *stub*
(it parses inputs and applies the same rule layers as the real path), and all
decisions flow from the agent findings + the published rules.

## Packaging for submission

Zip the `code/` directory. Exclude `.env`, `.argus_cache/`, `__pycache__/`, and
any virtualenv. `output.csv` (predictions) and the chat transcript log are
uploaded separately.
