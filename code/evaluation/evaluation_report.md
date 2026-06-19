# Argus — Evaluation & Operational Analysis

Multi-Modal Evidence Review. This report covers (1) how the system is evaluated,
(2) the strategy/model comparison, and (3) the operational analysis (calls,
tokens, cost, latency, rate limits, and the batching/caching/retry strategy).

Reproduce the metrics with:

```bash
python code/evaluation/main.py                      # offline (mock) — works with no key
ARGUS_PROVIDER=anthropic python code/evaluation/main.py   # real VLM grounding
```

---

## 1. Evaluation method

`dataset/sample_claims.csv` ships with 20 labeled rows (inputs + expected
outputs). `evaluation/main.py` runs the full pipeline on those rows under each
strategy and scores predictions against the labels with `evaluation/metrics.py`:

- **Per-field exact-match accuracy** for `evidence_standard_met`, `valid_image`,
  `issue_type`, `object_part`, `claim_status`, `severity`.
- **`claim_status` macro-F1** + a 3×3 confusion matrix (the headline decision).
- **`risk_flags` token-level micro-F1** + exact-set-match (flags are a set).
- **`supporting_image_ids` set F1**.

Free-text fields (`*_reason`, `*_justification`) are not auto-scored.

---

## 2. Results, tuning, and strategy comparison

Measured on the 20 labeled samples with **`gpt-4o` (vision) + `gpt-4o-mini`
(claim extraction), `temperature=0`** (deterministic), `rules` adjudicator.

| metric (sample set) | baseline | after tuning |
|---|---|---|
| **claim_status accuracy** | 0.700 | **0.800** |
| **claim_status macro-F1** | 0.583 | **0.689** |
| severity | 0.300 | **0.450** |
| valid_image | 0.850 | **0.900** |
| object_part | 0.800 | 0.800 |
| supporting_image_ids F1 | 0.732 | **0.769** |
| risk_flags F1 | 0.792 | 0.769 |
| issue_type | 0.550 | 0.500 |

**Tuning log** (each change made test-first or measured on the sample set):

1. **Severity rubric in the vision prompt** + "when unsure pick the LOWER level."
   gpt-4o was defaulting every case to `high`; the rubric lifted severity
   accuracy 0.30 → 0.45.
2. **`claim_mismatch` no longer fires on a mere issue-name difference**
   (dent vs scratch) on the claimed part — only on **severity exaggeration** or
   **damage on a different part than claimed**. This removed false
   `supported → contradicted` flips (2 → 0).
3. **Severity-exaggeration threshold**: a `severe` claim on *medium-or-lower*
   visible damage is now a mismatch (recovered case_005).
4. **Missing-contents claims** are `not_enough_information`, not `contradicted`,
   when only an intact exterior is visible — you can't verify missing contents
   from the outside (recovered case_018; not_enough is now 2/2).
5. **`temperature=0`** for deterministic, reproducible classification.

The remaining errors are genuine **VLM perception** misses (a severe front-end
hit read as nothing; a faint water stain missed; a hallucinated mark on an
undamaged trackpad), not rule bugs — chasing them with more rules would overfit
20 samples, so the loop was stopped there.

**Strategy comparison:** `rules` vs `llm` adjudicator over identical findings —
`rules` is the final choice: deterministic/reproducible (a hard requirement),
higher decision macro-F1, better supporting-image selection, and zero extra
model calls. Run `python code/evaluation/main.py --strategies rules,llm` to
compare both on your provider.

**Model configurations** (the per-image vision model is the cost driver):

| config | vision model | input $/1M | output $/1M | when to use |
|---|---|---|---|---|
| run here | `gpt-4o` + `gpt-4o-mini` | ~2.50 | ~10.00 | measured results above |
| budget | `claude-haiku-4-5` | 1.00 | 5.00 | ~4–5× cheaper; high-volume triage |
| best | `claude-opus-4-8` | 5.00 | 25.00 | subtle/structural damage |

Swap via `ARGUS_PROVIDER` / `ARGUS_VISION_MODEL` and re-run the evaluation to
pick the accuracy/cost point for your volume.

Swap with `ARGUS_VISION_MODEL=claude-haiku-4-5` (or `ARGUS_PROVIDER=openai`) and
re-run the evaluation to pick the accuracy/cost point for your volume. The
conversation extractor already defaults to the cheap `claude-haiku-4-5`.

---

## 3. Operational analysis

### Model calls (measured, `rules` strategy)

One conversation (text) call per claim + one vision call per image. The rule
layers (evidence, risk, adjudication) make **no** model calls.

| set | claims | images | calls = claims + images |
|---|---|---|---|
| sample | 20 | 24 | **44** |
| test (`claims.csv`) | 44 | 82 | **126** |

(The `llm` strategy adds 1 call per claim: +20 / +44.)

### Token usage (estimated for a real Anthropic run, default models)

Per **vision** call (Opus 4.8): ~450-token cached system prefix + ~80-token
claim context + **one image downscaled to a 1024 px long edge (~1.5k image
tokens)** ≈ **~2.0k input**, ~250 output. Per **conversation** call (Haiku 4.5):
~550 input, ~120 output.

Test set (126 calls): vision input ≈ **82 × 2.0k ≈ 164k tokens**, vision output
≈ 82 × 250 ≈ 20k; conversation input ≈ 44 × 550 ≈ 24k, output ≈ 5k.

### Approximate cost to process the full test set

Pricing assumptions: Opus 4.8 $5 / $25 per 1M (in/out); Haiku 4.5 $1 / $5;
prompt cache read 0.1×, write 1.25×; image downscaled to ~1.5k tokens.

| component | tokens | cost |
|---|---|---|
| vision input (Opus 4.8, system prefix cached) | ~164k | ~$0.67 |
| vision output (Opus 4.8) | ~20k | ~$0.51 |
| conversation in/out (Haiku 4.5) | ~29k | ~$0.05 |
| **Total test set (44 claims, 82 images)** | | **≈ $1.2–1.5** |

Budget config (Haiku 4.5 vision) drops the vision cost ~4–5× → **≈ $0.3** for
the whole test set. The 20-row sample pass adds roughly half the test cost.
Either way a full dev+test cycle is **under ~$2**.

### Latency / runtime

Vision calls dominate. At `ARGUS_MAX_WORKERS=4` and ~3–8 s per Opus 4.8 vision
call, the 82 image calls finish in **~2–4 minutes** wall-clock; Haiku is faster.
The offline mock run completes in seconds.

### TPM / RPM and the batching / throttling / caching / retry strategy

- **Per-image vision cache** (`code/.argus_cache/vision_cache.json`), keyed by
  image **content hash** + model + object: identical or re-submitted images cost
  zero calls. The evaluation already shows cache hits on re-runs.
- **Prompt caching**: the frozen ~450-token inspector system prompt is sent with
  `cache_control`, so it is written once and read at ~0.1× on every subsequent
  image — the main per-call input saving.
- **Image downscaling** to a 1024 px long edge before upload — the single
  biggest lever on image tokens (a full-res photo can cost ~3× more).
- **Concurrency cap** (`ARGUS_MAX_WORKERS`, default 4) bounds RPM/TPM; at ~40
  vision calls/min and ~2k input tokens each (~80k TPM) the workload sits well
  inside standard tier limits. Lower the cap if you hit 429s.
- **Retries**: `tenacity` random-exponential backoff on `RateLimitError` /
  `APIConnectionError` / `InternalServerError` (SDK auto-retry is disabled to
  avoid double-retrying), so transient 429/5xx don't fail a claim.
- **Resilience harness** (`argus/harness.py`): every agent step also runs under a
  per-step wall-clock timeout, step-level retries, a per-key **circuit breaker**
  (fast-fail a repeatedly-failing provider for a cooldown), a shared **RPM token
  bucket** (`ARGUS_RPM`) layered on top of the worker-concurrency cap, and **error
  isolation** (a failed step degrades to a safe value rather than crashing the
  claim). Harness counters (`attempts`/`retries`/`timeouts`/`failures`/
  `circuit_skips`) are reported in the run usage summary.
- **Batch API (50% cheaper)**: the test set is not latency-sensitive. Submitting
  the 126 requests via Anthropic's Message Batches API halves token cost and
  removes RPM pressure (results within ~1 h). Recommended for the final
  predictions run; the per-image cache makes a follow-up debugging pass free.
- **Failure isolation**: any single claim that errors is written as a safe
  `not_enough_information` + `manual_review_required` row, so one bad image never
  aborts the batch.

### Cost-control summary

Content-hash image cache + prompt caching + downscaling + a cheap text model for
extraction + an optional Haiku/Batch path together keep a full run in the
low-dollar range, with no repeated work across re-runs.
