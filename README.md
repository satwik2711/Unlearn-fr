# TOFU Causal-Audit Pivot

The active implementation lives in `pivot_experiment/`. Its scientific source
of truth is `context/experiment_pivot.md`, and its execution plan is
`pivot_experiment/plan.md`.

The historical `experiment/` directory is not used by this pipeline.

## Install

From the repository root:

```bash
.venv/bin/python -m pip install -e pivot_experiment
```

## Prepare pinned data and frozen splits

This downloads the small TOFU dataset and verifies model configs/tokenizer
metadata, but does not download model weights:

```bash
.venv/bin/python pivot_experiment/scripts/prepare.py \
  --verify-model-metadata
```

## Inspect the public-evaluation workload

This performs no model loading or inference:

```bash
.venv/bin/python pivot_experiment/scripts/run_public_eval.py \
  --state both \
  --dry-run
```

## Run FULL manually

Teacher-forced scoring only. This also caches all 16 `Q_END` discovery
activations for later patching and CAA construction:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_public_eval.py \
  --state full 2>&1 | tee pivot_experiment/artifacts/logs/full_public.log
```

## Run RETAIN manually

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_public_eval.py \
  --state retain 2>&1 | tee pivot_experiment/artifacts/logs/retain_public.log
```

Both commands are resumable with identical arguments. Neither command performs
greedy generation or computes ROUGE-L.

## Check Gate P0

This reads completed JSONL artifacts only and never loads a model:

```bash
.venv/bin/python pivot_experiment/scripts/check_gates.py --through P0
```

P0 passes only when `FULL − RETAIN` is at least 0.15 nats/token and the
author-clustered 95% interval is entirely above zero.

## Archived refusal-only calibration

The original refusal-SFT adapter and its failed P1 evidence are preserved under
`pivot_experiment/archive/idk_refusal_failed/`. The active pipeline never loads
those adapters.

## Inspect the suppression-IDK training workload

This freezes 400 correct/refusal/retain triples and validates the schedule, but
does not load FULL or start training:

```bash
.venv/bin/python pivot_experiment/scripts/train_idk_suppression.py --dry-run
```

## Train the removable suppression-IDK LoRA manually

The fresh LoRA starts from the original frozen FULL checkpoint. Its loss combines
refusal SFT, a direct correct-below-refusal likelihood margin of 2.5 nats/token,
and paired retain SFT. It saves candidates at steps 3, 6, 10, 16, and 25:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/train_idk_suppression.py \
  2>&1 | tee pivot_experiment/artifacts/logs/idk_suppression_training.log
```

Training can resume from a completed candidate checkpoint. For example:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/train_idk_suppression.py \
  --resume pivot_experiment/artifacts/checkpoints/idk_suppression/suppression-step-000010 \
  2>&1 | tee -a pivot_experiment/artifacts/logs/idk_suppression_training.log
```

## Evaluate and select suppression-IDK manually

This loads FULL once, evaluates all saved adapters using discovery authors and
`R_control`, caches all-layer discovery activations, selects one adapter, and
performs the mandatory adapter-off/on/off audit:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/evaluate_idk_suppression.py \
  2>&1 | tee pivot_experiment/artifacts/logs/idk_suppression_evaluation.log
```

No confirmation or reserve author is evaluated.

## Check Gates P0–P1

This performs no model loading:

```bash
.venv/bin/python pivot_experiment/scripts/check_gates.py --through P1
```

P1 requires suppression relative to FULL, behavioral proximity to RETAIN,
at least a 2.0-nat refusal-over-correct margin, preserved `R_control`, exact
base-weight integrity, and numerical restoration of FULL after adapter removal.
