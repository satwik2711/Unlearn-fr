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
