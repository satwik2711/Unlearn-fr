# TOFU Differential Causal Recoverability

The active experiment lives in `pivot_experiment/`. Its scientific and
implementation source of truth is `pivot_experiment/plan.md`.

The experiment now asks whether the utility-preserving `gd_02` Gradient
Difference state responds differently to causal restoration from `FULL` than
the TOFU-withheld `RETAIN` state. No tested GD checkpoint behavior-matched
`RETAIN`, so downstream results are explicitly exploratory and do not determine
whether a memory is intact or erased.

The historical `experiment/` directory is not used by this pipeline. The older
IDK-centered specification in `context/experiment_pivot.md` is superseded.

## Completed foundation

- Frozen TOFU author partitions and pinned public model revisions.
- Teacher-forced JSONL evaluator with no generation.
- One-pass FULL discovery score and all-layer `Q_END` cache.
- One-pass RETAIN discovery and `R_control` scores.
- P0 `FULL - RETAIN` gate: `PASS`.
- Two failed IDK calibrations retained as negative results:
  - `pivot_experiment/archive/idk_refusal_failed/gate_eval.json`
  - `pivot_experiment/archive/idk_suppression_failed/gate_eval.json`

The optional large local payload beneath each archive's ignored `data/`
directory is not part of Git.

## Install

```bash
.venv/bin/python -m pip install -e pivot_experiment
```

## Recheck completed P0

This reads stored artifacts and never loads a model:

```bash
.venv/bin/python pivot_experiment/scripts/check_gates.py --through P0
```

## Chunk 1 — GD screen (complete)

Inspect the exact authoritative workload without loading model weights:

```bash
.venv/bin/python pivot_experiment/scripts/evaluate_gd.py \
  --candidate gd_01 \
  --dry-run
```

Run the first frozen candidate manually:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/evaluate_gd.py \
  --candidate gd_01 \
  2>&1 | tee pivot_experiment/artifacts/logs/gd_01_evaluation.log
```

This single resumable run scores discovery and `R_control`, including the five
perturbed answers for each discovery question, and blindly caches all 16
discovery `Q_END` activations. It does not inspect confirmation or reserve
authors and does not generate text.

Check the GD behavior-match gate:

```bash
.venv/bin/python pivot_experiment/scripts/check_gates.py --through P1
```

All four frozen candidates were evaluated. The original behavior-match screen
failed. `gd_02` is now frozen for exploratory downstream work because it was the
only candidate to preserve `R_control`; it must not be described as
behavior-matched. Do not use the old IDK commands.

## Chunk 2 — exact differential patching

Validate all stored baselines, the `gd_02` freeze, and every FULL donor
activation without loading a model:

```bash
.venv/bin/python pivot_experiment/scripts/run_exact_patching.py \
  --phase matched \
  --dry-run
```

Run the resumable matched layer sweep manually:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_exact_patching.py \
  --phase matched \
  2>&1 | tee pivot_experiment/artifacts/logs/p2_matched.log
```

This computes 100 discovery questions × 16 layers × 2 receivers = 3,200
patched teacher-forced scores. It reuses the stored unpatched baselines and does
not open confirmation authors. Then run the model-free partial gate:

```bash
.venv/bin/python pivot_experiment/scripts/check_gates.py --through P2
```

If the layer sweep is complete, that check freezes `l*` and asks for the fixed
controls. Run them once:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_exact_patching.py \
  --phase controls \
  2>&1 | tee pivot_experiment/artifacts/logs/p2_controls.log

.venv/bin/python pivot_experiment/scripts/check_gates.py --through P2
```

The control phase adds four self-patches per receiver (one question from each
of four discovery authors) and one fixed
within-author mismatched FULL donor per discovery question. No full job was
started automatically.
