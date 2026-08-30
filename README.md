# TOFU IDK-Calibrated Causal Recovery

The active experiment lives in `pivot_experiment/`. Its final scientific and
implementation specification is `pivot_experiment/plan.md`.

The experiment localizes a residual recovery mechanism using the reversible
IDK `step-000025` adapter, constructs a `FULL - IDK` direction, and transfers
the frozen intervention to `gd_02` and the TOFU-withheld `RETAIN` reference.
It does not claim to determine whether memories are intact, erased or globally
inaccessible.

## Existing assets

- Frozen TOFU discovery, confirmation, reserve and `R_control` partitions.
- Pinned FULL, RETAIN and GD02 public model revisions.
- FULL, RETAIN and GD02 discovery scores.
- FULL and GD02 all-layer discovery activation caches.
- Archived IDK `step-000025` checkpoint, scores and all-layer activations.
- Historical IDK failures preserved under `pivot_experiment/archive/`.

The old P0/P1 reports remain historical results. The final experiment has no
scientific pass/fail gates; it uses immutable freeze artifacts and reports
positive, null or contrary outcomes.

## Install

```bash
.venv/bin/python -m pip install -e pivot_experiment
```

## Current status

Chunk 1 is complete. Revalidate the immutable four-state freeze without loading
a model:

```bash
.venv/bin/python pivot_experiment/scripts/freeze_final_states.py
```

Frozen result:

- IDK: archived `step-000025`, adapter SHA-256 verified;
- GD: pinned `gd_02`;
- discovery prompt alignment: 100/100 across all four states;
- FULL, IDK and GD02 caches: 50 valid two-row `[16, 2048]` shards each;
- mean IDK headroom: `0.1198518` nats/token, positive for 100/100 examples;
- confirmation: sealed;
- freeze hash:
  `0e8ba8a0f4816507da0f83cf98092d0419bd4b759bf53bb95574ae5f1e3e2d6c`.

## Chunk 2 — IDK-only layer localization

Validate the frozen workload without loading a tokenizer or model:

```bash
.venv/bin/python pivot_experiment/scripts/run_idk_localization.py --dry-run
```

Run the authoritative, resumable job manually:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_idk_localization.py \
  2>&1 | tee -a pivot_experiment/artifacts/logs/idk_localization.log
```

The job loads FULL plus IDK `step-000025` once, records same-runtime FULL/IDK
baselines, evaluates 100 discovery questions at all 16 layers, freezes `l*`
from IDK author-level fractional recovery only, and runs a four-author live
self-patch engineering audit. It does not read GD02 during selection and does
not open confirmation.

The 1,600 raw patch scores are already complete. Resuming after the archived-
baseline provenance failure will skip them: it adds only 200 baseline scores,
mechanically rebases the sweep, refreezes `l*`, and evaluates four audit rows.
Archived/current activation drift is retained as a diagnostic and is not used
as a hook-correctness test.

After it finishes, run the model-free completion audit:

```bash
.venv/bin/python pivot_experiment/scripts/check_stages.py --through B
```

The earlier `run_exact_patching.py` entry point is disabled because it implements
the superseded GD-selected-layer design.

## Chunk 3 — GD02/RETAIN patch transfer

Validate both frozen workloads without loading model weights:

```bash
.venv/bin/python pivot_experiment/scripts/run_patch_transfer.py --state gd02 --dry-run
.venv/bin/python pivot_experiment/scripts/run_patch_transfer.py --state retain --dry-run
```

Run the two independent, resumable jobs (they may be launched in separate
terminals if memory permits):

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_patch_transfer.py --state gd02 \
  2>&1 | tee -a pivot_experiment/artifacts/logs/gd02_patch_transfer.log
```

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_patch_transfer.py --state retain \
  2>&1 | tee -a pivot_experiment/artifacts/logs/retain_patch_transfer.log
```

Each job first writes 100 same-runtime receiver baselines, then 1,600 all-layer
matched-FULL patches. Neither job reads confirmation or changes the frozen
layer. Once both finish, produce and audit the model-free comparison:

```bash
.venv/bin/python pivot_experiment/scripts/finalize_patch_transfer.py
.venv/bin/python pivot_experiment/scripts/check_stages.py --through C
```

## Chunk 4 — FULL-IDK direction and alpha freeze

The model-free direction has been constructed from the 100 matched discovery
caches at frozen layer 14:

- direction norm: `6.796089`;
- direction freeze hash:
  `9b781bb673570c10e0a29ea71bd2316106a8518a36538e91374c7ca7054f42c9`.

Revalidate it and both workloads without loading weights:

```bash
.venv/bin/python pivot_experiment/scripts/build_steering_direction.py
.venv/bin/python pivot_experiment/scripts/run_alpha_steering.py --state idk --dry-run
.venv/bin/python pivot_experiment/scripts/run_alpha_steering.py --state retain --dry-run
```

Run the two resumable receiver jobs sequentially if memory is constrained:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_alpha_steering.py --state idk \
  2>&1 | tee -a pivot_experiment/artifacts/logs/idk_alpha_steering.log
```

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_alpha_steering.py --state retain \
  2>&1 | tee -a pivot_experiment/artifacts/logs/retain_alpha_steering.log
```

Each job writes 200 unsteered baselines and 600 steered scores for
`alpha = {0.5, 1, 2}` across discovery and `R_control`. GD02 and confirmation
are not read. After both finish, select alpha and seal confirmation without a
model:

```bash
.venv/bin/python pivot_experiment/scripts/finalize_alpha_selection.py
.venv/bin/python pivot_experiment/scripts/check_stages.py --through D
```

## Chunk 5 — sealed held-out confirmation

The pre-confirmation execution protocol and five norm-matched random
directions are frozen. Revalidate them with:

```bash
.venv/bin/python pivot_experiment/scripts/prepare_confirmation.py
```

Run FULL first to establish the shared held-out denominator, then run each
receiver. Each command is independently resumable:

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_confirmation.py --state full \
  2>&1 | tee -a pivot_experiment/artifacts/logs/confirmation_full.log
```

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_confirmation.py --state idk \
  2>&1 | tee -a pivot_experiment/artifacts/logs/confirmation_idk.log
```

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_confirmation.py --state gd02 \
  2>&1 | tee -a pivot_experiment/artifacts/logs/confirmation_gd02.log
```

```bash
caffeinate -dimsu env PYTHONUNBUFFERED=1 \
  .venv/bin/python pivot_experiment/scripts/run_confirmation.py --state retain \
  2>&1 | tee -a pivot_experiment/artifacts/logs/confirmation_retain.log
```

FULL writes 100 likelihood scores and 100 greedy generations. Each receiver
writes 700 likelihood scores (baseline, learned, five random controls) and 200
generations (baseline and learned). To separate the workloads, pass
`--phase scores` or `--phase generations`; a later `--phase all` safely resumes
whatever remains.

After all four states finish:

```bash
.venv/bin/python pivot_experiment/scripts/finalize_confirmation.py
.venv/bin/python pivot_experiment/scripts/check_stages.py --through E
```

## Chunk 6 — model-free final analysis

Generate all integrity-bound JSONL tables, PNG figures, the summary, and the
claim-bounded Markdown report without loading any model:

```bash
.venv/bin/python pivot_experiment/scripts/generate_final_report.py
.venv/bin/python pivot_experiment/scripts/check_stages.py --through F
```

Outputs are written under `pivot_experiment/artifacts/analysis/`. The report
distinguishes successful IDK calibration from the contrary GD02 transfer result
and preserves the component GD02 and RETAIN effects alongside the differential.
