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
  `54f4d4eb767e221e14afdb303b1bb8f403b13315a3153ee8a015bc09d4e56959`.

The earlier GD-selected-layer patching script remains obsolete and must not be
run. The next implementation step is Chunk 2: the IDK-only 16-layer patch sweep
and causal-layer freeze.

Runnable commands will be added here only after each new chunk is implemented
and validated.
