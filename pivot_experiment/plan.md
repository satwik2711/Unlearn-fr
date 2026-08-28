# Pivot Experiment Implementation Plan

## 1. Purpose and source of truth

This directory implements the frozen scientific specification in
`context/experiment_pivot.md` from scratch. The old `experiment/` directory is
historical and must not be imported, modified, or used as an artifact source.

The implementation is optimized around four rules:

1. teacher-forced target log-likelihood is the primary metric everywhere;
2. greedy generation and ROUGE-L are disabled by default and are never a gate;
3. every reusable score or activation is stored during its first necessary pass;
4. gate checks read saved artifacts and never load a model or repeat inference.

No confirmation author may be scored, inspected, or used until the layer,
direction, sign, and steering coefficient have been frozen.

## 2. Frozen experiment

### Public assets

- Dataset: `locuslab/TOFU`
- Forget data: `forget10` and `forget10_perturbed`
- Retain data: `retain90`
- `FULL`: public OpenUnlearning Llama-3.2-1B-Instruct full checkpoint
- `RETAIN`: public OpenUnlearning Llama-3.2-1B-Instruct retain90 checkpoint
- `GD`: primary public GradDiff checkpoint, with frozen ordered fallbacks
- `IDK`: removable LoRA trained locally over frozen `FULL`

Exact model IDs and revisions will be recorded in configuration and the run
manifest before evaluation.

### Frozen author partitions

- Discovery: 5 forget authors, approximately 100 QA pairs
- `D_patch`: first 3 discovery authors, approximately 60 QA pairs
- Confirmation: 5 different forget authors, approximately 100 QA pairs
- Reserve: remaining 10 forget authors, approximately 200 QA pairs
- `R_control`: 5 retain authors, approximately 100 QA pairs

Authors are sorted, shuffled once with seed 42, and written to an immutable
split manifest. Splits are never reconstructed implicitly during later jobs.

### Metrics

The authoritative per-example metrics are:

- mean correct-answer token log-probability;
- correct-minus-perturbed-answer log-probability margin.

Correct and perturbed sequences should be placed in the same evaluation stream
and batched together where memory permits. They are teacher-forced passes, not
autoregressive generation.

ROUGE-L may be added later as an explicitly requested descriptive appendix. It
must use a separate command and output path so it cannot slow or contaminate the
main pipeline.

## 3. Planned directory structure

```text
pivot_experiment/
  plan.md
  README.md
  pyproject.toml

  configs/
    experiment.yaml
    models.yaml
    idk_lora.yaml

  src/pivot_experiment/
    __init__.py
    paths.py
    config.py
    manifest.py
    data.py
    models.py
    records.py
    metrics.py
    hooks.py
    gates.py
    statistics.py

  scripts/
    prepare.py
    run_public_eval.py
    check_gates.py
    train_idk.py
    evaluate_idk.py
    evaluate_gd.py
    run_patching.py
    build_direction.py
    select_alpha.py
    run_confirmation.py
    analyze.py

  artifacts/
    manifests/
    splits/
    scores/
    activations/
    checkpoints/
    gates/
    results/
    figures/
    logs/
```

Only source, configuration, documentation, and small manifests belong in Git.
Downloaded models, checkpoints, activations, logs, and generated results remain
ignored.

## 4. Artifact contract

### Per-example results

JSONL is the authoritative format. Every record has enough provenance to be
joined or audited without rerunning a model:

```text
run_id
model_state
model_id
model_revision
adapter_id
split
author_id
example_id
answer_variant
intervention
layer
alpha
direction_seed
mean_target_logprob
token_count
```

Correct and perturbed results may be stored as separate rows or one structured
record, but the schema must be frozen before the first real run. CSV files are
derived views only; plots and gates read JSONL.

### Activations

Activations are tensors, so they are stored in `safetensors`, not JSONL. Each
shard has a JSON index mapping tensor rows to `author_id`, `example_id`, layer,
token position, dtype, model revision, and prompt hash.

Only the `Q_END` residual state is cached. Full token-by-token hidden states are
not stored.

### Resumability and integrity

- Jobs write append-safe shards and an atomic completion manifest.
- A completed shard is never silently overwritten.
- Unique record keys prevent duplicate examples after resume.
- Every consumer validates split hash, prompt hash, tokenizer revision, model
  revision, and scoring version.
- Gate results contain the hashes of all input artifacts.

## 5. Phase 1 — structure, dependencies, models, and data

This is the first implementation milestone.

### Build

1. Create the package and directory structure above.
2. Add pinned runtime dependencies without modifying the historical environment.
3. Implement centralized paths and typed configuration loading.
4. Implement the run manifest and deterministic seed setup.
5. Download/load TOFU and verify required fields and counts.
6. Derive the author identifier and freeze all author splits.
7. Implement a single model loader for `FULL`, `RETAIN`, LoRA-backed `IDK`, and
   public `GD` states.
8. Verify tokenizer identity, architecture, layer count, chat template, dtype,
   and `Q_END` indexing across public checkpoints.
9. Add tiny smoke tests that do not open confirmation data for evaluation.

Dataset preparation may identify confirmation rows and write their IDs into the
frozen split manifest. It must not send their text through a model before the
confirmation stage.

### Phase 1 acceptance criteria

- one command prepares assets and manifests idempotently;
- the frozen split file is deterministic under seed 42;
- no author appears in more than one forget partition;
- `R_control` contains only retain90 authors;
- all public checkpoints share the required tokenizer and 16-layer hook layout;
- a one-example discovery smoke test produces a valid teacher-forced score;
- no real experimental score is treated as complete during smoke testing.

## 6. Phase 2 — common evaluator and public-model run

### Common evaluator

Implement one batched evaluator used by every later script. It must:

- apply one frozen chat template;
- mask prompt tokens from answer loss;
- compute mean answer-token log-probability;
- score correct and perturbed answers consistently;
- optionally attach `Q_END` hooks;
- write per-example JSONL incrementally;
- run under `model.eval()` and `torch.no_grad()` or inference mode;
- never call `generate()` unless an explicit descriptive-generation flag is used.

### First authoritative public run

Run `FULL`, then `RETAIN`, grouped by model so each checkpoint is loaded once.

During the first `FULL` discovery pass:

- score correct and perturbed answers for all `D_CAA` rows;
- score `R_control`;
- cache all 16 layers of `Q_END` for all `D_CAA` rows.

The first three discovery authors in this cache are the `D_patch` source. The
same cache later supplies the `FULL` half of the CAA direction. `FULL` must not be
rerun for either purpose.

During the first `RETAIN` discovery pass:

- score correct and perturbed answers for all `D_CAA` rows;
- score `R_control`;
- store no activations, because none are needed before steering.

Do not score confirmation or reserve authors in this phase.

## 7. One read-only gate script

`scripts/check_gates.py` is the only gate entry point. It accepts `--through P0`
through `--through P5` and evaluates only gates whose required artifacts exist.

It must:

- load JSONL, manifests, and summaries only;
- never import Transformers or load a model;
- never perform inference or training;
- validate provenance and author separation before metrics;
- print a compact PASS / FAIL / BLOCKED table;
- write a machine-readable result to `artifacts/gates/`;
- return nonzero for FAIL or malformed/missing inputs;
- preserve a valid negative result when a scientific gate fails.

Gate definitions:

- `P0`: `FULL − RETAIN` discovery forget log-probability is at least 0.15
  nats/token and its author-clustered 95% interval is above zero.
- `P1`: selected IDK is suppressed, sufficiently matched to RETAIN, preserves
  `R_control`, and adapter removal restores FULL.
- `P2`: one frozen GD candidate is reasonably matched to RETAIN without
  catastrophic `R_control` loss.
- `P3`: at least one exact layer patch gives meaningful positive recovery across
  more than one discovery author.
- `P4`: the frozen learned direction restores held-out IDK more than RETAIN and
  more than norm-matched random directions.
- `P5`: report the frozen GD-versus-RETAIN recovery contrast only after P4 passes.

Thresholds not already frozen in the scientific pivot must be specified before
the producing run, never chosen after inspecting confirmation results.

## 8. Phase 3 — removable IDK training

Proceed only after `P0` passes.

1. Freeze every `FULL` base parameter.
2. Train the configured LoRA on all 400 forget questions with frozen refusal
   targets and a fixed paired retain sample.
3. Save several adapter checkpoints and the training trajectory.
4. Evaluate each checkpoint on discovery forget rows and `R_control` using the
   common evaluator.
5. Cache all-layer discovery `Q_END` activations during each checkpoint's scoring
   pass. These caches are small and prevent a later selected-IDK rerun.
6. Select the checkpoint using discovery metrics only.
7. Verify base-parameter hashes are unchanged.
8. In one process, test adapter off, on, and off again on a frozen audit subset;
   compare both off states with stored `FULL` scores within numerical tolerance.
9. Record the selected adapter and selected activation-shard IDs.

Run `check_gates.py --through P1`. If P1 fails, stop and report the calibrated
negative result rather than tuning on confirmation authors.

## 9. Phase 4 — public GD matching

1. Evaluate the primary GD checkpoint on discovery forget rows and `R_control`.
2. Run `P2` from saved metrics.
3. Only if it fails, evaluate the next candidate in the frozen order.
4. Freeze the first acceptable candidate, or the nearest allowed Pareto point if
   none passes, with the mismatch explicitly recorded.

No GD activations and no confirmation examples are inspected during selection.

## 10. Phase 5 — patching and steering implementation

The detailed code is implemented only after P0–P2 artifacts and the selected IDK
state exist, but the scientific sequence is already frozen.

### Exact layer patching

- Reuse cached `FULL` all-layer `Q_END` activations for `D_patch`.
- For each of 16 layers, run selected IDK while replacing only that layer's
  `Q_END` residual state with the paired FULL value.
- Reuse the stored unpatched IDK baseline; do not recompute it per layer.
- Run the same-state hook control on one small frozen subset.
- Save every per-example effect and freeze the best layer from discovery only.
- Run gate P3 before constructing or interpreting a steering direction.

The 16 patched IDK forward conditions are unavoidable causal interventions. They
are teacher-forced and require no generation.

### CAA direction

- Read the frozen layer from P3.
- Reuse cached `FULL` and selected-IDK activations on all `D_CAA` rows.
- Compute the raw mean `FULL − IDK` vector without another model pass.
- Store the raw/unit vectors, norm, author vectors, cosine diagnostics, source
  example IDs, and artifact hashes.

### Alpha selection

- Test only alpha 0.5, 1.0, and 2.0 on discovery authors.
- Group conditions by model so each model is loaded once per job.
- Reuse saved unsteered baselines; run only the steered conditions.
- Define the utility term explicitly as the selected IDK model steered on
  `R_control` relative to the same unsteered IDK state.
- Freeze alpha before opening confirmation authors.

### Confirmation

After layer, vector, sign, and alpha are frozen:

- score confirmation baselines for `FULL`, `IDK`, `RETAIN`, and `GD`;
- apply the learned direction to all four states;
- apply five norm-matched random directions to IDK, RETAIN, and GD;
- group all conditions by loaded model;
- write results incrementally so analysis never requires another forward pass.

Run P4 and P5 from saved confirmation artifacts. Reserve authors remain untouched
unless the frozen optional validation is explicitly started.

## 11. Run minimization ledger

The following work must be reused rather than repeated:

| Artifact produced once | Reused for |
|---|---|
| FULL discovery scores | P0, IDK matching reference, behavioral figure |
| RETAIN discovery scores | P0, IDK/GD matching, behavioral figure |
| FULL `D_CAA` all-layer cache | `D_patch` source and FULL half of CAA |
| selected-IDK discovery scores | P1, patching baseline, alpha baseline |
| selected-IDK all-layer cache | IDK half of CAA |
| GD discovery scores | P2 and behavioral figure |
| unsteered discovery scores | every alpha delta |
| confirmation baselines | every learned/random intervention delta |

Permitted repeated forward conditions are limited to work that changes the model
computation: candidate checkpoints, 16 layer patches, three frozen alphas, the
learned steering intervention, and five random-direction controls.

## 12. Execution checkpoints

We will implement and review the project in this order:

1. **Scaffold and loading:** package, configs, manifests, data, splits, models.
2. **Evaluation:** common teacher-forced evaluator and activation writer.
3. **Public run:** FULL/RETAIN discovery artifacts produced once.
4. **Gate system:** one read-only gate script; formally evaluate P0.
5. **IDK:** training, checkpoint scoring, activation caching, reversibility, P1.
6. **GD:** ordered candidate matching and P2.
7. **Patching:** exact 16-layer experiment and P3.
8. **Steering:** cached CAA construction and discovery-only alpha selection.
9. **Confirmation:** frozen learned/random interventions, P4, and P5.
10. **Analysis:** statistics, figures, findings, and valid negative-result report.

No later phase starts merely because a process completed; its preceding gate must
pass formally from intact saved artifacts.
