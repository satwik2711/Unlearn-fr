# Tonight's Execution Plan: Causal Audit of Apparent Forgetting

**Status:** preflight complete; experiment not started  
**Date prepared:** 2026-08-26 (Asia/Kolkata)  
**Scientific specification:** `context/experiments.md`  
**Research context:** `context/research.md`  
**Execution target:** complete the frozen experiment, its required controls, statistics, figures, findings, and executive summary without tuning on confirmation authors

## 1. Objective and scope

Run the complete experiment defined in `context/experiments.md`:

1. construct and audit `TOFU-Alias`;
2. establish `BASE`, `FULL`, `RETAIN`, `IDK`, and `GD`;
3. behaviorally match the suppressed/unlearned states to `RETAIN` using discovery authors only;
4. learn a `FULL - IDK` residual-stream recovery direction using discovery authors only;
5. freeze the layer, coefficient, direction, normalization, and checkpoint choices;
6. run the confirmation authors once;
7. produce all required metrics, controls, intervals, figures, and claim-limited reports.

“Complete the experiment” means the frozen main experiment and mandatory controls. Items explicitly marked optional or out of scope in `experiments.md` remain optional and must not delay the confirmatory result.

No FT/KD study, second dataset, second scientific model, second unlearning method, SAE/crosscoder analysis, multi-seed replication, or training-time control intervention will be added tonight.

## 2. Authorization boundary for the execution agent

For the later execution run, the agent is authorized to perform the following without pausing for routine confirmation:

- create and edit files under this repository;
- create a repository-local virtual environment and install required Python packages into it;
- download public TOFU data and the missing public Qwen configuration/tokenizer files;
- arrange the already-downloaded model weights into complete local model directories without modifying their contents;
- run CPU, MPS, and filesystem diagnostics;
- run training, evaluation, activation extraction, steering, statistics, and plotting jobs;
- create checkpoints, caches, logs, metrics, figures, reports, and manifests under the experiment directory;
- use `caffeinate` to keep the Mac awake;
- resume failed or interrupted jobs from the latest validated checkpoint;
- adapt implementation details necessary for Qwen 3.5 and Apple MPS while preserving the scientific controls and recording every deviation.

The agent is not authorized to:

- push commits, open pull requests, or modify remote repositories;
- purchase cloud compute or start a paid external service;
- delete or overwrite user-authored context files or unrelated work;
- expose credentials or tokens in logs;
- silently change scientific thresholds, confirmation data, or required controls;
- claim successful forgetting, retained intact memories, or general mechanistic validity beyond the interpretation matrix.

If a scientific gate fails, the agent should continue only along the failure path explicitly permitted by `experiments.md`, produce the corresponding negative result, and avoid repeatedly asking the user how to proceed.

## 3. Access and environment audit

### Codex access

The current planning session reports:

- filesystem: unrestricted;
- network: enabled;
- approval policy: `never`;
- project trust: `/Users/satwikpandey/Dev/unlearn-fr` is marked `trusted` in the user Codex configuration;
- project-local Codex config: none present.

This is the desired no-prompt execution profile. Official Codex configuration supports `approval_policy = "never"` for non-interactive operation and `sandbox_mode = "danger-full-access"` (or the `:danger-full-access` permission profile) for unrestricted command execution.

Important: the user-level `~/.codex/config.toml` currently records project trust but does not persist `approval_policy` or `sandbox_mode`. The present run received unrestricted access from its host/session profile. A newly started agent must therefore visibly show **Full Access** (filesystem/network unrestricted) and a no-approval/never policy at launch. Do not assume the current session grant automatically transfers to a new session.

Even with that profile, operating-system prompts, authentication failures, provider outages, insufficient RAM, unsupported MPS operations, or exhausted disk can still block work. Those are runtime blockers rather than Codex approval prompts.

Reference: [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

### Machine

- MacBook Air, Apple M2, 8-core integrated GPU;
- 16 GiB unified memory;
- macOS 26.6.1;
- approximately 226 GiB free disk at preflight;
- connected to AC power;
- `caffeinate` is available;
- no CUDA/NVIDIA runtime;
- no `tmux` detected.

The experiment must use Apple MPS where supported and execute one memory-intensive model job at a time. Parallelize lightweight data/statistics work only; do not load multiple 2B model states concurrently.

### Local model weights

Present:

| Local file | Size | Verified official SHA-256 |
|---|---:|---|
| `models/qwen3.5-0.8b.safetensors` | 1.6 GiB | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `models/qwn3.5-2b.safetensors` | 4.2 GiB | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |

Both hashes exactly match the official Qwen repositories. The 2B filename contains `qwn` rather than `qwen`; treat that as a local filename only and record it exactly in provenance.

Missing locally:

- `config.json`;
- `model.safetensors.index.json`;
- tokenizer files;
- chat template;
- processor/preprocessor metadata;
- complete Hugging Face directory structure.

At execution start, fetch the missing small files at frozen revisions and construct complete local model directories. Reuse the existing large weights via a validated symlink or other non-rewriting arrangement. Recompute the weight hash after arrangement. Do not redownload 5.8 GiB of weights unless validation shows the existing files are unusable.

Frozen upstream revisions observed at preflight:

- `Qwen/Qwen3.5-2B`: `15852e8c16360a2fea060d615a32b45270f8a8fc`;
- `Qwen/Qwen3.5-0.8B`: `2fc06364715b967f1860aea9cf38778875588b17`;
- `locuslab/TOFU`: `324592d84ae4f482ac7249b9285c2ecdb53e3a68`.

Record these in `run_manifest.json`; if an exact revision cannot be fetched, stop before training and report the mismatch.

### Network and authentication

- Hugging Face model and TOFU dataset endpoints returned HTTP 200;
- the Qwen repositories and TOFU are public and ungated;
- Hugging Face CLI authentication is active for user `ErgoEdge`;
- approximately 226 GiB free disk is sufficient for dependencies, data, adapters, caches, merged model copies, metrics, and figures if checkpoints are managed deliberately.

No external app/plugin is required.

### Python environment

The repository currently has no project environment, dependency file, or installed ML stack in the active Python 3.13 environment. In particular, `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `safetensors`, NumPy, pandas, SciPy, scikit-learn, ROUGE, and plotting packages are absent.

This is the first practical setup task. Use `uv` to create an isolated repository-local environment with a Python version supported by PyTorch and the required Qwen 3.5 Transformers implementation. Pin every resolved version in a lockfile and record it in the manifest. Do not install into the system interpreter and do not require `sudo`.

### Repository state

The worktree is already dirty before experiment implementation:

- tracked deletion: `context/research (2).md`;
- untracked: `context/research.md`;
- untracked: `context/.DS_Store`.

These are user-owned changes. Preserve them. Do not restore, delete, stage, or commit them. Add experiment outputs without treating the pre-existing state as agent-created.

## 4. Feasibility decision

Full-parameter training of the 2B model is not a safe plan on 16 GiB unified memory. BF16 weights alone occupy about 4.2 GiB; gradients, optimizer states, activations, framework overhead, and the hybrid architecture would exceed the practical memory envelope.

Use the LoRA route already permitted by `experiments.md`:

- train `FULL` and `RETAIN` with the identical LoRA configuration;
- target verified language projection modules across both linear-attention/DeltaNet and full-attention layers;
- merge each adapter into a separate local model state before cross-state activation comparison if memory and disk permit;
- keep `IDK` as a separate removable refusal LoRA over frozen `FULL`;
- use an adapter-mediated `GD` state if full-weight GD is infeasible, and state explicitly that this does not test irreversible weight-level erasure.

Do not infer that LoRA support is available merely because PEFT imports. Qwen 3.5 module targeting, MPS backward support, BF16 behavior, gradient checkpointing, adapter merge, residual hooks, and activation replacement must all pass Gate 0.

The 16-hour target is ambitious on this laptop and is not guaranteed by access alone. Before the 2B scientific run, benchmark representative 0.8B and 2B forward/backward steps and forecast each stage from measured throughput. Preserve scientific gates and controls if the estimate exceeds the nominal budget; reduce epochs only after examining acquisition curves, as specified in `experiments.md`.

## 5. Execution principles

1. **Provenance before training.** Freeze upstream revisions, hashes, alias map, data splits, formatter, seeds, model class, dependency versions, and configuration first.
2. **One formatter and scorer.** All states use the same chat template, answer-token mask, tokenization, and mean answer log-probability implementation.
3. **Discovery/confirmation firewall.** Confirmation authors may be transformed and trained on where the model-state construction requires it, but they must never influence checkpoint selection, layer, coefficient, sign, normalization, stopping rule, or code debugging based on outcomes.
4. **Gate-driven execution.** A failed prerequisite changes the permitted claim; it does not justify weakening the gate silently.
5. **Resume by construction.** Every long stage writes atomic checkpoints, manifests, logs, and completion markers so a process or agent interruption does not restart the night.
6. **One heavy job at a time.** Avoid memory pressure and thermal throttling on the fanless M2.
7. **Absolute recovery plus contrast.** A positive `IDK - RETAIN` or `GD - RETAIN` contrast must not be called recovery if it is produced mainly by harming `RETAIN`. Report each model's absolute `Delta s` as well as the contrast.
8. **No result invention.** Missing, failed, or incomplete measurements remain explicitly missing, failed, or incomplete.

## 6. Ordered execution plan

### Phase 0 — bootstrap and durable logging

1. Capture start time, git status, machine details, disk, memory, power state, and current access profile in a preflight artifact.
2. Start the orchestration shell under `caffeinate`.
3. Create `experiment/logs/` and a timestamped master log.
4. Create a repository-local `uv` environment and install the pinned stack.
5. Validate imports and print package versions without exposing environment-variable values.
6. Download only the missing Qwen ancillary files at the frozen revisions.
7. Construct complete local 0.8B and 2B directories, verify every expected file, and recheck weight hashes.
8. Create the initial `run_manifest.json` before any model mutation.

Exit condition: a fresh process can load the frozen configuration, tokenizer, and weights using only paths recorded in the manifest.

### Phase 1 — data acquisition and immutable audit

1. Download TOFU at the frozen revision into the local cache.
2. Inspect the actual schema of every required split before writing transformations.
3. Derive stable author/entity IDs and canonical row IDs.
4. Generate the 200 nonce aliases with seed 42 and apply them consistently to every identity-bearing field.
5. Freeze and hash the alias map and transformed dataset.
6. Verify split row counts, author counts, no forget/retain entity overlap, and the `forget10 + retain90 = full` invariant.
7. Measure duplicates and token-length distributions.
8. Freeze `F_discovery`, `F_confirmation`, `R_control`, real-author, world-fact, and optional compatible holdout subsets.
9. Write `data_audit.json` and `frozen_splits.json` atomically.

No split or alias changes are allowed after the first model-training step.

### Phase 2 — Gate 0 infrastructure smoke test

Use the 0.8B model for no more than 30 minutes unless diagnosing a concrete blocking failure:

1. text-only load without unnecessary vision inputs;
2. Qwen chat formatting and assistant-only labels;
3. one MPS forward/backward optimizer step;
4. LoRA attachment across verified language modules;
5. gradient checkpointing and cache settings;
6. residual output capture at all 24 complete language layers;
7. `Q_END` token identification;
8. one residual-stream activation edit;
9. metric and checkpoint serialization;
10. adapter disable/enable and merge behavior.

Then run a short representative 2B smoke test. Record peak memory, examples/second, tokens/second, checkpoint size, and projected wall time.

**Gate 0 passes only if** the 2B model completes forward/backward, exposes all layer outputs, and accepts a reproducible `Q_END` edit. If unsupported MPS operations appear, try a pinned compatible framework version or a narrowly documented CPU fallback. Do not begin scientific training with an unvalidated execution path.

### Phase 3 — BASE audit

1. Score all transformed forget questions with `BASE` before any project training.
2. Save per-item mean answer log-probability, perturbed margin, greedy output, ROUGE-L, and exact match.
3. Flag base-correct items without deleting them.
4. Freeze the intent-to-audit set and the predeclared base-correct sensitivity set.

### Phase 4 — train and select FULL

1. Train the 2B `FULL` LoRA on transformed `full` for up to five epochs.
2. Save an adapter checkpoint and complete evaluation after every epoch.
3. Preserve optimizer/scheduler state for resume.
4. Select the earliest checkpoint satisfying the acquisition rule.
5. Record the entire trajectory and the exact checkpoint decision.

**Gate 1:** `FULL` must clearly improve over `BASE` on both forget and retain facts and meet the frozen ROUGE-L rule, or use the strongest checkpoint only as the specification permits. If the model fails to acquire the facts, stop downstream causal claims and write an acquisition-failure result.

### Phase 5 — train and select RETAIN

1. Reset to the byte-identical `BASE` initialization.
2. Train `RETAIN` with the identical LoRA parameterization, formatter, optimizer family, epoch protocol, effective batch size, and seed.
3. Report the unequal unique-example/token budget explicitly.
4. Optionally run repeated-retain update matching only if it does not threaten the main experiment.
5. Evaluate forget, retain, real-author, and world-fact sets.

**Gate 2:** `FULL` must score materially above `RETAIN` on transformed forget bindings while outside-forget utility remains comparable. Investigate reconstruction or split contamination before continuing if separation is weak.

### Phase 6 — construct IDK and GD trajectories

Run these sequentially from the selected `FULL` state.

For `IDK`:

1. freeze `FULL` completely;
2. attach a new removable refusal LoRA;
3. train on varied refusal targets paired with retain preservation;
4. save frequent checkpoints;
5. verify that disabling the adapter restores `FULL` numerically and leaves the underlying files unchanged.

For `GD`:

1. branch independently from selected `FULL`;
2. minimize `-L(F) + lambda_R L(R)` with one retain example per forget example;
3. save frequent resumable checkpoints;
4. record that adapter-mediated GD preserves an accessible underlying `FULL` and therefore limits conclusions about weight-level erasure.

Using `F_discovery` only, select behaviorally matched `IDK` and `GD` states using the frozen 0.15 nats/token, paired-bootstrap, perturbed-margin, and retain-utility rules.

**Gate 3:** if only one state can be matched, run the calibrated comparison for that state and narrow the claim. If matching fails, save the Pareto trajectory and do not make a strong absence-versus-inaccessibility claim.

### Phase 7 — cache discovery activations

1. Load one model state at a time.
2. Cache float32 `Q_END` residual activations for every discovery question and all 24 layers.
3. Index by model, checkpoint, layer, author, question, token location, tokenizer revision, and dataset hash.
4. Treat `A_PREV` as exploratory and run it only after the main `Q_END` result.
5. Validate cache equality on a repeated sample before using it to form directions.

### Phase 8 — discovery sweep and intervention freeze

1. Compute normalized layer-wise `FULL - IDK` mean directions.
2. Compute the secondary `FULL - GD` directions separately, without using them as the primary calibration.
3. Generate sign-reversed, 20 norm-matched isotropic random, and label-shuffled controls.
4. Sweep all 24 layers and coefficients `{-2, -1, -0.5, 0.5, 1, 2}` on discovery authors only.
5. Evaluate the frozen selectivity objective `J` and retain-control effects.
6. Inspect whether apparent selectivity comes from positive `IDK` recovery or negative effects in `RETAIN`.
7. Select exactly one primary layer and coefficient.
8. Write an immutable intervention manifest containing the direction hash, sign, layer, coefficient, norm scale, discovery data hash, code hash, and selection timestamp.

After this manifest is written, confirmation-dependent implementation changes require invalidating and rerunning the entire confirmation stage; they may not be patched around individual confirmation outcomes.

### Phase 9 — Gate 4 confirmatory run

Open `F_confirmation` for outcomes once and apply the frozen intervention to all required cells:

- `IDK`, `GD`, `RETAIN`, and `FULL` with no intervention;
- each relevant state with the learned direction;
- `IDK`, `GD`, and `RETAIN` with random, shuffled, and sign-reversed controls.

Before interpreting effects, report—but do not tune on—the unsteered confirmation baseline and whether `IDK`, `GD`, and `RETAIN` remained behaviorally comparable on these authors.

Save QA-level values immediately. Compute:

- absolute per-model `Delta s`;
- `C_IDK = Delta s(IDK) - Delta s(RETAIN)`;
- `C_GD = Delta s(GD) - Delta s(RETAIN)`;
- correct-versus-perturbed margin changes;
- target-rank improvement fraction;
- ROUGE-L and exact match;
- effects on `R_control`, real authors, and world facts;
- author-wise distributions and consistency.

**Gate 4:** the assay is calibrated only if held-out `IDK` recovery is positive in absolute terms, exceeds `RETAIN`, exceeds the random-direction distribution, and does not create comparable unrelated utility effects.

**Gate 5:** interpret `GD` only if Gate 4 passes. If calibration fails, the result is a falsification of this mean linear audit, not evidence of deletion.

### Phase 10 — statistics and figures

1. Treat authors as independent clusters.
2. Run at least 5,000 author-clustered bootstrap resamples with a recorded seed.
3. Report mean, median, 95% interval, and count of authors with positive effects.
4. Report both intent-to-audit and acquired-only sensitivity analyses.
5. Keep the small effective sample size—ten confirmation authors—visible; bootstrap count must not be described as increased statistical power.
6. Produce the main held-out recovery plot with author-level points.
7. Produce behavioral-matching trajectories and the discovery heatmap.
8. Regenerate every plot from saved tidy metrics, not from in-memory state.

### Phase 11 — reports and reproducibility audit

1. Write `findings.md` using only the permitted cell from the interpretation matrix.
2. Write `executive_summary.md` from the matching frozen claim template.
3. Write `README.md` with exact setup, resume, evaluation, and figure-reproduction commands.
4. Verify that every manifest path exists and every reported number is traceable to `metrics_long.csv` or `author_effects.csv`.
5. Run a clean-process reproduction of statistics and figures without loading model weights.
6. Capture final git status without staging or committing user files.

## 7. Required outputs and completion criteria

The run is complete when the following exist and validate:

```text
experiment/
  configs/
    base.yaml
    full.yaml
    retain.yaml
    idk.yaml
    grad_diff.yaml
  src/
    data.py
    train.py
    evaluate.py
    hooks.py
    steer.py
    statistics.py
    figures.py
  artifacts/
    run_manifest.json
    data_audit.json
    frozen_splits.json
    checkpoint_selection.json
    metrics_long.csv
    author_effects.csv
    main_recovery_plot.png
    behavioral_matching.png
    discovery_heatmap.png
  findings.md
  executive_summary.md
  README.md
```

Also retain:

- timestamped logs for every stage;
- dependency lockfile;
- checkpoint lineage and completion markers;
- direction/control hashes;
- base-correct item flags;
- a machine-readable gate-status summary;
- enough checkpoint state to resume any interrupted long-running stage.

A scientifically valid negative result counts as completion. Skipping a mandatory control or opening confirmation early does not.

## 8. Monitoring protocol for the user

The agent should provide concise progress updates at these boundaries:

- environment ready;
- data frozen;
- Gate 0 result and measured wall-time forecast;
- each completed training epoch;
- Gates 1–3 decisions;
- discovery sweep complete and intervention frozen;
- confirmation opened;
- Gates 4–5 decisions;
- statistics/figures complete;
- final reproducibility audit.

During long commands, emit or surface a heartbeat at least every 60 seconds when the interface permits. Logs should include timestamps, stage, epoch/checkpoint, elapsed time, ETA based on measured throughput, memory pressure, loss, and last durable artifact.

The agent should automatically retry transient downloads and resume interrupted training. It should not repeatedly retry deterministic OOMs, unsupported operations, corrupt artifacts, or failed scientific gates.

## 9. Before-start checklist

Immediately before the execution agent begins, verify:

- [ ] the session visibly shows Full Access/unrestricted filesystem and network;
- [ ] approval policy is no-prompt/never rather than on-request;
- [ ] the project path is `/Users/satwikpandey/Dev/unlearn-fr`;
- [ ] the Mac is connected to power;
- [ ] at least 150 GiB disk remains free;
- [ ] no other memory-intensive local model job is running;
- [ ] the user accepts that the laptop may remain under sustained load overnight;
- [ ] the agent is told to execute this file and `context/experiments.md`, not redesign the study;
- [ ] no request is made to push, purchase compute, or modify unrelated files.

Suggested start instruction:

> Execute `plan.md` and the frozen specification in `context/experiments.md` end to end. You are authorized to install repository-local dependencies, download the pinned public metadata/data, create experiment code and artifacts, and run all in-scope jobs without routine confirmation. Preserve existing user changes, never tune on confirmation authors, obey every scientific gate, keep me updated at the checkpoints in the plan, and produce a valid negative result if a gate fails. Do not push, purchase compute, or modify unrelated files.

## 10. Known risks to watch tonight

| Risk | Detection | Response |
|---|---|---|
| Qwen 3.5 unsupported by installed Transformers | import/model-class smoke test | pin a compatible release; record exact version |
| MPS unsupported operation | 0.8B/2B backward smoke | try compatible version or narrow CPU fallback; do not hide fallback |
| 2B OOM | measured peak memory during representative step | reduce microbatch, enable checkpointing, keep effective batch via accumulation, use LoRA |
| Runtime exceeds night | measured tokens/second forecast | inspect acquisition curves before reducing epochs; preserve mandatory controls |
| Thermal throttling | falling throughput and system pressure | one heavy job at a time; keep on AC; allow cooling pause without changing science |
| Model directory incomplete | clean-process load failure | fetch pinned ancillary files and verify hashes before training |
| Confirmation leakage | code/log audit and immutable intervention manifest | invalidate confirmation and report the breach; never tune around it |
| Generic anti-refusal direction | similar gains in `RETAIN` or unrelated sets | fail calibration or narrow claim |
| Contrast driven by damaged `RETAIN` | absolute per-model `Delta s` | require absolute positive recovery, not contrast alone |
| Discovery matching fails on confirmation | unsteered confirmation baseline | report mismatch; do not retune; narrow interpretation |
| User worktree overwritten | initial/final status diff | preserve pre-existing deletion and untracked files |

## 11. Final feasibility summary

The repository, network, disk, public-data access, Hugging Face authentication, and exact official model weights are available. The current Codex session has the correct unrestricted/no-approval profile. The run is not yet turnkey because the Python ML environment and complete model directories do not exist; both can be created automatically without administrator access or interactive authentication.

The largest uncertainty is compute feasibility, not permissions. The permitted LoRA path makes the 2B experiment plausible on 16 GiB unified memory, but measured Gate 0 throughput must determine the realistic wall-time forecast. No scientific conclusion should be traded for the appearance of finishing within 16 hours.
