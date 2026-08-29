# IDK-Calibrated Causal Recovery — Final Experiment Plan

## 1. Scientific question

This document is the final scientific and implementation source of truth for
`pivot_experiment/`. It supersedes every earlier P0–P5 and GD-selected-layer
plan.

The question is:

> Does a residual intervention calibrated entirely on a known reversible IDK
> adapter recover proportionally more target-answer evidence from `GD02` than
> from the TOFU-withheld `RETAIN` reference?

The experiment tests recovery under one specified residual-stream assay. It
does not determine whether a memory is globally intact, erased, or inaccessible.

The frozen experimental spine is:

$$
\boxed{
\text{FULL}\rightarrow\text{IDK layer localization}
\rightarrow l^*
\rightarrow \operatorname{CAA}(\text{FULL}-\text{IDK})
\rightarrow \text{held-out IDK/GD02/RETAIN steering}
}
$$

`IDK` calibrates the mechanism, `RETAIN` is the negative reference, and `GD02`
is the unknown transfer state.

## 2. Scope revision and prior negative results

Two earlier IDK calibration gates were too restrictive for the narrower causal
question:

1. refusal-only SFT was exactly reversible and utility-preserving, but missed
   the former target-suppression threshold and did not behavior-match `RETAIN`;
2. direct suppression did not meet its former calibration gate, and its large
   untracked checkpoint payload is no longer available.

The failures remain valid negative results at:

- `archive/idk_refusal_failed/gate_eval.json`
- `archive/idk_suppression_failed/gate_eval.json`

They must not be rewritten as passes. The final experiment simply does not
require IDK to behavior-match `RETAIN`. It requires IDK to be a real,
utility-preserving, removable intervention with positive target-likelihood
headroom relative to `FULL`.

The active IDK is the archived refusal adapter `step-000025`:

- adapter hash:
  `0b634c3a2a9be7a53b2e3890b42e7e1b41a8bef963fd5db686b1c83ddc43aa90`;
- mean `FULL - IDK`: `0.1198518` nats/token;
- mean `R_control` degradation: `0.1129755` nats/token;
- adapter-off equality with `FULL`: exact within the stored audit;
- positive `FULL - IDK` headroom: 100 of 100 discovery questions.

This is a weak reversible suppression control, not a `RETAIN`-matched forgetting
model. That limitation must appear in every result report.

## 3. Frozen model states

| State | Construction | Role |
|---|---|---|
| `FULL` | pinned public TOFU full checkpoint | activation donor and accessible reference |
| `IDK` | `FULL` plus archived refusal LoRA `step-000025` | reversible calibration receiver |
| `GD02` | pinned public GradDiff `gd_02` checkpoint | unknown transfer receiver |
| `RETAIN` | pinned public TOFU retain90 checkpoint | withheld-data reference receiver |

`GD02` is:

- repository:
  `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch5`;
- revision: `940728b3791615faa07279b771b73e83c13b0c6c`;
- discovery mean: `-1.3191201` nats/token;
- `R_control` degradation from `FULL`: `0.0877231` nats/token.

`GD02` does not behavior-match `RETAIN`; their discovery means differ by
`1.4674758` nats/token. Normalized recovery is therefore required alongside raw
recovery.

The tokenizer always comes from pinned `FULL`. All states use the same rendered
prompt, frozen prompt date `28 Aug 2026`, answer boundary and token IDs. The
archived records remain the frozen behavioral description and provenance
check. Intervention effects, however, must be differenced against a baseline
scored in the same software/device runtime as the intervention job. This avoids
mixing small MPS/BF16 cross-run drift into a causal effect. Stage A records the
archived compatibility audit rather than pretending the evaluator-version
labels are identical.

## 4. Frozen data boundaries

Reuse the immutable split manifest and split hash:

- discovery: five forget authors, 100 questions;
- confirmation: five different forget authors, approximately 100 questions;
- reserve: remaining forget authors, unopened in the main experiment;
- `R_control`: five retain authors, approximately 100 questions.

Discovery may be used only for:

- IDK layer localization;
- direct GD02/RETAIN patch-transfer measurement at the already selected layer;
- construction of the `FULL - IDK` direction;
- alpha selection using IDK, RETAIN and `R_control` only.

GD02 discovery results may never change the layer, direction, sign, alpha,
random seeds or analysis rules. Confirmation remains sealed until all those
choices and their hashes are recorded in the freeze manifest. Reserve remains
unopened.

## 5. Authoritative behavioral metric

For answer tokens $a_1,\ldots,a_m$:

$$
s(q,a;M)
=
\frac{1}{m}\sum_{t=1}^{m}
\log P_M(a_t\mid q,a_{<t}).
$$

Mean correct-answer token log-probability, measured in nats/token, is the
primary mechanistic outcome. Every intervention also retains the unpatched
score required to calculate a paired effect.

Greedy generation is not used across the layer sweep. At the selected-layer and
final learned-steering conditions only, store greedy generations and report a
frozen TOFU answer-overlap score as a secondary behavioral diagnostic. This
diagnostic does not select a layer or alpha.

The inferential unit is the author. Always report:

- question-level records;
- author means;
- the mean across authors;
- number of authors with the expected sign;
- author-clustered bootstrap 95% intervals using 5,000 resamples and seed 42.

## 6. Raw and normalized recovery

For state $M\in\{\text{IDK},\text{GD02},\text{RETAIN}\}$, layer $l$, question
$q$ and author $a$:

$$
\Delta_l^M(q)
=
s(M_{\text{patched},l};q)-s(M;q).
$$

Because baseline target likelihood differs substantially across states,
fractional recovery is calculated at the author level, not by averaging
unstable per-question ratios:

$$
RF_{l,a}^M
=
\frac{
\overline{s(M_{\text{patched},l})-s(M)}_a
}{
\overline{s(\text{FULL})-s(M)}_a
},
$$

$$
RF_l^M=\frac{1}{5}\sum_a RF_{l,a}^M.
$$

No ratio is clipped. Values below zero and above one remain visible. Raw
recovery in nats/token is always co-primary in tables and plots. If any author
denominator is non-positive or smaller than `0.02` nats/token, flag that
author’s normalized result as unstable and retain the raw effect; this is a
reporting warning, not a stopping rule.

## 7. No scientific pass/fail gates

The final pipeline has no scientific stopping gates. Every completed stage
produces a frozen decision or an analysis report, regardless of whether the
effect is positive, null or contrary.

Validation failures remain hard errors: incompatible model revisions, stale
prompt hashes, missing examples, invalid activation shapes, duplicate cells or
opened confirmation data must stop execution until corrected. These are data
integrity checks, not scientific gates.

The old `P0` and candidate-screen results remain historical context. The active
model-free audit command reports stages as `INCOMPLETE` or `COMPLETE` and never
changes a scientific choice after its freeze artifact exists.

## 8. Stage A — reconcile and freeze assets

Before any new model run:

1. Freeze `FULL`, `RETAIN`, `GD02` and IDK `step-000025` revisions/hashes.
2. Create portable active references to the archived IDK checkpoint, discovery
   scores, `R_control` scores and all-layer discovery activations. Do not copy or
   retrain them unnecessarily.
3. Rewrite stale absolute archive paths through a deterministic resolver; do
   not mutate the archived source records.
4. Validate 100 aligned discovery examples, prompt hashes, tokenization,
   activation shapes `[batch, 16, 2048]` and split hashes.
5. Record the exact 100-example IDK headroom distribution and five author-level
   denominators.
6. Mark the current GD-selected P2 implementation obsolete; it must not be run
   as the final experiment.

Output: `artifacts/freeze/final_states.json`.

## 9. Stage B — IDK-only layer localization

For each discovery question and every layer $l\in\{0,\ldots,15\}$, patch the
same-question cached `FULL` residual at `Q_END` into IDK:

$$
h_l^{IDK}(q,Q_{END})\leftarrow h_l^{FULL}(q,Q_{END}).
$$

Before intervention scoring, run one unpatched pass over discovery with the IDK
adapter off (`FULL`) and on (`IDK`) in the intervention runtime. Do not generate
text during the 16-layer sweep. Store all 1,600 patched scores incrementally.
Store raw patched log-probabilities separately from the derived recovery table,
so completed patched forwards can be mechanically rebased if an interrupted
job reveals a baseline-provenance problem.

Calculate raw and author-level normalized recovery, then freeze:

$$
l^*=\arg\max_l RF_l^{IDK},
$$

with ties resolved in favor of the lower layer index. Record all layer curves,
per-author effects and intervals even if the best effect is weak or negative.

Run a four-author self-patch audit at $l^*$ to validate the hook. This is an
engineering check only.

Outputs:

- `artifacts/scores/idk_runtime_baselines.jsonl`;
- `artifacts/interventions/idk_layer_sweep.jsonl`;
- `artifacts/interventions/idk_layer_sweep_rebased.jsonl`;
- `artifacts/freeze/causal_layer.json`.

## 10. Stage C — direct patch-transfer readout

After $l^*$ is immutable, patch `FULL` into GD02 and RETAIN at all 16 layers for
complete descriptive curves:

$$
h_l^{GD02}(q,Q_{END})\leftarrow h_l^{FULL}(q,Q_{END}),
$$

$$
h_l^{RETAIN}(q,Q_{END})\leftarrow h_l^{FULL}(q,Q_{END}).
$$

These jobs may run independently because neither can affect layer selection.
Before each sweep, score its 100 receiver baselines in the intervention runtime;
reuse Chunk 2's current-evaluator FULL baselines for the shared denominators and
the frozen FULL activation cache for donors. Write 1,600 cells per receiver.

The prespecified transfer readout is evaluated only at the already frozen
$l^*$:

$$
C_{patch}
=
RF_{l^*}^{GD02}-RF_{l^*}^{RETAIN}.
$$

Report $C_{patch}$, both component fractional recoveries, both raw recoveries,
five author effects and the clustered interval. A null or negative value is a
valid result and does not stop later stages.

Outputs:

- `artifacts/scores/gd02_runtime_baseline.jsonl`;
- `artifacts/scores/retain_runtime_baseline.jsonl`;
- `artifacts/interventions/gd02_layer_sweep.jsonl`;
- `artifacts/interventions/retain_layer_sweep.jsonl`;
- `artifacts/results/patch_transfer.json`.

## 11. Stage D — calibrated direction construction

At frozen $l^*$, build the direction entirely from existing discovery caches:

$$
v
=
\frac{1}{N}\sum_q
\left(h_{l^*}^{FULL}(q,Q_{END})-h_{l^*}^{IDK}(q,Q_{END})\right).
$$

Validate example, prompt, model, layer, token-position and hidden-size alignment
before subtraction. Store the unnormalized mean vector, its L2 norm, dtype,
sign, source-example hash and tensor hash. Positive alpha means movement from
IDK toward FULL.

No GD02 activation or score may be read while constructing this direction.

Output: `artifacts/directions/full_minus_idk.safetensors` plus JSON manifest.

## 12. Stage E — alpha selection without GD02

On discovery, test the frozen grid:

$$
\alpha\in\{0.5,1.0,2.0\}.
$$

Apply each alpha to IDK and RETAIN at `Q_END` after layer $l^*$. Also measure
the raw `R_control` score change in both receivers. Do not evaluate GD02 during
alpha selection.

Define:

$$
U(\alpha)
=
\max\left(
|\Delta s_{R_{control}}^{IDK}|,
|\Delta s_{R_{control}}^{RETAIN}|
\right),
$$

and freeze $\beta=1.0\ \text{nat}^{-1}$ in:

$$
J(\alpha)
=
RF^{IDK}(\alpha)
-RF^{RETAIN}(\alpha)
-\beta U(\alpha).
$$

Choose the alpha with maximum $J$; ties choose the smaller alpha. Report all
components for all three alphas. No threshold stops the experiment.

Before opening confirmation, write one immutable manifest containing:

- all four state revisions and adapter hash;
- split and prompt hashes;
- $l^*$ and layer-selection input hash;
- direction tensor hash, sign and norm;
- $\alpha^*$ and the complete alpha table;
- $\beta=1.0\ \text{nat}^{-1}$;
- random seeds `[1042, 2042, 3042, 4042, 5042]`;
- generation and analysis settings.

Output: `artifacts/freeze/confirmation.json`.

Operationally, each receiver first records 200 same-runtime unsteered
baselines (100 discovery and 100 `R_control`), followed by 600 steered scores
(three alphas across both subsets). This keeps every steering delta within one
receiver runtime. The complete alpha table is also stored in
`artifacts/results/alpha_selection.json`.

## 13. Stage F — held-out confirmation and random controls

Only after the confirmation freeze exists, open confirmation authors. First
score `FULL` once on confirmation to provide the denominator for every
fractional-recovery endpoint. Then apply the same frozen layer, direction and
alpha to IDK, GD02 and RETAIN.

For each intervention receiver, evaluate:

1. unsteered baseline;
2. learned `FULL - IDK` direction;
3. five isotropic random directions generated from the frozen seeds and
   rescaled to exactly $\|v\|_2$.

All conditions use identical questions, answers and prompt provenance. Group
conditions by receiver so each model is loaded once, and write every cell
incrementally for exact resumption.

The two prespecified confirmation endpoints are:

$$
C_{IDK}
=
RF_{confirm}^{IDK}(\text{learned}),
$$

which asks whether the learned direction reverses known removable suppression,
and:

$$
C_{transfer}
=
RF_{confirm}^{GD02}(\text{learned})
-RF_{confirm}^{RETAIN}(\text{learned}),
$$

which asks whether the same calibrated direction recovers proportionally more
target evidence in GD02 than in RETAIN.

For both endpoints, report the learned result, all five random-direction
results, the learned direction’s rank among the six directions, raw recovery,
author effects and clustered intervals. Five controls support a basic
specificity comparison, not a precise tail probability.

Store greedy generations for FULL, receiver baselines and learned-direction
conditions only. Random-direction conditions remain likelihood-only.

## 14. Interpretation

If IDK recovery is positive and the learned direction exceeds the random
controls, the intervention is calibrated under this assay.

If additionally GD02 fractional recovery exceeds RETAIN on held-out authors,
the permitted conclusion is:

> A residual direction calibrated on a removable IDK intervention recovered
> proportionally more target-answer evidence from GD02 than from the
> TOFU-withheld RETAIN reference under the tested intervention.

Do not claim that this proves GD memory is intact, that unlearning merely hides
knowledge, or that the result generalizes beyond this layer, token position,
direction family, model family or dataset.

If IDK does not recover, report that the reversible control was not causally
recoverable under the assay; GD02 comparisons are then descriptive.

If GD02 and RETAIN are similar, report no differential transfer recovery.

If RETAIN recovers more, report the contrary result directly. No result causes
retuning on confirmation authors.

## 15. Run-minimization ledger

| Existing or new artifact | Reused for |
|---|---|
| existing FULL scores and all-layer discovery cache | all patch donors, archived behavior and direction |
| archived IDK step-000025 scores/cache/checkpoint | archived behavior, layer localization, direction and steering |
| one same-runtime FULL/IDK discovery baseline pass | IDK patch-effect and headroom denominators |
| existing GD02 scores/cache | baseline and descriptive analysis; cache is not used to construct the direction |
| existing RETAIN scores | baselines and denominators |
| 1,600 IDK patch cells | layer localization only |
| 1,600 GD02 patch cells | full curve and frozen-layer transfer readout |
| 1,600 RETAIN patch cells | full curve and frozen-layer reference readout |
| one cached `FULL - IDK` vector | all alpha and confirmation conditions |
| three alpha conditions on discovery and `R_control` for IDK/RETAIN | alpha selection without GD02 |
| one FULL confirmation baseline pass | all held-out fractional-recovery denominators |
| one grouped confirmation run per intervention receiver | learned and random held-out endpoints |

Never rerun a same-runtime baseline when its manifest matches the intervention
job. Never regenerate discovery activations or raw patched scores already
present. Audit code never loads a model.

## 16. Implementation chunks

### Chunk 1 — final asset freeze

Implement portable archived-IDK loading, validate all four states, preserve the
old negative reports, and emit `final_states.json`. No model run.

### Chunk 2 — IDK layer localization

Implement and manually run one FULL/IDK runtime-baseline pass and the 16-layer
IDK patch sweep. Freeze $l^*$ from IDK only and run the small live-activation
self-patch engineering audit. Archived/current activation drift is reported as
a diagnostic, not treated as hook failure.

### Chunk 3 — GD02/RETAIN patch transfer

Implement and manually run the two independent 16-layer receiver sweeps. Read
their scientific comparison only at frozen $l^*$. Finalization and completion
auditing are model-free, and a negative or null comparison remains valid.

### Chunk 4 — direction and alpha

Build `FULL - IDK` from caches without loading a model. Run IDK/RETAIN discovery
steering for the three alphas and `R_control`; freeze alpha and confirmation.

### Chunk 5 — confirmation

Score held-out FULL once, then run learned and five random directions on IDK,
GD02 and RETAIN. Add greedy-generation diagnostics for FULL, receiver baselines
and learned conditions.

### Chunk 6 — stored-artifact analysis

Generate layer curves, author-level normalized/raw recovery tables, steering
plots, random-control comparisons, generation diagnostics and a claim-bounded
positive, null or contrary report. No model is loaded.

Chunk 1 is complete. Chunk 2's 1,600 raw intervention scores are complete; its
initial derived freeze was invalidated because those scores were differenced
against an archived runtime. Resume the same command to score 200 same-runtime
FULL/IDK baselines, mechanically rebase the completed sweep, refreeze the
layer, and run four live self-patch audit rows. Do not run the obsolete
GD02/RETAIN P2 command; Chunk 3 will replace it with the frozen-layer transfer
implementation after Chunk 2 finishes.
