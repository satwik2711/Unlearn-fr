# Differential Causal Recoverability — Implementation Plan

## 1. Status and question

This document is the active scientific and implementation source of truth for
`pivot_experiment/`. It supersedes the earlier IDK-calibrated audit described in
`context/experiment_pivot.md`.

After the frozen candidate screen failed to find a behavioral match, the
revised exploratory question is:

> Under the tested intervention, does the utility-preserving `gd_02` GradDiff
> state respond differently to causal information restoration from `FULL` than
> the TOFU-withheld `RETAIN` reference?

The experiment tests **differential causal recoverability**, not whether a
memory is intact, erased, or merely inaccessible.

The frozen experimental spine is:

$$
\boxed{
\text{freeze utility-preserving gd\_02 with an unmatched-baseline limitation}
\rightarrow
\text{FULL}\rightarrow\{\text{GD},\text{RETAIN}\}\text{ exact patching}
\rightarrow
\text{freeze }l^*
\rightarrow
\operatorname{CAA}(\text{FULL}-\text{GD})
\rightarrow
\text{held-out GD-versus-RETAIN steering}
}
$$

## 2. Scope revision from the failed IDK calibration

Two removable-LoRA calibration attempts were completed and failed P1:

1. refusal-only SFT learned to emit a refusal without sufficiently reducing the
   teacher-forced likelihood of the correct answer;
2. direct target suppression produced progressively stronger suppression, but
   damaged `R_control` before approaching `RETAIN`-like forgetting or the
   required refusal margin.

Their formal reports remain at:

- `archive/idk_refusal_failed/gate_eval.json`
- `archive/idk_suppression_failed/gate_eval.json`

Their optional local payloads live under ignored `data/` directories. IDK is no
longer a required model state, positive control, gate, layer-selection source,
or steering-direction source. Neither failed adapter may be tuned further using
confirmation authors.

Because no candidate behavior-matched `RETAIN`, this experiment is exploratory.
Permitted claim if the new experiment succeeds:

> The utility-preserving `gd_02` state showed greater differential response to
> the tested `FULL` residual intervention than `RETAIN`, despite unequal
> baseline target likelihoods.

Forbidden claims include “the memory is intact,” “the memory was not erased,”
and “GD only hides knowledge.” The failed reversible calibration does not
support those stronger interpretations.

## 3. Frozen states and public assets

### Main states

| State | Construction | Experimental role |
|---|---|---|
| `FULL` | public OpenUnlearning TOFU full checkpoint | accessible donor/reference |
| `RETAIN` | public OpenUnlearning retain90 checkpoint | TOFU-withheld reference receiver |
| `GD` | pinned `gd_02` public GradDiff checkpoint | exploratory unlearned receiver |

`RETAIN` means that forget10 was withheld during standardized TOFU fine-tuning;
it does not mean those facts were absent from all pretraining.

### Dataset and architecture

- Dataset: `locuslab/TOFU`, pinned revision already recorded in configuration.
- Forget data: `forget10` and `forget10_perturbed`.
- Retain data: `retain90`.
- Model family: Llama-3.2-1B-Instruct.
- Decoder layers: 16.
- Hidden size: 2048.
- Tokenizer source: pinned `FULL` tokenizer.
- Primary residual location: output of complete decoder block `model.layers[l]`.
- Primary token position: `Q_END`, the final prompt token immediately before
  the first answer token.

All states must use the same frozen rendered prompt, prompt date, tokenizer
revision, token IDs, answer boundary, and evaluator version. A calendar-dependent
chat template is prohibited.

### Ordered GD candidates

The ordered candidate list is frozen before evaluation:

1. `GradDiff_lr1e-05_alpha1_epoch10`
2. `GradDiff_lr1e-05_alpha2_epoch5`
3. `GradDiff_lr3e-05_alpha1_epoch5`
4. `GradDiff_lr3e-05_alpha1_epoch10`

Exact Hugging Face repository IDs and immutable revisions must be added to
`configs/models.yaml` before the first candidate run. Do not search the broader
checkpoint collection after seeing results.

All four candidates were evaluated. None met the frozen behavior-match
criterion. `gd_02` was subsequently frozen by an explicit scope revision
because it was the only candidate that passed the original `R_control` utility
threshold. Its discovery mean is `-1.3191201`, versus `-2.7865959` for
`RETAIN`; the absolute gap is `1.4674758` nats/token. Its `R_control`
degradation is `0.0877231`, within the frozen `0.20` threshold. This choice is
post-screen and exploratory, and must never be described as behavior-matched.

## 4. Frozen data boundaries

Reuse the existing immutable split manifest and split hash:

- discovery: 5 forget authors, approximately 100 questions;
- exact patching: all 5 discovery authors, approximately 100 questions; the
  older 3-author `patch_authors` subset remains in the immutable manifest but is
  not used by the revised differential screen;
- confirmation: 5 different forget authors, approximately 100 questions;
- reserve: remaining 10 forget authors, untouched by the MVP;
- `R_control`: 5 retain authors, approximately 100 questions.

Discovery may be used for GD checkpoint selection, exact layer selection,
direction construction, and alpha selection.

Confirmation may be opened only after the selected GD checkpoint, layer,
direction bytes, direction sign, alpha, random seeds, and all thresholds have
been written to an immutable freeze manifest. Confirmation results may never
change those choices. Reserve remains unopened unless a frozen optional
replication is started after the main result.

## 5. Metrics and statistical unit

For answer tokens $a_1,\ldots,a_m$:

$$
s(q,a;M)
=
\frac{1}{m}\sum_{t=1}^{m}
\log P_M(a_t\mid q,a_{<t}).
$$

Mean correct-answer token log-probability is authoritative everywhere. Greedy
generation and ROUGE-L are not gates and are disabled in the main pipeline.

Correct-minus-perturbed margin is a secondary behavioral diagnostic:

$$
m(q;M)=s(q,a_{correct};M)-\frac{1}{5}\sum_j s(q,a_{perturbed,j};M).
$$

The inferential unit is the author, not the individual question. Report:

- example mean and median;
- effect for every author;
- number of authors with the expected sign;
- author-clustered bootstrap 95% interval using 5,000 resamples and seed 42.

No threshold may be changed after confirmation is opened.

## 6. Artifact and execution contract

### Authoritative storage

- JSONL: per-example scores and intervention effects.
- JSON: manifests, gate reports, selections, and frozen decisions.
- safetensors: residual activations and direction vectors.
- CSV: optional derived views only; gates never read CSV.

Every score record must identify the state, exact model revision, split hash,
author, example, prompt hash, evaluator version, intervention, layer, alpha,
direction hash/seed, mean target log-probability, and token count.

### Run minimization

1. Load each public checkpoint once per script invocation.
2. Store every needed baseline score during its first evaluation.
3. Cache all-layer discovery `Q_END` activations during each GD candidate’s
   first behavioral evaluation. The caches remain sealed during candidate
   selection; only the selected candidate’s cache is consumed afterward.
4. Reuse the existing `FULL` discovery scores and all-layer activation cache.
5. Reuse existing `RETAIN` discovery and `R_control` scores.
6. Never recompute an unpatched or unsteered baseline per intervention.
7. Gate checks are read-only and never load a model.
8. Jobs are resumable only when their complete configuration/provenance hash is
   identical.

Confirmation is evaluated once after the freeze manifest exists. Conditions are
grouped by receiver model so `GD` and `RETAIN` are each loaded once.

## 7. Gate sequence

`scripts/check_gates.py` remains the only gate entry point. Its active meanings
become P0–P5 below. Historical IDK reports remain archived and are not emitted as
an active gate.

### P0 — FULL/RETAIN laboratory (already passed)

Requirements:

- mean `FULL - RETAIN` discovery score at least `0.15` nats/token;
- author-clustered 95% interval entirely above zero;
- identical frozen examples and prompt/tokenization provenance.

Existing result: `PASS`, mean difference `+2.6042456`, interval
`[+2.3936176, +2.8500798]`.

### P1 — GD screen and exploratory freeze

Select a candidate using discovery and `R_control` only. A candidate is eligible
only if:

$$
|\bar s_{GD}-\bar s_{RETAIN}|\le 0.15,
$$

and its mean `R_control` degradation relative to `FULL` is at most `0.20`
nats/token.

Select the first eligible candidate in the frozen order and stop evaluating
candidates. This sequential rule minimizes runs and prevents post-hoc selection
from an expanding candidate set. Report the correct-minus-perturbed margin,
per-author effects, and distribution summaries as diagnostics, not additional
selection criteria.

The original behavior-match screen failed for all four candidates. That failure
is retained as `artifacts/results/gd_candidate_screen.json`. Following an
explicit scope revision, P1 may close only as an **exploratory freeze** when the
configured `downstream_gd_candidate` has immutable model provenance, complete
discovery scores and activations, and passes the unchanged `R_control` utility
threshold. The gate must record `behavior_matched: false`, the baseline gap, and
`scope: exploratory_unmatched`. It may not authorize the original confirmatory
claim.

### P2 — differential exact patching

On all five discovery authors, for every layer
$l\in\{0,\ldots,15\}$, run:

$$
h_l^{GD}(q,Q_{END})\leftarrow h_l^{FULL}(q,Q_{END}),
$$

$$
h_l^{RETAIN}(q,Q_{END})\leftarrow h_l^{FULL}(q,Q_{END}).
$$

Using stored unpatched baselines, calculate:

$$
P_l^{GD}=s(GD_{patched,l})-s(GD),
$$

$$
P_l^{RETAIN}=s(RETAIN_{patched,l})-s(RETAIN),
$$

$$
D_l=P_l^{GD}-P_l^{RETAIN}.
$$

Freeze:

$$
l^*=\arg\max_l D_l.
$$

P2 passes only if:

- self-patching `GD <- GD` and `RETAIN <- RETAIN` changes scores by at most
  `1e-4` nats/token on the frozen audit subset;
- mean $D_{l^*}\ge 0.10$ nats/token;
- $D_{l^*}>0$ for at least 4 of 5 discovery authors;
- the author-clustered 95% interval for $D_{l^*}$ is above zero;
- paired FULL activations recover more than a deterministic mismatched-question
  FULL donor at the selected layer.

For the final item, the frozen scalar check is that the mean matched-minus-
mismatched patch advantage, averaged equally across the two receivers, is
strictly positive. Per-receiver advantages are always reported.

The mismatched donor is a fixed within-author cyclic permutation with no
self-pairs. It is a content-specificity control, not another selection surface.

If no layer passes, record “no differential causal recoverability detected by
exact Q_END patching” and stop before direction construction.

### P3 — direction construction and discovery alpha freeze

At frozen $l^*$, use all discovery questions and already-cached activations:

$$
v=\frac{1}{N}\sum_q
\left(h_{l^*}^{FULL}(q)-h_{l^*}^{GD}(q)\right).
$$

Store the raw vector, its unit vector and norm, per-author vectors, cosine
diagnostics, source IDs, layer, residual location, and input hashes. Steering
uses the raw mean difference:

$$
h'_{l^*}(Q_{END})=h_{l^*}(Q_{END})+\alpha v.
$$

Evaluate only $\alpha\in\{0.5,1.0,2.0\}$ on discovery in both receivers. For
each alpha define:

$$
J(\alpha)=
\Delta s_{GD}
-\Delta s_{RETAIN}
-0.5\max\left(
|\Delta s_{R,GD}|,
|\Delta s_{R,RETAIN}|
\right).
$$

Select the largest $J$, breaking ties toward smaller alpha. P3 passes only if:

- the direction is finite, nonzero, and reproducible from its recorded inputs;
- discovery differential recovery is at least `0.10` nats/token;
- the maximum absolute `R_control` change in either receiver is at most `0.20`;
- the selected alpha has positive differential recovery in at least 4 of 5
  discovery authors.

After P3, write a freeze manifest containing selected GD revision, $l^*$,
direction hash, positive sign, alpha, hook semantics, confirmation IDs, five
random seeds, thresholds, and all upstream artifact hashes.

### P4 — held-out differential steering

Only after the freeze manifest is complete, score confirmation baselines and the
single learned intervention in `GD` and `RETAIN`:

$$
C_{steer}
=
\mathbb E[\Delta s(GD,+\alpha v)]
-
\mathbb E[\Delta s(RETAIN,+\alpha v)].
$$

P4 passes only if:

- mean $C_{steer}\ge 0.10$ nats/token;
- $C_{steer}>0$ for at least 4 of 5 confirmation authors;
- its author-clustered 95% interval is above zero.

Confirmation baseline mismatch is reported as a frozen sensitivity diagnostic;
it cannot trigger candidate reselection or alpha tuning.

### P5 — held-out specificity and utility

Generate five isotropic random directions from the frozen seeds and rescale each
to exactly $\|v\|_2$. Apply the same frozen alpha and hook location to `GD` and
`RETAIN` confirmation questions.

P5 passes only if:

- learned $C_{steer}$ exceeds every one of the five random-direction
  GD-minus-RETAIN contrasts;
- the learned direction’s already-frozen discovery `R_control` change remains
  within `0.20` nats/token in both receivers;
- all intervention records match the freeze manifest exactly.

P4 is the primary scientific endpoint. P5 establishes basic direction
specificity and utility; it is not a second opportunity to tune the result.

## 8. Implementation stages

### Stage 0 — preserve completed work and revise the gate system

1. Keep existing FULL/RETAIN data, scores, activations, manifests, and P0.
2. Keep both failed IDK `gate_eval.json` reports as negative calibration results.
3. Remove IDK from active gate dependencies and documentation.
4. Add the frozen prompt date to active configuration and provenance.
5. Update `check_gates.py` so active P1–P5 implement this document without
   importing model libraries.
6. Do not start any model evaluation during this stage.

Acceptance: P0 still passes from existing artifacts; P1 reports `BLOCKED`
because GD artifacts do not yet exist; confirmation has not been scored.

### Stage 1 — GD configuration and evaluator

Implement:

- pinned ordered GD candidate configuration;
- a candidate evaluator for discovery and `R_control`;
- blind all-layer discovery activation caching in the same candidate pass;
- atomic JSONL/manifests and resume checks;
- P1 candidate validation and deterministic selection.

Smoke tests may use synthetic/tiny models and separate paths. They must not
create authoritative experimental artifacts or touch confirmation examples.

The user starts each full candidate evaluation manually. Evaluate candidates in
frozen order and stop immediately when the first eligible candidate passes P1;
do not download or evaluate later candidates unnecessarily.

### Stage 2 — exact differential patching (exploratory unmatched comparison)

Implement one patching script that:

- loads frozen `gd_02` once and `RETAIN` once;
- reads paired FULL donor activations from the existing cache;
- runs all 16 exact matched patches for each receiver on all five discovery
  authors;
- reuses stored unpatched baselines;
- performs self-patch audit conditions on a small frozen subset;
- performs the fixed mismatched-question control only at the candidate winning
  layer after the 16-layer table is complete;
- writes all per-example effects before P2 is checked.

Layer selection and every P2 calculation occur in the read-only gate code.

### Stage 3 — direction and alpha selection

Build the raw `FULL - GD` vector without loading a model by reading cached
discovery activations. Validate layer/token/model alignment before subtraction.

Then run one grouped discovery steering job:

- `GD` with alpha 0.5, 1.0, and 2.0;
- `RETAIN` with the same three alphas;
- `GD` and `RETAIN` `R_control` with the same alphas;
- no repeated unsteered baselines.

P3 selects alpha and writes the immutable confirmation freeze manifest.

### Stage 4 — confirmation and controls

One grouped confirmation script performs:

- unsteered baseline and learned-direction condition for `GD`;
- unsteered baseline and learned-direction condition for `RETAIN`;
- the five frozen norm-matched random directions in both receivers;
- incremental writes so a crash resumes without duplicating completed cells.

Do not score confirmation `FULL`, IDK, or reserve states for the MVP because they
are unnecessary for the primary contrast.

Run P4 and P5 only after all frozen confirmation cells are complete.

### Stage 5 — analysis

Generate from stored artifacts only:

- behavioral matching table and distributions;
- 16-layer $P_l^{GD}$, $P_l^{RETAIN}$, and $D_l$ plot;
- per-author selected-layer patch effects;
- discovery alpha/selectivity plot;
- confirmation learned-versus-random contrasts;
- utility table;
- a claim-bounded positive or negative result report.

No analysis script loads a language model.

## 9. Run ledger

| Produced once | Reused for |
|---|---|
| existing FULL discovery scores | P0, P1 reference, patch/steering baselines |
| existing RETAIN discovery scores | P0, P1 matching, P2/P3 baselines |
| existing FULL all-layer discovery cache | both receiver patch donors and FULL half of CAA |
| each GD candidate discovery scores | P1 and behavioral analysis |
| each GD candidate `R_control` scores | P1 utility |
| each GD candidate blind discovery cache | selected GD half of CAA; no selected-GD rerun |
| selected GD unpatched scores | P2 and P3 baselines |
| 32 exact layer conditions | 16 layers × 2 receivers for P2 |
| selected-layer mismatched donor | P2 content specificity |
| six alpha conditions | 3 alphas × 2 receivers for P3 |
| six alpha utility conditions | 3 alphas × 2 receiver `R_control` states |
| confirmation baselines | all learned/random deltas |
| confirmation learned/random conditions | P4, P5, and final analysis |

No FULL or RETAIN public baseline is rerun unless a provenance validator proves
the stored artifact incompatible with the frozen prompt contract. Such an
incompatibility must be reported before any replacement run begins.

## 10. Stopping and interpretation rules

- P0 failure: the public laboratory lacks the required FULL/RETAIN contrast.
- P1 screen failure: no tested public GD checkpoint is behavior-matched. Preserve
  that negative result; downstream work is an explicitly exploratory unmatched
  comparison using the separately frozen utility-preserving `gd_02` state.
- P2 failure: no differential exact-patching recovery was detected.
- P3 failure: no selective shared FULL−GD steering direction was found on
  discovery.
- P4 failure: no held-out differential causal recovery was detected.
- P5 failure with P4 success: held-out differential recovery was observed, but
  direction specificity or utility was not established.
- P4 and P5 success: report differential causal recoverability under the tested
  residual intervention, with no claim that memory is globally intact.

At every scientific failure, save the formal gate JSON and produce the valid
negative result. Do not relax gates, add candidates, change layer rules, or tune
on confirmation authors.

## 11. Step-by-step execution order

1. Implement Stage 0 and formally revalidate P0.
2. Implement Stage 1; smoke-test without authoritative model runs.
3. User manually runs the first GD candidate.
4. Check P1; run the next frozen candidate only if required.
5. Freeze `gd_02` with the unmatched-baseline limitation and implement Stage 2.
6. User manually runs exact differential patching; check P2.
7. Build direction from caches; implement/run discovery alpha job; check P3.
8. Audit the freeze manifest before opening confirmation.
9. User manually runs the grouped confirmation job once.
10. Check P4/P5 and generate the final stored-artifact analysis.

The next action is **Stage 0 only**. No GD model run begins until its code,
configuration, artifact schema, and P1 gate have been reviewed.
