# Experiment Plan: A Causal Audit of Apparent Forgetting

## MATS 16-hour project using TOFU and Qwen 3.5 2B

**Status:** frozen first-experiment specification  
**Companion document:** `research.md` explains the broader research agenda. This file defines the first executable test only.  
**Primary model:** local Qwen 3.5 2B safetensors checkpoint  
**Dataset:** a frozen alias-transformed TOFU `forget10` / `retain90`, with TOFU perturbed answers and utility sets  
**Primary unlearning method:** Gradient Difference  
**Research budget:** 16 hours for experiments and write-up, plus 2 hours for the executive summary

---

## 1. What this experiment is for

The broader project asks whether mechanistic evidence can distinguish robust forgetting from behavioral non-recall. A model that no longer answers a question may have lost the relevant information, may never have learned it, or may retain information whose normal readout has been disrupted.

This experiment does **not** try to solve that whole problem. It tests whether one small causal audit has any signal:

> **When an unlearned model and a never-exposed model show similarly low recall, can a recovery intervention learned on one subset of forgotten authors selectively increase target-answer evidence for held-out authors in the unlearned model, without doing so in the never-exposed model?**

The proposed intervention is a layer-wise residual-stream direction averaged over discovery authors. It is then evaluated on authors that contributed nothing to the direction. The held-out design matters: an intervention built separately from every target answer could simply insert the answer rather than reveal residual information.

The intended contribution is a calibrated causal comparison, not a claim that a linear direction proves that a memory exists or is absent.

---

## 2. Dataset decision: use TOFU, not hand-written facts or MUSE

### Decision

Use the existing [TOFU dataset](https://huggingface.co/datasets/locuslab/TOFU) rather than generating 60 new fictitious facts, but apply the deterministic alias transformation below.

TOFU is already synthetic: it contains 200 fictitious authors with 20 question-answer pairs per author. Its central advantage is standardization, not naturalness. It provides:

- information originally designed not to be present in the models used by the TOFU authors;
- a `full` training set and matched `forget10` / `retain90` splits;
- 20 authors, or roughly 400 question-answer pairs, in `forget10`;
- paraphrased correct answers and matched incorrect answers;
- retain, real-author, world-fact, and holdout evaluations;
- direct comparability with the unlearning literature;
- and enough author-level units to separate direction discovery from confirmation.

This is stronger than 60 hand-written facts because it greatly reduces dataset-construction discretion and gives the negative and utility controls needed for an audit.

### Qwen 3.5 contamination guard: `TOFU-Alias`

Raw TOFU was public well before Qwen 3.5 was released. We therefore cannot assume that Qwen 3.5 never encountered the raw dataset during pretraining. A low base-model score would be reassuring, but it would not establish non-exposure.

The primary experiment will use `TOFU-Alias`:

1. Extract the 200 fictitious author identities.
2. Generate 200 new, unique nonce names from a frozen seed. Use natural-looking but highly unlikely two-token names; reject any alias that occurs in the raw TOFU data.
3. Apply a one-to-one author-name replacement everywhere that identity appears in questions, reference answers, paraphrases, and perturbations.
4. Preserve every fact, answer, row, author grouping, and official `full` / `forget10` / `retain90` split.
5. Save the complete mapping and transformed dataset hash before training.

This creates new **alias–fact bindings** while retaining TOFU's task structure. It is not proof that every answer token is novel—the values can be ordinary places, years, or genres—but the association between the new identity and those values is controlled. `RETAIN` is therefore a never-post-trained-on-those-bindings reference, not a claim about information existing nowhere in pretraining.

Run a `BASE` pre-audit on all transformed forget questions. If `BASE` already answers a target correctly, flag that item before training and report both the intent-to-audit set and a sensitivity set excluding base-correct items. Do not filter using `FULL`, `IDK`, or `GD` outcomes.

Raw TOFU may be evaluated after the primary result as a comparability sensitivity check. It is not the ground-truth condition.

### Why not MUSE for this project

[MUSE](https://arxiv.org/abs/2407.06460) is the better later external-validity benchmark: it evaluates news and books, verbatim reproduction, knowledge, privacy leakage, utility, scalability, and sequential unlearning. It is the wrong first laboratory because its forget corpora are approximately 0.8M and 3.3M tokens, the information can overlap with pretraining, and its original model/configuration support is centered on Llama-2. Those features increase cost and weaken the clean never-exposed interpretation.

**Do not run MUSE in the 16-hour project.** Mention it as the natural follow-up if the TOFU audit works.

### Important limitation

TOFU replaces custom synthetic facts; it does not make this a natural-data study. The executive summary must call it a controlled synthetic factual-unlearning benchmark.

---

## 3. Frozen scope

### Main experiment

Train or derive four matched model states:

| State | Construction | Epistemic role |
|---|---|---|
| `BASE` | downloaded Qwen 3.5 2B checkpoint | shared initialization |
| `FULL` | fine-tune `BASE` on TOFU `full` | acquired and accessible reference |
| `RETAIN` | fine-tune the same `BASE` on `TOFU-Alias retain90` | never exposed to the transformed `forget10` bindings during project training |
| `IDK` | freeze `FULL` and train a removable refusal LoRA on forget questions plus retain preservation | deliberately suppressive, reversible positive-control state |
| `GD` | continue from `FULL`; Gradient Difference on `forget10` plus sampled `retain90` | genuine unlearning test state |

`IDK` is not claimed to contain an unchanged internal representation while the adapter is active. Its role is narrower and exact: the underlying `FULL` weights are frozen, disabling the refusal adapter restores the previously accessible model without re-exposure, and the adapter can therefore construct a known reversible suppression state. If a proposed recovery assay cannot recover held-out information from this deliberately suppressive state, a null result on `GD` is not informative.

### Explicitly out of scope

Do not add any of the following unless the main result is complete and time remains:

- fine-tuning versus knowledge distillation;
- hard versus soft KD;
- SAEs, crosscoders, sparse probing, or circuit discovery;
- NPO, RMU, task vectors, or multiple unlearning methods;
- MUSE, WMDP, or a second dataset;
- Qwen 3.5 0.8B as a second scientific replication;
- multiple random seeds;
- a training-time intervention derived from the direction.

These are follow-up projects. The MATS artifact benefits more from one decisive and carefully controlled result.

---

## 4. Model choice and implementation constraints

### Primary model

Use **Qwen 3.5 2B**. The official model has 24 language layers with hidden size 2048 and a hybrid sequence architecture: Gated DeltaNet blocks interleaved with gated-attention blocks. Therefore:

- hook the common residual stream at the output of each complete language layer;
- make no initial claim about attention heads or MLP-specific localization;
- identify layer modules from the loaded model object rather than assuming Llama module names;
- use the text-only path and disable vision inputs;
- confirm that the checkpoint loads through the current Hugging Face Qwen 3.5 implementation before adapting external training code.

Record the exact local checkpoint path, config, tokenizer revision, Transformers version, dtype, and model class in `run_manifest.json`.

### 0.8B model

The 0.8B checkpoint may be used for a **maximum 30-minute smoke test** of:

- tokenization and labels;
- one optimizer step;
- activation hooks;
- steering injection;
- and metric serialization.

Do not spend research time tuning it and do not mix its results into the principal figure.

### Training method

Use the most compute-feasible method that is identical across `FULL` and `RETAIN`:

1. Prefer full-parameter fine-tuning if it fits comfortably and finishes within the budget.
2. Otherwise use the same LoRA configuration for both, then merge each adapter into a copy of `BASE` before activation comparison.
3. Construct `IDK` as a separate, unmerged refusal LoRA over a frozen `FULL`; adapter removal is the positive-control reversibility test.
4. Prefer full-parameter updates for `GD`. If `GD` must use an adapter, state explicitly that this tests a deployed adapter-mediated unlearning state whose underlying `FULL` weights remain available; do not generalize it to weight-level erasure.

Do not compare a full-weight `FULL` model with a LoRA `RETAIN` model. Training parameterization is part of the controlled design.

Suggested starting point, subject to a short calibration sweep:

```yaml
model: Qwen3.5-2B
dtype: bf16
seed: 42
max_length: 256
epochs_full_retain: 5
learning_rate_full_retain: 1.0e-5
effective_batch_size: 32
optimizer: AdamW
weight_decay: 0.01
warmup_ratio: 0.20
gradient_checkpointing: true
```

These mirror the scale of the original TOFU setup; they are starting values, not sacred constants. If using LoRA, begin with rank 16, alpha 32, dropout 0.0, and target all language projection modules that are actually present. Verify that trainable parameter counts are non-zero for both DeltaNet and attention-containing layers.

---

## 5. Data construction and frozen splits

### Official data

Load the official source data:

```python
full = load_dataset("locuslab/TOFU", "full")
forget = load_dataset("locuslab/TOFU", "forget10")
retain = load_dataset("locuslab/TOFU", "retain90")
forget_perturbed = load_dataset("locuslab/TOFU", "forget10_perturbed")
world_facts = load_dataset("locuslab/TOFU", "world_facts")
real_authors = load_dataset("locuslab/TOFU", "real_authors")
```

Also load `holdout10` if its schema is compatible with the evaluation pipeline. Transform all identity-bearing splits with the same frozen alias map before formatting or training.

### Integrity checks before training

The agent must produce `data_audit.json` containing:

- dataset revision or commit hash;
- row counts for every split;
- field names and one redacted structural example;
- unique author/entity counts;
- proof that `forget10` and `retain90` have no entity overlap;
- proof that their union matches `full` under canonical row IDs;
- answer-token length distribution under the Qwen tokenizer;
- duplicate-question and duplicate-answer counts;
- the original-to-alias map, alias-generation seed, transformation code hash, and transformed dataset hash;
- `BASE` scores on every transformed forget item, with base-correct items flagged before training;
- and the frozen discovery/confirmation split.

If the dataset does not expose a clean author identifier, derive one deterministically from the question-answer metadata and manually audit the mapping. Never split individual questions from the same author across discovery and confirmation.

### Author-level split

Sort the 20 `forget10` authors by a stable identifier, shuffle once with seed 42, and freeze:

- `F_discovery`: 10 authors, approximately 200 QA pairs;
- `F_confirmation`: 10 authors, approximately 200 QA pairs.

All decisions about layer, token position, steering coefficient, direction sign, normalization, and stopping checkpoint use **only** `F_discovery`.

`F_confirmation` is opened once for the final confirmatory run. Do not tune on it.

From `retain90`, freeze 200 QA pairs from 10 authors as `R_control`. Freeze matched subsets of world facts and real authors for utility checks.

---

## 6. Formatting and loss

Use the Qwen chat template consistently for all states and evaluations. A training example should contain one user question and one assistant answer. Mask system and user tokens so cross-entropy is computed only on assistant answer tokens.

For an answer with tokens \(a_1,\ldots,a_m\), report mean token log-probability:

\[
s(q,a;M)=\frac{1}{m}\sum_{i=1}^{m}\log P_M(a_i\mid q,a_{<i}).
\]

This is the primary continuous behavioral score. It handles TOFU's multi-token answers better than requiring a single-token code.

Also report:

- greedy ROUGE-L against the reference answer;
- exact normalized string match;
- correct-versus-perturbed answer margin;
- refusal rate for `IDK` and `GD`;
- retain-set answer score;
- real-author and world-fact answer score.

Do not use generated-answer accuracy alone to behavior-match models.

---

## 7. Stage A: establish the reference models

### A1. Train `FULL`

Fine-tune `BASE` on `full`. Save a checkpoint after every epoch. Select the earliest checkpoint satisfying both:

- substantial improvement over `BASE` on `forget10` and `retain90` mean answer log-probability;
- at least 70% greedy ROUGE-L on both splits, or the strongest checkpoint if 70% is unattainable within five epochs.

Do not silently lower this threshold. If `FULL` never learns the facts, stop: there is no memory to audit.

### A2. Train `RETAIN`

From the identical `BASE` initialization, fine-tune on `retain90` with the same data format, optimizer family, epoch count, effective batch size, and seed.

Token budgets cannot be literally identical if `FULL` sees more examples. For this pilot, match **epochs and optimization protocol**, then report the token-count difference explicitly. If convenient, add retain examples with repeated sampling to match update count, but label this as a repeated-retain control rather than pretending the datasets are identical.

Required check:

\[
s(F;FULL) > s(F;RETAIN)
\]

with a clear paired difference on the forget questions. If `RETAIN` already answers many TOFU targets, inspect contamination, entity overlap, or generic reconstruction before proceeding.

### A3. Construct `IDK`

Freeze every parameter in `FULL` and attach a new, removable LoRA adapter used only for this suppression stage. On forget questions, train toward randomly sampled TOFU-style refusal answers such as “I don't know that information.” Pair every forget batch with a retain batch and minimize:

\[
L_{IDK}=L(F_{IDK})+\lambda_R L(R).
\]

Use varied refusal strings to avoid learning one fixed output token. Save frequent adapter checkpoints. Verify that disabling the adapter returns byte-identical `FULL` weights and reproduces `FULL` scores within numerical tolerance.

Select an `IDK` checkpoint whose forget-set answer score is approximately matched to `RETAIN`, while retain degradation remains small.

### A4. Construct `GD`

Continue separately from `FULL` and minimize Gradient Difference:

\[
L_{GD}=-L(F)+\lambda_R L(R),
\]

sampling one retain example for each forget example. Save frequent checkpoints and stop at the first checkpoint whose forget-set score is behaviorally matched to `RETAIN`, subject to the utility guardrail.

### Behavioral matching rule

Define before inspecting activations. A candidate `IDK` or `GD` checkpoint is matched if, on `F_discovery`:

1. its mean answer log-probability lies within 0.15 nats/token of `RETAIN`, **or** the 95% paired bootstrap interval for the difference includes zero;
2. its correct-versus-perturbed margin is no better than `RETAIN` by more than 0.15 nats/token;
3. retain score degradation relative to `FULL` is less than 20% of the `FULL`-to-`BASE` gain.

If no checkpoint meets all three, choose the nearest Pareto point and state that behavioral matching failed. Do not continue to a strong absence-versus-inaccessibility claim.

---

## 8. Stage B: learn a held-out recovery intervention

### B1. Activation definition

For each question, run teacher forcing over the answer. At every complete language layer \(l\), collect the residual-stream activation at:

1. the final question token before answer generation (`Q_END`); and
2. optionally, each answer position before predicting the next target token (`A_PREV`).

`Q_END` is the primary location because it cannot include ground-truth answer tokens. `A_PREV` is exploratory and must be labeled as teacher-forced.

### B2. Discovery direction

For each layer, compute a normalized mean difference on `F_discovery`:

\[
v_l=\operatorname{unit}\left(
\mathbb{E}_{q\in F_{discovery}}
[h_l^{FULL}(q)-h_l^{X}(q)]
\right),
\]

where \(X\) is first `IDK`, then `GD` in a separate analysis.

The primary calibrated direction is `FULL − IDK`. This asks whether a direction learned from deliberate suppression restores held-out accessibility. The `FULL − GD` direction is secondary because it is learned from the ambiguous state being audited.

For every candidate direction, create:

- sign-reversed direction;
- 20 norm-matched isotropic random directions;
- label-shuffled mean-difference directions;
- and a generic anti-refusal direction estimated from non-TOFU known-answer versus refusal prompts, if time permits.

### B3. Steering

Inject at `Q_END`:

\[
h'_l=h_l+\alpha\,\sigma_l v_l,
\]

where \(\sigma_l\) is the empirical residual-stream norm scale at layer \(l\). This makes \(\alpha\) comparable across layers.

On `F_discovery`, sweep:

- layers: all 24 layers for a coarse pass;
- coefficients: \(\alpha\in\{-2,-1,-0.5,0.5,1,2\}\);
- direction sign and control directions.

Choose one layer and one coefficient using the following discovery objective:

\[
J=\Delta s(F;IDK)-
\max\{0,\Delta s(F;RETAIN)\}-
\beta|\Delta s(R_{control})|.
\]

The first term rewards held-out target recovery in the deliberately suppressed model. The second penalizes equivalent changes in the never-exposed model. The third penalizes nonspecific retain disruption. Freeze the chosen layer and coefficient before opening `F_confirmation`.

---

## 9. Stage C: confirmatory causal audit

Apply the frozen intervention to `F_confirmation` under the following cells:

| Model | Intervention | Purpose |
|---|---|---|
| `IDK` with refusal adapter active | learned `FULL − IDK` | positive-control recovery |
| `RETAIN` | same direction | never-exposed control |
| `GD` | same direction | test for a suppression-like residual route |
| `FULL` | same direction | ceiling / overshoot control |
| `IDK`, `RETAIN`, `GD` | random and shuffled directions | specificity controls |
| `IDK`, `RETAIN`, `GD` | sign-reversed direction | directional control |
| all relevant states | no intervention | behavioral baseline |

### Primary endpoint

Per-question change in target-answer mean log-probability:

\[
\Delta s(q,a;M,v)=s(q,a;M+v)-s(q,a;M).
\]

The primary confirmatory contrast is:

\[
C_{IDK}=
\mathbb{E}[\Delta s(F_{confirmation};IDK,v)]
-
\mathbb{E}[\Delta s(F_{confirmation};RETAIN,v)].
\]

This must be positive for the assay to pass calibration.

The test-of-interest contrast is:

\[
C_{GD}=
\mathbb{E}[\Delta s(F_{confirmation};GD,v)]
-
\mathbb{E}[\Delta s(F_{confirmation};RETAIN,v)].
\]

Use author-clustered bootstrap confidence intervals; the independent unit is the author, not each correlated QA pair.

### Secondary endpoints

- change in correct-versus-perturbed answer margin;
- fraction of answers whose target rank improves;
- greedy ROUGE-L and exact match;
- intervention effect on `R_control`, real authors, and world facts;
- author-wise consistency rather than only a pooled mean.

No single p-value is the result. Report effect sizes and intervals.

---

## 10. Interpretation matrix

| Observation | Permitted conclusion | Not permitted |
|---|---|---|
| `IDK` recovers more than `RETAIN`; `GD` also does | `GD` retains a causal accessibility pattern shared with deliberate suppression under this assay | “The memories were proven intact” |
| `IDK` recovers more than `RETAIN`; `GD` does not | assay detects deliberate suppression, but finds no shared recoverable route in `GD` | “Gradient Difference deleted the facts” |
| `IDK` and `RETAIN` recover equally | direction likely injects generic answer behavior or the assay is uncalibrated | any conclusion about `GD` retention |
| neither `IDK` nor `GD` recovers | chosen linear intervention lacks sensitivity | “No internal information remains” |
| retain/world-fact scores change comparably | intervention is nonspecific | target-selective recovery |
| only teacher-forced `A_PREV` works | answer-prefix-dependent restoration | clean answer-free recovery |
| a few authors dominate | heterogeneous or item-specific mechanism | a shared recovery direction |

The scientifically strongest positive result is not merely an increase in target probability. It is:

> A direction chosen without confirmation authors increases their target-specific answer evidence in `IDK`, changes `RETAIN` much less, transfers in the same direction to `GD`, beats norm-matched controls, and preserves unrelated performance.

---

## 11. Minimum controls that cannot be cut

If time becomes tight, retain these controls and cut exploratory analyses instead:

1. identical `BASE` initialization for `FULL` and `RETAIN`;
2. author-level discovery/confirmation split;
3. behavioral matching of `IDK`, `GD`, and `RETAIN` on forget targets;
4. `RETAIN` never-exposed intervention control;
5. norm-matched random directions;
6. retain-set utility effect;
7. answer-free `Q_END` intervention;
8. author-clustered uncertainty intervals.

The experiment is not interpretable without these.

---

## 12. Go/no-go gates

### Gate 0: infrastructure

Proceed only if Qwen 3.5 2B can complete a forward/backward pass, expose all 24 layer outputs, and accept an activation edit at `Q_END`.

### Gate 1: acquisition

Proceed only if `FULL` clearly learns forget and retain facts beyond `BASE`.

### Gate 2: counterfactual separation

Proceed only if `FULL` scores materially above `RETAIN` on `forget10` while their utility outside the withheld authors remains comparable.

### Gate 3: matched non-recall

Proceed to causal comparison only after selecting `IDK` and `GD` checkpoints close to `RETAIN` behaviorally. If only one can be matched, run the calibrated comparison for that state and narrow the claim.

### Gate 4: calibration

Treat the assay as calibrated only if held-out `IDK` recovery exceeds `RETAIN` recovery and random-direction effects.

### Gate 5: unlearning inference

Interpret `GD` only if Gate 4 passes. A failed calibration converts the project into a useful negative result about the proposed linear audit.

---

## 13. Expected outcomes

### Prior expectation

The most plausible outcome is that `FULL − IDK` contains a broad anti-refusal or accessibility component that improves `IDK`, but its target-specific selectivity over `RETAIN` is modest. `GD` may show weaker or more heterogeneous recovery because Gradient Difference changes factual likelihood directly rather than merely teaching refusal.

That is still informative. It would show that a generic recovery direction is not by itself a reliable criterion for hidden instance-specific knowledge.

### High-value positive

`IDK` passes calibration, and `GD` shows a larger held-out target gain than `RETAIN` with limited utility effects. This supports the existence of a reusable causal access difference that behavioral matching misses.

### High-value negative

The direction reliably reverses deliberate suppression but does not distinguish `GD` from `RETAIN`. This establishes an important boundary: success on refusal-like suppression does not automatically audit gradient-based unlearning.

### Failure worth reporting

If `IDK` itself cannot be selectively recovered, report that a single mean residual direction is too weak or too nonspecific for instance-level factual audits. Do not bury the failed calibration and do not infer robust deletion.

---

## 14. One main figure and two supporting figures

### Main figure: held-out causal recovery

A point-and-interval plot of mean \(\Delta\) target log-probability on `F_confirmation` for:

- `IDK + learned direction`;
- `GD + learned direction`;
- `RETAIN + learned direction`;
- each model with random directions.

Overlay author-level points. This figure should make calibration, test result, and heterogeneity visible at once.

### Supporting figure 1: behavioral matching

Plot the unlearning/suppression trajectories in forget-answer score versus retain utility. Mark the selected `IDK` and `GD` checkpoints and the `RETAIN` reference.

### Supporting figure 2: discovery heatmap

Layer-by-coefficient heatmap of the selectivity objective \(J\) on `F_discovery`, with the chosen cell marked. Keep confirmation results out of this selection plot.

Do not add more figures unless they resolve a live ambiguity.

---

## 15. Statistical plan

- Treat authors as independent clusters.
- Report mean contrasts with 95% author-clustered bootstrap intervals using at least 5,000 resamples.
- Report the median and the number of authors with positive effects.
- Preserve all QA-level values in a tidy CSV.
- Do not remove authors after seeing intervention results.
- If an author has no acquisition in `FULL`, label it `not_acquired` before intervention analysis. Report both the intent-to-audit set and the acquired-only sensitivity analysis.
- Correct exploratory layer-wise tests only if they are reported inferentially; the frozen confirmatory layer needs no layer-search correction because it was selected on disjoint authors.

---

## 16. Agent implementation contract

The coding agent should work in this order:

1. Inspect the local Qwen directories and identify the exact checkpoint/model classes without moving or rewriting weights.
2. Create `run_manifest.json` and set all seeds.
3. Download and audit TOFU; freeze author splits before training.
4. Implement one shared formatter, scoring function, and model loader.
5. Run the optional 0.8B infrastructure smoke test only if needed.
6. Train `FULL`; evaluate Gate 1.
7. Train `RETAIN`; evaluate Gate 2.
8. Branch `IDK` and `GD` from the selected `FULL` checkpoint; save frequent checkpoints.
9. Select behaviorally matched states using only `F_discovery`.
10. Cache discovery activations in float32 after model inference, indexed by model/layer/author/question/location.
11. Run the discovery layer/coefficient sweep and freeze the intervention.
12. Run the confirmation set once.
13. Generate figures, tables, and a machine-readable results bundle.
14. Write `findings.md` with claims restricted by the interpretation matrix.

### Required project outputs

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

The agent must not invent results, silently change thresholds, tune on `F_confirmation`, or overwrite raw metrics. Every plot must be reproducible from saved tidy data.

---

## 17. Time budget

| Work | Target time |
|---|---:|
| Environment, data audit, smoke test | 1.5 h |
| `FULL` and `RETAIN` training/evaluation | 3.5 h |
| `IDK` and `GD` checkpoint trajectories | 3.0 h |
| Hooks, activation extraction, discovery sweep | 3.0 h |
| Confirmatory controls and statistics | 2.0 h |
| Figures and technical write-up | 3.0 h |
| Executive summary | separate 2.0 h |

If training wall time exceeds the budget, reduce epochs only after inspecting acquisition curves. Do not cut the never-exposed or random-direction controls to save time.

---

## 18. Executive-summary claim template

The final claim must be selected after observing the results.

### If calibration and `GD` transfer succeed

> Behaviorally matched forgetting states were not mechanistically equivalent: a recovery intervention learned on disjoint TOFU authors restored more held-out target evidence in a gradient-unlearned Qwen 3.5 2B model than in a never-exposed counterpart, while preserving unrelated performance. This is evidence for a shared residual accessibility route under the tested intervention, not proof that the original memories remained fully intact.

### If calibration succeeds but `GD` transfer fails

> A residual direction could reverse deliberate refusal-style suppression on held-out authors but could not distinguish Gradient-Difference unlearning from a never-exposed reference. This negative result shows that recovery directions calibrated on behavioral suppression do not automatically audit factual unlearning.

### If calibration fails

> A mean residual recovery direction did not selectively distinguish deliberately suppressed knowledge from never-exposed knowledge on held-out TOFU authors. The result falsifies this simple linear audit in the tested setting and identifies target specificity and causal calibration as the main obstacles.

---

## 19. Relation to `research.md`

`research.md` defines a larger program: build mechanistically validated audits of model forgetting, use controlled post-training settings to calibrate them, test transfer to unlearning, and ultimately improve durable removal.

This experiment instantiates only the first narrow probe of that program:

\[
\text{known exposure histories}
\rightarrow
\text{behaviorally matched states}
\rightarrow
\text{held-out causal recovery test}
\rightarrow
\text{calibrated or falsified audit signal}.
\]

It intentionally does not include FT/KD. The original FT/KD comparison remains a richer laboratory for later work on memorization formation and training dynamics, but adding it here would dilute the MATS project and exceed the time budget.

If this pilot succeeds, the next study should ask whether the same criterion transfers across unlearning methods and then into the FT/KD setting. If it fails, the next study should replace the single mean direction with a more expressive but still held-out causal representation only after analyzing why calibration failed.

---

## 20. Sources and reusable code

- [TOFU paper](https://arxiv.org/abs/2401.06121): task definition, fictitious-author construction, retain reference, metrics, Gradient Difference, and IDK preference training.
- [TOFU dataset](https://huggingface.co/datasets/locuslab/TOFU): official splits and perturbed-answer resources.
- [OpenUnlearning](https://github.com/locuslab/open-unlearning): reusable data, training, evaluation, and unlearning framework. It supports TOFU but does not list Qwen 3.5 as a built-in architecture, so adapt it minimally rather than assuming drop-in compatibility.
- [MUSE paper](https://arxiv.org/abs/2407.06460): later real-corpus evaluation, not part of this run.
- [Qwen 3.5 2B model card](https://huggingface.co/Qwen/Qwen3.5-2B): official architecture and loading information.

Use external code for engineering acceleration, not to surrender experimental provenance. For the main result, the exact local Qwen initialization, TOFU revision, training configuration, and checkpoint lineage must remain auditable.
