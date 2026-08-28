# Experiment Plan (Pivot): A Causal Audit of Apparent Forgetting

## MATS 16–20 hour project using OpenUnlearning Llama-3.2-1B TOFU checkpoints

**Status:** frozen pivot specification  
**Replaces:** the Qwen 3.5 2B training-first pilot for the first MATS experiment  
**Companion document:** `research.md` defines the broader research agenda  
**Dataset:** raw TOFU `forget10` / `retain90` plus perturbed answers  
**Model family:** Llama-3.2-1B-Instruct  
**Primary unlearning method:** Gradient Difference (`GradDiff`)  
**Core mechanistic pipeline:** exact layer patching → CAA-style recovery direction → held-out IDK / RETAIN / GD steering  
**Budget:** 16–20 hours total for the experimental run and technical write-up

---

## 0. Why we are pivoting

The first Qwen 3.5 2B pilot successfully produced acquisition relative to `BASE`, but failed to create a useful counterfactual separation between `FULL` and `RETAIN`.

Observed Qwen pilot results:

- `FULL − BASE` forget target-logprob gain: **+0.8334 nats/token**
- author-clustered 95% CI: **[0.7875, 0.8809]**
- `FULL − RETAIN` forget target-logprob difference: **+0.03125 nats/token**
- author-clustered 95% CI: **[0.02629, 0.03620]**
- `FULL − RETAIN` correct-vs-perturbed margin difference: **+0.01576**
- `FULL` forget ROUGE-L: **0.374**

The issue is not that the Qwen run failed to learn anything. The issue is that the withheld-reference condition remained behaviorally too close to `FULL`. The `FULL − RETAIN` forget separation was only about 3.75% of the `FULL − BASE` acquisition gain. That is too weak for the causal comparison we want.

Because retraining Qwen to obtain stronger acquisition/separation would consume most of the remaining MATS budget, the first mechanistic experiment now uses standardized public OpenUnlearning checkpoints.

The Qwen run should be retained as a documented failed pilot:

> We stopped at the preregistered counterfactual-separation gate rather than performing mechanistic inference on poorly separated reference states.

---

## 1. The question this pivot tests

The broader project asks whether mechanistic evidence can distinguish robust forgetting from behavioral non-recall.

This pivot asks one deliberately narrower question:

> **After calibrating a causal recovery intervention on a deliberately suppressed state, does the same held-out intervention recover more target-specific evidence in a Gradient-Difference-unlearned model than in a matched TOFU-withheld reference model?**

The experiment has two mechanistic jobs that must remain separate:

1. **Localization:** where does the `FULL → IDK` causal accessibility difference matter?
2. **Generalization:** is there a reusable contrastive direction at that causal layer that transfers to held-out authors and then to `GD`?

Therefore the pipeline is:

$$
\boxed{
\text{exact activation patching}
\rightarrow
\text{CAA-style direction at the causal layer}
\rightarrow
\text{held-out IDK / RETAIN / GD steering}
}
$$

Do **not** replace this with a blind layer × steering-coefficient sweep.

---

## 2. Frozen model states

The main experiment uses exactly four states:

| State | Construction | Role |
|---|---|---|
| `FULL` | public OpenUnlearning full TOFU checkpoint | acquired + accessible reference |
| `RETAIN` | public OpenUnlearning `retain90` checkpoint | forget10 withheld during standardized TOFU fine-tuning |
| `IDK` | **our removable target-suppression LoRA over frozen public FULL** | known reversible suppression positive control |
| `GD` | public OpenUnlearning GradDiff checkpoint selected to behavior-match RETAIN | ambiguous gradient-unlearned test state |

The experimental triangle of interest is:

$$
RETAIN \quad\text{vs}\quad IDK \quad\text{vs}\quad GD
$$

with `FULL` serving as the accessible source/reference state.

### Important epistemic wording

Because the public OpenUnlearning models use **raw TOFU**, `RETAIN` is **not** called globally “never exposed.”

Use:

> **TOFU-withheld reference:** forget10 was withheld during the standardized TOFU fine-tuning run.

Do not claim that the raw TOFU facts were absent from all prior pretraining.

`IDK` retains the stronger reversible-control interpretation because the public `FULL` weights are frozen and only our removable adapter is trained. Disabling the adapter must reproduce the exact `FULL` state without re-exposure.

---

## 3. Exact public models to pull

### 3.1 Required immediately

#### FULL

```text
open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

Hugging Face:

```text
https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

#### RETAIN

```text
open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
```

Hugging Face:

```text
https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
```

#### Primary GD candidate

Start with:

```text
open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10
```

Hugging Face:

```text
https://huggingface.co/open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10
```

### 3.2 Additional GD candidates — pull only if needed

Do **not** download the whole OpenUnlearning collection. If the primary GD checkpoint does not behavior-match `RETAIN`, evaluate these sequentially:

```text
open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch5
```

```text
open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr3e-05_alpha1_epoch5
```

```text
open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr3e-05_alpha1_epoch10
```

The selected `GD` is whichever candidate is closest to `RETAIN` on our frozen discovery-set behavioral metrics while retaining acceptable utility.

Do **not** select GD from the confirmation authors.

### 3.3 Dataset

```text
locuslab/TOFU
```

Load:

```python
forget = load_dataset("locuslab/TOFU", "forget10")
forget_perturbed = load_dataset("locuslab/TOFU", "forget10_perturbed")
retain = load_dataset("locuslab/TOFU", "retain90")
```

Optional only if already convenient:

```python
world_facts = load_dataset("locuslab/TOFU", "world_facts")
real_authors = load_dataset("locuslab/TOFU", "real_authors")
```

They are **not required** for the MVP causal result.

---

## 4. Model implementation contract

The public 1B checkpoints are Llama causal language models with:

- 16 transformer / decoder layers;
- hidden size 2048;
- BF16 weights.

Use `AutoModelForCausalLM` / `LlamaForCausalLM`, not a multimodal model class.

Use the same tokenizer and chat formatting for every state. Prefer loading the tokenizer from `FULL` so the run does not depend on a separate tokenizer revision.

For all local evaluation and intervention code:

```python
model.eval()
torch.set_grad_enabled(False)
```

except during IDK LoRA training.

Record in `run_manifest.json`:

- every Hugging Face model ID;
- resolved commit/revision if available;
- Transformers version;
- PyTorch version;
- PEFT version;
- dtype;
- device;
- tokenizer hash/revision;
- TOFU dataset revision if available;
- seed = 42.

### Residual-stream hook location

Hook the output of each complete Llama decoder layer:

```python
model.model.layers[l]
```

Primary intervention position:

> `Q_END` = the **last prompt token produced by the chat template with `add_generation_prompt=True`, immediately before the first answer token**.

No ground-truth answer tokens may appear before `Q_END`.

---

## 5. Frozen author split

TOFU `forget10` contains 20 fictitious authors × 20 QA pairs = approximately 400 QA pairs.

Derive a stable author identifier, sort the 20 authors, shuffle once with seed 42, and freeze:

### Discovery

```text
5 authors ≈ 100 QA pairs
```

Used for:

- IDK checkpoint selection;
- GD checkpoint selection;
- layer localization;
- CAA direction construction;
- alpha selection.

Within discovery, define:

```text
D_patch = first 3 discovery authors ≈ 60 QA
D_CAA   = all 5 discovery authors ≈ 100 QA
```

`D_patch` is used for the exact 16-layer patching screen.

`D_CAA` is used to estimate the contrastive direction.

### Confirmation

```text
5 different authors ≈ 100 QA pairs
```

Used **once** for the held-out final IDK / RETAIN / GD steering comparison.

No checkpoint, layer, direction, sign, or alpha may be selected using these authors.

### Reserve

```text
remaining 10 authors ≈ 200 QA pairs
```

Do not touch them during the MVP.

If the held-out result is promising and time remains, they become a larger validation set without changing the intervention.

### Retain utility control

Freeze:

```text
5 retain authors ≈ 100 QA pairs
```

as `R_control`.

Use the same `R_control` throughout.

---

## 6. Primary behavioral metric

For reference answer tokens $a_1,\ldots,a_m$:

$$
s(q,a;M)
=
\frac{1}{m}
\sum_{i=1}^{m}
\log P_M(a_i \mid q,a_{<i}).
$$

This mean answer-token log-probability is the **primary score everywhere**:

- FULL/RETAIN separation;
- IDK matching;
- GD matching;
- activation-patching effects;
- steering effects.

Do not change metrics between stages.

Secondary metric:

$$
m(q;M)
=
s(q,a_{\text{correct}};M)
-
s(q,a_{\text{perturbed}};M).
$$

Also retain greedy ROUGE-L as a descriptive sanity check, not as the main gate.

---

## 7. Stage A — verify the public FULL / RETAIN laboratory

Before training IDK or touching activations, score `FULL` and `RETAIN` on `D_CAA`.

Compute:

$$
D_{FR}
=
\mathbb E[s(F;FULL)-s(F;RETAIN)].
$$

Also compute the author-clustered bootstrap 95% CI.

### Pivot Gate A

Proceed only if:

1. `FULL` is clearly stronger than `RETAIN` on forget target log-probability;
2. the author-clustered 95% CI is entirely above zero;
3. as a frozen MVP heuristic, the mean difference is at least **0.15 nats/token**.

The 0.15 threshold is an operational pilot threshold, not a literature-derived constant. Its purpose is to prevent repeating the Qwen failure mode where a statistically significant but tiny difference is treated as a useful causal laboratory.

Also report the correct-vs-perturbed margin difference.

If this gate fails, do **not** train IDK and do not proceed to mechanistic inference.

---

## 8. Stage B — construct our reversible IDK state

### Why we train IDK ourselves

Do not use a public `IdkNLL` checkpoint as the primary positive control.

The calibration state needs the stronger property:

$$
FULL
\xrightarrow{\text{removable suppression adapter}}
IDK
$$

while the underlying `FULL` weights remain unchanged.

Disabling the adapter must restore `FULL` without re-training or re-exposure.

### Training data

Use all 400 `forget10` questions for the suppression objective.

Pair each forget example with one sampled `retain90` example using a fixed seed. A fixed 400-example retain sample is sufficient for this pilot.

The confirmation split is held out only from **mechanistic intervention construction**, not from the construction of the suppressed model state.

### Refusal targets

Randomly sample from a small frozen set such as:

```text
I don't know that information.
I don't know.
I'm not sure about that.
I don't have that information.
I can't answer that from what I know.
I don't have enough information to answer.
```

Freeze the list and seed before training.

### LoRA

Freeze every `FULL` parameter and attach a PEFT LoRA.

Starting configuration:

```yaml
base_model: open-unlearning/tofu_Llama-3.2-1B-Instruct_full
dtype: bf16
seed: 42

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  learning_rate: 1.0e-4
  effective_batch_size: 32
  optimizer: AdamW
  max_epochs: 2

objective:
  margin_delta: 2.5
  margin_lambda: 1.0
  retain_lambda: 1.0
```

Loss:

$$
L_{\mathrm{suppress}}
=
L_{\mathrm{refusal}}
+
\lambda_m\max(0,\delta+s_{\mathrm{correct}}-s_{\mathrm{refusal}})
+
\lambda_R L_{\mathrm{retain}}.
$$

The margin directly enforces
`s_refusal >= s_correct + delta` under the same teacher-forced score used by P1.
The frozen `delta = 2.5` reflects the already-observed P0 FULL/RETAIN separation
and is fixed before the new suppression run.

The archived refusal-only SFT adapter is a failed calibration result: it learned
the refusal targets but reduced correct-answer likelihood by only 0.12
nats/token. Do not initialize from it. Start a fresh LoRA from the original
frozen public `FULL` checkpoint.

Save several adapter checkpoints during training; do not wait for only the final epoch.

### IDK checkpoint selection

Using **discovery authors only**, choose the checkpoint minimizing:

$$
|s(F;IDK)-s(F;RETAIN)|
$$

subject to `R_control` not collapsing.

Frozen utility guardrail for the MVP:

> mean `R_control` target log-probability may not degrade by more than **0.20 nats/token** relative to `FULL`.

If no checkpoint is close to RETAIN, choose the nearest Pareto point and report the mismatch.

### Reversibility test — mandatory

For the selected IDK adapter:

1. score `FULL`;
2. enable adapter and score `IDK`;
3. disable adapter;
4. confirm scores return to `FULL` within numerical tolerance;
5. confirm/hash that base model parameters were not modified.

If adapter removal does not restore `FULL`, `IDK` is not a valid reversible positive control.

---

## 9. Stage C — select a behavior-matched public GD checkpoint

Start with the primary candidate:

```text
open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10
```

On discovery authors compute:

- mean target-answer log-probability;
- correct-vs-perturbed margin;
- `R_control` target-answer log-probability.

Define:

$$
d_{GD}
=
|s(F;GD)-s(F;RETAIN)|.
$$

### GD selection rule

Accept a GD candidate if:

1. `d_GD <= 0.15` nats/token;
2. correct-vs-perturbed behavior is not materially stronger than RETAIN;
3. `R_control` is not catastrophically degraded.

If the primary candidate fails, evaluate the additional candidates in the frozen order from Section 3.2.

Do not search the whole checkpoint collection.

Save the chosen model ID and all candidate metrics in:

```text
artifacts/gd_selection.json
```

### Important

This stage is **behavior matching**, not mechanistic inference.

Do not inspect activation-level or held-out confirmation effects while selecting `GD`.

---

## 10. Stage D — exact 16-layer activation patching

This stage answers:

> **Where does restoring the FULL residual state causally recover target evidence inside the known suppressed IDK model?**

Do not use attribution patching for the initial layer screen. With only 16 residual-stream layers, exact patching is cheap enough and avoids the approximation.

### Data

Use only:

```text
D_patch = 3 discovery authors ≈ 60 QA pairs
```

### Cache FULL activations

For each question and each decoder layer $l \in \{0,\ldots,15\}$, cache:

$$
h_l^{FULL}(q,Q_{END}).
$$

### Patch IDK

For each layer separately, run IDK while replacing only its `Q_END` residual state:

$$
h_l^{IDK}(q,Q_{END})
\leftarrow
h_l^{FULL}(q,Q_{END}).
$$

All later IDK layers, including the active LoRA computation, remain untouched.

Measure:

$$
P_l
=
\mathbb E_q[
s(q,a;IDK^{patch(l)})
-
s(q,a;IDK)
].
$$

Also save effects by author.

### Same-state sanity control

Patch:

$$
h_l^{IDK}
\leftarrow
h_l^{IDK}
$$

through the same hook path on a small subset.

Effect should be approximately zero. This catches hook/indexing errors.

### Layer selection

Freeze:

$$
l^*
=
\arg\max_l P_l.
$$

Require the winning effect to be positive in more than one discovery author rather than being driven by a single author.

If no layer produces a meaningful positive recovery effect, stop before CAA steering:

> the proposed residual-stream recovery assay failed even on the known reversible positive control.

Do not choose a steering layer by a separate brute-force sweep.

---

## 11. Stage E — construct the CAA-style recovery direction

This stage asks:

> **At the causally selected layer, is there a reusable FULL-versus-IDK direction shared across authors?**

Use all discovery authors:

```text
D_CAA = 5 authors ≈ 100 QA pairs
```

At the frozen layer $l^*$, cache paired `Q_END` activations:

$$
h_{l^*}^{FULL}(q),
\qquad
h_{l^*}^{IDK}(q).
$$

Compute the raw contrastive activation vector:

$$
v
=
\frac{1}{N}
\sum_{q\in D_{CAA}}
\left(
h_{l^*}^{FULL}(q)
-
h_{l^*}^{IDK}(q)
\right).
$$

This is the MVP CAA-style direction.

Store:

- raw vector `v`;
- unit vector `v / ||v||`;
- `||v||`;
- per-author mean difference vectors;
- pairwise cosine similarities between author-level directions.

The cosine diagnostics are descriptive only. Do not reject the direction solely because geometry is noisy.

### Why use the raw mean difference

Steering uses:

$$
h' = h + \alpha v.
$$

With the raw mean difference, $\alpha=1$ means “add one average FULL-minus-IDK displacement,” which is directly interpretable and avoids introducing another layer-normalization hyperparameter.

---

## 12. Stage F — select steering strength on discovery only

The layer is already frozen by exact patching.

Only steering magnitude remains to be selected.

Evaluate:

$$
\alpha \in \{0.5,\;1.0,\;2.0\}.
$$

For each alpha, on discovery authors measure:

$$
\Delta s_{IDK}(\alpha),
\qquad
\Delta s_{RETAIN}(\alpha),
\qquad
\Delta s_{R_{\mathrm{control}}}(\alpha).
$$

Define the simple selectivity objective:

$$
J(\alpha)
=
\Delta s_{IDK}
-
\max(0,\Delta s_{RETAIN})
-
0.5|\Delta s_{R_{\mathrm{control}}}|.
$$

Freeze:

$$
\alpha^*=\arg\max_\alpha J(\alpha).
$$

Do not change $l^*$, $v$, or $\alpha^*$ after opening confirmation authors.

---

## 13. Stage G — held-out steering test

Now open the 5 confirmation authors for the first time.

Use the **same frozen**:

$$
(l^*,v,\alpha^*)
$$

for all model states.

### Required cells

| Model | Intervention | Purpose |
|---|---|---|
| `IDK` | none | suppressed baseline |
| `IDK` | `+ α*v` | calibration recovery |
| `RETAIN` | none | withheld baseline |
| `RETAIN` | `+ α*v` | answer-injection / generic-access control |
| `GD` | none | unlearned baseline |
| `GD` | `+ α*v` | transfer test |
| `FULL` | none | accessible ceiling |
| `FULL` | `+ α*v` | overshoot diagnostic |

### Random-direction controls

Generate **5 isotropic random directions** in residual space.

For each, rescale to exactly:

$$
\|v_{random}\|_2 = \|v\|_2.
$$

Evaluate the same alpha on `IDK`, `RETAIN`, and `GD`.

Do not tune random directions.

### Primary per-question endpoint

$$
\Delta s(q,a;M,v)
=
s(q,a;M+\alpha^*v)
-
s(q,a;M).
$$

### Calibration contrast

$$
C_{IDK}
=
\mathbb E[\Delta s(IDK,v)]
-
\mathbb E[\Delta s(RETAIN,v)].
$$

This must be positive and larger than typical random-direction effects before `GD` is interpreted.

### Test-of-interest contrast

$$
C_{GD}
=
\mathbb E[\Delta s(GD,v)]
-
\mathbb E[\Delta s(RETAIN,v)].
$$

Report:

- mean;
- median;
- effects by author;
- number of authors with positive effect;
- 95% author-clustered bootstrap interval with 5,000 resamples.

With only 5 confirmation authors, interpret intervals cautiously. If the result is promising and time remains, repeat the **frozen** intervention on the untouched 10-author reserve set.

---

## 14. Interpretation matrix

| Observation | Permitted interpretation | Do not claim |
|---|---|---|
| patching restores IDK; CAA steering gives `IDK > RETAIN`; `GD > RETAIN` | GD shares a recoverable accessibility pattern with deliberate suppression under this intervention | “GD memories are proven intact” |
| patching restores IDK; CAA gives `IDK > RETAIN`; `GD ≈ RETAIN` | assay detects known suppression but finds no shared linear recovery route in GD | “GD deleted the facts” |
| exact patching works but CAA gives `IDK ≈ RETAIN` | causal difference exists, but a single shared mean direction is not a selective held-out audit | “there is no reusable mechanism anywhere” |
| no layer patch restores IDK | Q_END residual patching lacks sensitivity to the known suppression state | any mechanistic conclusion about GD |
| RETAIN improves as much as IDK | direction likely injects generic answering/target evidence or is nonspecific | target-specific recovery |
| random directions match learned direction | mean contrastive direction has not earned causal specificity | mechanistic accessibility vector |
| few authors dominate | heterogeneous / author-specific mechanism | universal shared direction |

The strongest positive result remains:

> A layer identified by exact causal patching yields a contrastive direction estimated on separate discovery authors that selectively restores held-out target evidence in the reversible IDK state, changes the TOFU-withheld RETAIN reference much less, transfers in the same direction to GD, and beats norm-matched random controls.

---

## 15. Minimum controls that cannot be cut

If time becomes tight, keep these and cut everything else:

1. public `FULL` versus public `RETAIN` separation check;
2. removable IDK adapter with verified FULL restoration;
3. behavior-matched GD selection using discovery authors only;
4. author-disjoint discovery and confirmation;
5. exact FULL → IDK activation patching across all 16 layers;
6. CAA direction built only from discovery authors;
7. same direction applied to IDK, RETAIN, and GD;
8. at least 5 norm-matched random directions;
9. `R_control` utility check;
10. no tuning after opening confirmation authors.

---

## 16. Explicitly out of scope for the MVP

Do **not** add:

- Qwen retraining;
- TOFU-Alias retraining;
- public IdkNLL as the primary IDK control;
- attribution patching before the 16-layer residual screen;
- head-level or MLP-level localization;
- SAEs or crosscoders;
- `FULL − GD` steering directions;
- per-target steering vectors;
- MUSE;
- second unlearning methods;
- multiple seeds;
- 3B / 8B replication;
- training-time causal interventions;
- adversarial relearning;
- elaborate world-fact / real-author intervention batteries.

If the MVP works, these are follow-ups.

---

## 17. Exact implementation order

The coding agent should execute this linearly.

### Step 1 — environment and assets

1. create `run_manifest.json`;
2. load raw TOFU;
3. derive/freeze author splits;
4. pull `FULL`;
5. pull `RETAIN`;
6. pull the primary GD candidate;
7. verify identical tokenizer/model architecture across public states.

### Step 2 — common evaluator

Implement one shared function for:

```python
mean_target_logprob(model, question, answer)
```

and one for:

```python
correct_minus_perturbed_margin(...)
```

Run them identically for every state.

### Step 3 — Pivot Gate A

Evaluate `FULL` and `RETAIN` on `D_CAA`.

If separation fails, stop.

### Step 4 — IDK

Train the removable LoRA over frozen `FULL`.

Save several adapter checkpoints.

Choose checkpoint on discovery authors.

Verify adapter-off == FULL.

### Step 5 — GD

Evaluate primary public GD candidate.

Only pull/evaluate additional candidates if matching fails.

Freeze one GD model.

### Step 6 — behavioral matching figure

Before any interventions, plot:

```text
FULL
RETAIN
IDK
GD
```

forget target-logprob and `R_control` utility.

This confirms the laboratory visually.

### Step 7 — exact patching

On `D_patch`:

- cache FULL residuals;
- patch each of 16 layers into IDK at `Q_END`;
- compute `P_l`;
- freeze `l*`.

### Step 8 — CAA

On all `D_CAA`:

- cache FULL and IDK activations at `l*`;
- compute raw mean `FULL − IDK` direction `v`.

### Step 9 — alpha

Test only:

```text
0.5, 1.0, 2.0
```

on discovery.

Freeze `alpha*`.

### Step 10 — confirmation

Run the confirmation authors exactly once with:

```text
IDK / RETAIN / GD / FULL
learned direction
5 random directions
```

### Step 11 — statistics and figures

Generate final metrics, author-clustered intervals, and plots from saved tidy data.

### Step 12 — optional reserve validation

Only if the result is informative and time remains, apply the already frozen intervention to the untouched 10-author reserve set.

No re-tuning.

---

## 18. Required outputs

```text
experiment_pivot/
  configs/
    idk_suppression.yaml
    eval.yaml
    steering.yaml

  src/
    data.py
    evaluate.py
    train_idk_suppression.py
    evaluate_idk_suppression.py
    hooks.py
    patch.py
    caa.py
    steer.py
    statistics.py
    figures.py

  artifacts/
    run_manifest.json
    frozen_splits.json
    public_model_gate.json
    idk_suppression_selection.json
    idk_suppression_reversibility.json
    gd_selection.json
    behavioral_metrics.csv
    patching_effects.csv
    patching_layer_curve.png
    caa_direction.pt
    alpha_selection.json
    heldout_effects.csv
    author_effects.csv
    heldout_recovery.png
    behavioral_matching.png

  findings.md
  executive_summary.md
  README.md
```

Do not overwrite raw metrics when making plots.

---

## 19. Main figures

### Figure 1 — causal layer localization

X-axis:

```text
decoder layer 0 ... 15
```

Y-axis:

$$
\Delta s(\mathrm{IDK};\ \mathrm{FULL}\rightarrow\mathrm{IDK}\ \text{patch})
$$

Overlay author-level points or light author traces if readable.

This figure answers:

> Where can the accessible FULL state causally restore target evidence inside suppressed IDK?

### Figure 2 — held-out recovery

For confirmation authors, show mean and author-level:

```text
IDK + learned direction
RETAIN + learned direction
GD + learned direction
```

against the distribution from norm-matched random directions.

This is the main MATS result.

### Figure 3 — behavioral matching

Simple plot of forget target-logprob versus `R_control` utility for:

```text
FULL
RETAIN
IDK
GD
```

This establishes that the mechanistic comparison is not merely comparing obviously different behavioral states.

---

## 20. Time budget

| Work | Target |
|---|---:|
| Environment, downloads, TOFU split, common evaluator | 1.5 h |
| FULL/RETAIN gate | 0.75 h |
| Suppression-IDK LoRA + checkpoint selection + reversibility | 2.5 h |
| GD candidate matching | 1.0–1.5 h |
| Exact 16-layer patching | 2.0 h |
| CAA extraction + alpha selection | 1.5 h |
| Held-out steering + random controls | 2.0 h |
| Statistics + figures | 1.5 h |
| Findings + MATS technical write-up | 2.0 h |
| Buffer / debugging / reserve validation | 2–4 h |

Total:

```text
~16–20 hours
```

If downloading or one training stage runs long, cut reserve validation and secondary metrics first.

Do **not** cut RETAIN, IDK reversibility, exact patching, held-out authors, or random controls.

---

## 21. Go / no-go gates

### Gate P0 — public laboratory

Proceed only if public `FULL` materially exceeds public `RETAIN` on discovery forget targets.

### Gate P1 — reversible suppression

Proceed only if:

- IDK is behaviorally suppressed;
- IDK is reasonably close to RETAIN;
- mean `s_refusal − s_correct` is at least 2.0 nats/token;
- `R_control` degradation is at most 0.20 nats/token;
- adapter removal restores FULL.

### Gate P2 — GD matching

Proceed only if a public GradDiff checkpoint can be made reasonably comparable to RETAIN without catastrophic utility loss.

If matching is imperfect, record it and narrow the final claim.

### Gate P3 — causal localization

Proceed to CAA only if at least one exact FULL → IDK residual patch gives meaningful positive target recovery.

### Gate P4 — held-out CAA calibration

Interpret GD only if the frozen learned direction restores held-out IDK more than RETAIN and more than norm-matched random directions.

### Gate P5 — GD inference

Only after P4:

$$
C_{GD}
=
\Delta GD-\Delta RETAIN
$$

is interpreted.

---

## 22. Claim templates

### If IDK calibrates and GD transfers

> Exact activation patching first localized a residual-stream layer where restoring the accessible FULL state causally increased target evidence inside a deliberately suppressed model. A contrastive direction estimated at that layer on separate authors then generalized to held-out targets: it restored more target evidence in the reversible IDK state than in the TOFU-withheld RETAIN reference, and the same frozen intervention also produced selective recovery in a Gradient-Difference-unlearned checkpoint. This supports a shared causal accessibility pattern under the tested intervention; it does not prove that the original memories remain fully intact.

### If IDK calibrates and GD does not transfer

> Exact patching and held-out contrastive steering successfully reversed a deliberately constructed suppression state, but the same calibrated intervention did not distinguish Gradient-Difference unlearning from the TOFU-withheld RETAIN reference. This suggests that successful recovery of refusal-like suppression does not automatically transfer to gradient-based factual unlearning.

### If exact patching works but CAA calibration fails

> FULL-to-IDK activation patching identified a causal residual-state difference, but its average contrastive direction did not selectively generalize to held-out suppressed targets over the RETAIN reference. The result suggests that the relevant accessibility difference is not captured by a single shared linear direction in this setting.

### If patching itself fails

> Replacing individual Q_END residual states from the accessible FULL model did not reliably restore target evidence in the reversible IDK state. The proposed layer-level residual assay therefore failed calibration before any inference about Gradient Difference was made.

---

## 23. Relation to the broader research program

This pivot does **not** test the full FT/KD research program in `research.md`.

It is the smallest causal instrument-building experiment:

$$
\text{accessible FULL}
\rightarrow
\text{known reversible suppression IDK}
\rightarrow
\text{causal layer localization}
\rightarrow
\text{held-out shared recovery direction}
\rightarrow
\text{RETAIN vs GD comparison}.
$$

Its purpose is to learn whether there is enough mechanistic signal to justify a larger study.

If positive, the next experiment should increase reference-state rigor using controlled alias/canary exposure and test transfer across unlearning methods.

If negative, analyze where calibration failed before adding more expressive interpretability methods.

---

## 24. Verified external resources

OpenUnlearning repository:

```text
https://github.com/locuslab/open-unlearning
```

OpenUnlearning paper:

```text
https://arxiv.org/abs/2506.12618
```

Public FULL:

```text
https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

Public RETAIN90:

```text
https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
```

Public unlearned-model collection:

```text
https://huggingface.co/collections/open-unlearning/tofu-unlearned-models
```

Primary GradDiff candidate:

```text
https://huggingface.co/open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10
```

TOFU:

```text
https://huggingface.co/datasets/locuslab/TOFU
```

OpenUnlearning's official evaluation configuration uses the Llama-3.2-1B FULL model as the target and `retain90` evaluation logs as the `forget10` reference. Their published TOFU model family includes `full`, `retain90`, `retain95`, and `retain99`; the public unlearned collection contains multiple Llama-3.2-1B GradDiff hyperparameter checkpoints.

---

## 25. One-line frozen experiment

$$
\boxed{
\text{public FULL}
+
\text{public RETAIN}
+
\text{our removable suppression IDK LoRA}
+
\text{behavior-matched public GD}
}
$$

followed by:

$$
\boxed{
\text{exact 16-layer FULL}\rightarrow\text{IDK patching}
\rightarrow
\text{CAA direction at }l^*
\rightarrow
\text{held-out IDK / RETAIN / GD steering}
}
$$

That is the MATS MVP. Do not expand the scope until this pipeline has either produced or falsified a calibrated signal.
