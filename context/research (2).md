# Mechanistic Audits of Model Forgetting

## Using post-training memorization as a controlled laboratory for reliable unlearning

## Project thesis

GDPR erasure requests, copyright compliance, unlearning audits, benchmark decontamination, and hazardous-capability removal all require evidence that a model no longer retains particular information or capabilities. In practice, model-level claims are usually supported by behavioral tests: evaluators prompt the model, fail to elicit the target, and treat non-production as evidence of removal.

That inference is structurally underdetermined. The same behavioral failure can result from genuine removal, failure to acquire the information, representational reorganization, weakened retrieval, refusal, or an inadequate elicitation method. A stronger attack can sometimes recover behavior that a standard evaluation declared gone. Behavioral evidence can measure accessibility under an evaluation protocol; by itself, it cannot establish why access failed.

The long-term goal of this project is to build a **causal, mechanistically validated audit of model forgetting**:

> **Can we distinguish robust removal from information that remains causally available but behaviorally inaccessible, and can that distinction guide interventions that make forgetting durable?**

The project does not assume that apparently forgotten information is secretly retained. Genuine non-acquisition, robust removal, altered representation, and suppressed accessibility are competing explanations. The task is to discriminate among them with calibrated causal evidence.

Fine-tuning and knowledge distillation are not the ultimate object of study. They are the controlled laboratory in which the audit can be built and falsified. They provide matched models, known exposure records, large differences in measured memorization, dense training checkpoints, and counterfactual control arms that are unavailable for frontier models or retrospective copyright claims.

The research program therefore proceeds in four steps:

\[
\text{construct known reference states}
\rightarrow
\text{identify a causal criterion}
\rightarrow
\text{validate it in the FT/KD laboratory}
\rightarrow
\text{test whether it predicts robust forgetting after unlearning}.
\]

If the criterion transfers, it can support stronger audits of unlearning, contamination, privacy, and safety-relevant capability removal. If it fails to transfer, the failure will identify which mechanistic evidence is specific to memorization or to a particular post-training objective.

---

## The failure of behavioral forgetting audits

Behavioral non-recall is an ambiguous negative result. Exact extraction has a clear positive interpretation: under a specified prompt and decoding procedure, the model produced the target. Failure to extract has no equally strong negative interpretation.

This ambiguity creates a common failure mode across domains:

- **Model-level erasure and privacy:** A model may stop reproducing personal or copyrighted text under standard prompts while retaining sequence-specific information recoverable through another context or later update.
- **Unlearning:** A forget-set score may improve even when the relevant capability or representation survives and resurfaces after fine-tuning.
- **Benchmark decontamination:** A model may fail to reproduce an item verbatim while still carrying exposure-specific information that affects performance.
- **Hazardous-capability removal:** A model may suppress an answer through policy or retrieval changes while preserving the underlying capability.
- **Alignment:** Refusal and behavioral compliance can conceal rather than remove the computation supporting an unwanted behavior.

The project will not attempt to prove the unlimited claim that information exists nowhere in a neural network. No finite probe can certify that. It will instead develop an operationally defensible conclusion:

> **no detectable target-specific causal influence under a calibrated family of behavioral, representational, and intervention-based assays.**

That is stronger than “we prompted it and it did not answer,” while remaining falsifiable.

---

## Why fine-tuning and distillation are the laboratory

[Borkar et al., *Memorization Dynamics in Knowledge Distillation for Language Models* (2026)](https://arxiv.org/abs/2601.15394) supplies an unusually useful controlled contrast. Distilled students retain much of a teacher's competence improvement while displaying substantially less extractable memorization than conventionally fine-tuned students.

A matched FT/KD experiment can provide:

- identical student initialization;
- identical training sequences and token budgets;
- a large, known difference in measured memorization;
- complete records of which examples each condition encountered;
- dense checkpoints showing when the trajectories diverge;
- and a matched never-exposed arm created by withholding target sequences.

The last point requires care. An exposure record establishes whether a model encountered a sequence; it does not reveal whether the model encoded or retained it. Likewise, natural text may already be partly reconstructable from pretraining. The experiment therefore needs high-entropy canaries or equivalently controlled targets for strict exposure ground truth, alongside natural sequences for external validity.

The laboratory should contain three reference states whose provenance is known:

1. **Never exposed:** matched targets withheld from the relevant training run.
2. **Acquired and accessible:** direct training produces demonstrable target-specific recall.
3. **Acquired then reversibly inaccessible:** a controlled intervention suppresses demonstrated recall without changing the memory-bearing weights, and removing that intervention restores it without re-exposure.

The FT/KD comparison is then applied to a fourth, genuinely ambiguous population: examples recalled after direct fine-tuning but not after distillation. The calibrated audit asks which reference state those examples resemble and whether causal interventions support that classification.

Hard and soft KD remain mechanism-revealing ablations:

\[
\text{one-hot data target}
\longleftrightarrow
\text{hard teacher target}
\longleftrightarrow
\text{soft teacher distribution}.
\]

They can reveal whether target sharpness, teacher uncertainty, or access to alternative continuations changes the causal mechanism. Their rare teacher-specific difference is not large enough to serve as the project's primary population.

---

## Why this requires mechanistic interpretability

More sensitive extraction attacks still measure behavior. A membership classifier can detect exposure-correlated signal without showing that the model uses it. Training curves can show when two objectives diverge without identifying the computation responsible.

The project therefore targets the causal chain:

\[
\boxed{
\text{target exposure or removal}
\rightarrow
\text{internal computation}
\rightarrow
\text{target accessibility}
}
\]

A candidate audit signal must do more than correlate with recall. It should satisfy three requirements:

1. **Discrimination:** separate known never-exposed, accessible, and reversibly inaccessible reference states on held-out targets.
2. **Causal mediation:** manipulating the implicated state changes target-specific continuation or capability expression without inserting the target through the intervention.
3. **Predictive validity:** forecast whether apparently removed behavior will resurface under subsequent fine-tuning or stronger elicitation.

A candidate control mechanism must satisfy a fourth:

4. **Selectivity:** manipulating it reduces target information or capability more strongly than it harms held-out language modeling and useful competence.

These requirements make causal interpretability the bridge from a behavioral forgetting claim to an auditable and eventually controllable mechanism.

---

## Research questions

### RQ1: Can causal internal evidence distinguish known absence from known inaccessibility?

Construct the reference states above and test whether sequence-specific internal changes, causal patching effects, or feature-level interventions discriminate them on held-out targets. This stage calibrates the audit rather than deciding in advance what “retained” information must look like.

### RQ2: What mechanism explains the memorization difference between fine-tuning and distillation?

Track same-initialization FT and KD models through intermediate checkpoints. Identify where their target-specific trajectories diverge, then use bidirectional causal intervention and feature-level decomposition to determine which computations mediate the difference.

### RQ3: Does the calibrated criterion predict whether unlearning is robust?

Apply at least one open unlearning method to a controlled forget set. Test whether the criterion measured immediately after unlearning predicts recovery under subsequent benign fine-tuning, adversarial relearning, or stronger elicitation better than behavioral forget-set metrics alone.

### RQ4: Can the mechanism improve removal or prevent unwanted retention?

Use the validated mechanism to design an intervention and measure whether it moves the forgetting–competence frontier. Depending on the result, the intervention may target activation features, gradients, loss weighting, distillation temperature, or training-data routing.

RQ1 builds the instrument. RQ2 supplies the mechanistic model organism. RQ3 establishes relevance to unlearning. RQ4 turns explanation into training control.

---

## Experimental spine

This document specifies the scientific structure rather than a final compute plan. The smallest decisive experiment should instantiate the following design.

### 1. Paired training conditions

Begin from the same student initialization and train matched models under:

\[
\text{CE fine-tuning},
\qquad
\text{soft KD},
\qquad
\text{hard KD as a diagnostic ablation}.
\]

Hold the dataset, sequence order, token budget, optimizer family, and checkpoint schedule fixed where the objectives permit. Use multiple seeds before making claims about shared geometry.

Add two matched control arms:

- a model trained with the target evaluation subset withheld, providing known non-exposure;
- and an exposed, demonstrably recalling model placed behind a reversible intervention that blocks recall without changing the underlying trained weights.

These controls calibrate the causal audit before it is applied to ambiguous KD-filtered or unlearned examples.

### 2. Dense trajectory measurement

Exact extraction remains the strict behavioral endpoint:

\[
\text{prefix}_{50}
\rightarrow
\text{exact suffix}_{50}.
\]

Because exact recall is rare, it cannot be the only dependent variable. For every evaluated sequence and checkpoint, also measure:

- target suffix log-likelihood;
- true-suffix versus matched-decoy likelihood ratio;
- exposure or rank-based scores;
- teacher entropy and teacher–student divergence;
- base-model and trained-model perplexity;
- compressibility and other established data-level predictors.

These continuous measures reveal trajectories below the exact-extraction threshold. They should not be interpreted as proof that a memory is present; they are outcomes used to identify where causal analysis is most informative.

### 3. Natural examples plus calibrated controls

The main scientific population should remain natural training sequences so that the project explains the phenomenon observed under realistic training.

Synthetic canaries serve narrower roles:

- calibrating measurement sensitivity;
- providing known unexposed and exposed controls;
- testing the recovery power of interventions;
- and establishing whether a null result reflects a weak assay.

Canaries should not replace natural examples or artificially define the mechanism through an overexposure regime unlike the target phenomenon.

The principal comparison groups are:

| Group | Fine-tuned student | Distilled student | Role |
|---|---:|---:|---|
| Recalled by both | High recall | High recall | Easy/shared memorization |
| FT-only | High recall | Low recall | Main contrast requiring explanation |
| Continuous KD evidence without recall | High recall | Subthreshold change | Tests possible latent or reorganized influence |
| Recalled by neither | Low recall | Low recall | Matched control population |
| Unexposed canaries | Not trained | Not trained | Calibrated negative controls |
| Reversibly suppressed targets | Previously recalled | Recall blocked | Known inaccessible reference state |

### 4. Bidirectional causal patching

For FT-only and matched control sequences, transfer activations between corresponding FT and KD checkpoints.

Patch from FT into KD:

\[
h_{\mathrm{KD}}^{(l,t)}
\leftarrow
h_{\mathrm{FT}}^{(l,t)}
\]

and measure whether target continuation probability is restored.

Patch from KD into FT:

\[
h_{\mathrm{FT}}^{(l,t)}
\leftarrow
h_{\mathrm{KD}}^{(l,t)}
\]

and measure whether target continuation is suppressed.

The initial causal outcome should be teacher-forced target-token log-likelihood across the suffix, followed by free-generation extraction as the stricter test. Patching must be compared with matched non-target sequences, random components, and same-condition controls to exclude generic model-quality effects.

This stage asks:

> At which layers, token positions, and components does the objective-specific trajectory begin to causally affect sequence continuation?

### 5. Feature and geometry analysis

Only after causal sites are identified should the project test their internal organization. Candidate tools include component decomposition and a paired dictionary or crosscoder trained across corresponding FT and KD checkpoints.

The target is precise: identify features that causally mediate the memorization difference between the objectives. The analysis should determine whether causal effects are organized as:

- one shared direction;
- several reusable feature families;
- shared control structure combined with instance-specific content;
- objective-specific representational organization;
- or predominantly example-specific mechanisms.

Low-dimensionality is one possible result, not an assumption.

### 6. Transfer to unlearning

Apply a reproducible open unlearning method to a controlled forget set. Measure behavioral forgetting and the calibrated mechanistic criterion immediately after unlearning. Then test robustness under:

- benign downstream fine-tuning;
- targeted or adversarial relearning;
- stronger elicitation and alternative prompts;
- and, where feasible, activation-level restoration attempts.

The decisive comparison is:

\[
\text{behavioral forget score alone}
\quad\text{versus}\quad
\text{behavioral score + mechanistic criterion}
\]

for predicting which targets later resurface. Without this transfer test, the project can claim a mechanism of FT/KD memorization but not a general audit of model forgetting.

### 7. Causal control

For candidate mechanisms supported by prediction and patching, test inference-time interventions such as feature ablation or subspace projection:

\[
h' = h - P_{\mathcal M}h.
\]

Evaluate:

\[
\Delta\text{target memorization}
\quad\text{against}\quad
\Delta\text{held-out loss and competence}.
\]

Only if an inference-time intervention is selective should the project advance to training-time control, such as representation penalties, loss reweighting, gradient projection, temperature adjustment, or selective gradient routing. The intervention should follow from the discovered mechanism rather than being chosen in advance.

---

## Competing hypotheses and observations

These hypotheses are deliberately downstream of the project objective. They describe observations the experiments may support; none is presupposed by the title or thesis.

### Hypothesis A: distillation prevents sequence-specific encoding

KD-filtered sequences resemble matched unexposed controls across calibrated behavioral and causal assays. FT activations do not reveal a sequence-general causal route that restores the corresponding continuation in KD, and later post-training does not preferentially recover it.

**Implication:** The objective changes whether sequence-specific information is acquired. Control should focus on targets, gradient flow, and early training dynamics.

### Hypothesis B: distillation preserves information with weaker output coupling

KD-filtered sequences remain distinguishable from unexposed controls, and sequence-general causal interventions recover target-specific continuation without supplying the answer through the intervention.

**Implication:** Reduced extraction partly reflects weakened accessibility rather than lack of sequence-specific influence. This would affect privacy, contamination, and unlearning audits.

### Hypothesis C: fine-tuning and distillation reorganize the information differently

Both conditions contain target-specific information, but it is represented or used through different features and pathways. Direct activation transfer may initially fail until the relevant cross-model representation is aligned.

**Implication:** Behavioral memorization differences can arise from objective-dependent representational organization rather than a scalar difference in memory strength.

### Hypothesis D: shared control structure governs instance-specific content

The sequence contents remain high-dimensional and example-specific, while a reusable feature family, subspace, or circuit motif controls whether they influence exact continuation.

**Implication:** A shared causal target may permit selective steering even though the memories themselves do not occupy one semantic direction.

### Hypothesis E: mechanisms are predominantly instance-specific

No stable reusable mechanism survives controls for exposure, entropy, perplexity, compressibility, and training drift. Causal sites vary substantially across sequences and seeds.

**Implication:** Universal post-hoc steering is unlikely to control memorization. Data selection, example-level gradient routing, or objective design may be more appropriate.

### Hypothesis F: some apparent memorization is reconstruction

Some exact reproductions show little sequence-specific change relative to unexposed or counterfactual models, particularly for predictable or compressible text.

**Implication:** Exact match can conflate exposure-driven memorization with reconstruction from generic competence. This distinction could improve contamination and copyright analysis.

### Hypothesis G: mechanisms differ across supervision objectives

Direct FT, hard KD, and soft KD diverge at different causal stages. For example, hard targets may strengthen exact continuation while soft distributions preserve useful uncertainty—but the specific mechanism is left open.

**Implication:** Hard-versus-soft KD becomes a diagnostic for explaining which aspects of memorization are governed by target form rather than the entire research question.

### Hypothesis H: mechanistic evidence predicts resurfacing after unlearning

Among targets that pass the same behavioral forget threshold, those retaining stronger calibrated causal evidence are more likely to return after further fine-tuning, stronger elicitation, or adversarial relearning.

**Implication:** The audit supplies information that behavioral evaluation misses and can rank apparently successful unlearning outcomes by residual risk.

---

## Evidence standards

### Behavioral non-recall is not evidence of absence

Exact extraction has a clear positive interpretation under a specified prompt. Failure to extract is compatible with several internal states. The project will therefore avoid claims that information is absent merely because a prompt or probe fails.

The strongest defensible language is operational:

> no detectable sequence-specific effect under a calibrated family of behavioral and causal assays.

### A probe is not a mechanism

Linear or nonlinear decodability can provide evidence that information is available to an analyst, but not that the model uses it. Mechanistic claims require causal mediation or intervention.

### Patching must not insert the target

An intervention that transfers target-specific activations may demonstrate sufficiency of a state, but a general control method cannot be trained separately on each held-out suffix or covertly supply the answer it is meant to reveal. Sequence-generalization and decoy controls are essential.

### Localization is not editability

[Hase et al., *Does Localization Inform Editing?* (2023)](https://arxiv.org/abs/2301.04213) shows that causal localization does not automatically identify the best place to edit knowledge. The proposal distinguishes:

- where the FT/KD computation differs;
- which difference causally mediates recall;
- and which intervention permits selective control.

### Mechanistic complexity must earn itself

[Borkar et al. (2026)](https://arxiv.org/abs/2601.15394) already identifies strong behavioral and data-level predictors. Compare:

\[
\text{data and behavioral features},
\qquad
\text{mechanistic features},
\qquad
\text{combined model}.
\]

If internal features neither improve early prediction nor support causal control, the mechanistic account has not earned its added complexity.

---

## Intellectual lineage

### Training dynamics

[Biderman et al., *Position: Don't Just “Fix it in Post”: A Science of AI Must Study Training Dynamics* (2026)](https://arxiv.org/abs/2606.06533) argues for progressing from early prediction to intervention and eventually training design. This project follows that structure while making causal interpretability the bridge between observation and control.

### Distillation and memorization

[Borkar et al. (2026)](https://arxiv.org/abs/2601.15394) establishes the memorization–competence separation and the objective-dependent selectivity that this project aims to explain internally.

### Mechanistic inheritance

[Blank et al., *Subliminal Learning Is Steering Vector Distillation* (2026)](https://arxiv.org/abs/2606.00995) demonstrates that some teacher-to-student behavioral inheritance is mediated by compact activation structure. The present project asks a broader question: whether post-training objectives create reusable causal structure governing the inheritance or formation of instance-specific information, without assuming that the content itself lies in one direction.

### Robustness of removal

[Lee et al., *Distillation Robustifies Unlearning* (2025)](https://arxiv.org/abs/2506.06278) shows why behavioral suppression and robust removal can diverge. In this project, suppressed accessibility is one possible explanation for low recall, to be tested against encoding prevention and representational reorganization.

### Safety-relevant post-training inheritance

[Engels and Nanda, *Why Do Naive SFT Filters For Safety Properties Fail?* (2026)](https://www.alignmentforum.org/posts/wyZRNgpeiPeRXB6eT/why-do-naive-sft-filters-for-safety-properties-fail) motivates the larger safety problem: teacher-generated data can transfer properties that surface filtering does not reliably control. Memorized sequences provide a cleaner first domain in which exposure and reproduction can be measured precisely.

---

## What would constitute success?

### Minimum scientific result

A causal criterion separates held-out never-exposed, acquired-and-accessible, and acquired-then-reversibly-inaccessible targets above behavioral and data-level baselines. The FT/KD experiment then identifies when and where the ambiguous memorization trajectories diverge.

### Strong mechanistic result

A feature, component, subspace, or repeated motif that:

- emerges consistently across relevant sequences and seeds;
- predicts future memorization beyond established baselines;
- causally mediates FT/KD continuation differences;
- and generalizes to held-out sequences.

### Strong audit result

Among targets that satisfy the same behavioral forgetting threshold after unlearning, the mechanistic criterion predicts which ones later resurface under fine-tuning, stronger elicitation, or adversarial relearning. This is the result that would justify calling the method an audit of model forgetting rather than only an analysis of distillation.

### Strong control result

An intervention derived from the mechanism that moves the memorization–competence frontier:

\[
\text{less unwanted sequence-specific recall or influence}
\quad\text{with}
\quad
\text{limited loss of useful competence}.
\]

### Valuable negative result

A careful demonstration that mechanistic differences are predominantly instance-specific, unstable across seeds, unhelpful beyond entropy and perplexity, or non-transferable to unlearning would constrain universal mechanistic audits. It would also identify where the project must fall back to method-specific, data-level, or gradient-level evidence.

---

## Broader audit targets

Unlearning is the first transfer domain, not optional future decoration. If the calibrated criterion predicts robustness there, subsequent work can test:

- **model-level erasure and privacy claims**, to estimate residual information risk beyond greedy extraction;
- **copyright compliance**, to distinguish failure to reproduce from lack of target-specific causal influence;
- **benchmark decontamination**, to detect exposure-driven influence below exact recall;
- **distillation**, to control which teacher information reaches the student;
- **alignment and hazardous-capability removal**, to distinguish robust mechanistic change from locally suppressed behavior;
- and **post-training design**, to steer trajectories before unwanted information or capabilities consolidate.

The first paper should not attempt all of these domains. It needs the calibrated FT/KD laboratory and one credible unlearning transfer test. The remaining applications define the program that follows.

---

## Research identity and fit

The concise research identity is:

> **Build causal audits that distinguish robust model forgetting from behavioral non-recall, using post-training memorization as the controlled laboratory and mechanistic interpretability as the instrument.**

For Mireshghallah's agenda, the project turns the memorization–capacity–competence relationship into a practical audit for contamination, privacy, and unlearning, while studying the early dynamics that determine whether information survives post-training.

For Nanda's agenda, it uses causal interpretability to test whether post-training changes underlying computation or only its behavioral expression, then uses the validated mechanism to steer what models retain. It builds from the existence of low-dimensional behavioral inheritance without requiring memorized sequences to obey the same geometry.

The overlap is not “apply a steering vector to memorization.” It is:

\[
\boxed{
\text{turn “the model did not produce it” into a causal, stress-tested claim about forgetting}
}.
\]

---

## Current boundary of the proposal

This document defines the question, contribution, competing explanations, and experimental logic. A separate experimental plan should specify:

- model family and scale;
- dataset and natural-sequence sampling;
- the never-exposed and reversible-inaccessibility calibration design;
- checkpoint density;
- the first causal-patching implementation;
- feature-learning method, if justified by the patching result;
- the unlearning method and resurfacing stress test;
- seed count and statistical power;
- compute requirements;
- and preregistered figures and falsification criteria.

The first experiment should be selected to discriminate among the hypotheses above rather than to presuppose any one of them.
