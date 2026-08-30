# Final report: IDK-calibrated causal recoverability in TOFU

## Question

When an IDK-calibrated residual intervention is transferred to a Gradient Difference checkpoint and a TOFU-withheld RETAIN reference, does it recover proportionally more target-answer evidence from GD02?

## Frozen design

- States: FULL, reversible IDK adapter `step-000025`, GD02, and RETAIN.
- Discovery authors selected layer **14** using IDK only.
- Direction: mean `FULL − IDK` Q_END residual, norm **6.796089**.
- Frozen alpha: **1**.
- Confirmation: five unseen authors, five norm-matched random directions, no retuning.
- Primary metric: teacher-forced mean target log-probability in nats/token; fractional recovery is aggregated within author before averaging authors.

## Calibration attempt and scope revision

Two reversible IDK constructions failed the original P1 behavioral calibration. Refusal-only `step-000025` produced a FULL-minus-IDK gap of **0.1199 nats/token** and RETAIN distance **2.4844**. Direct suppression produced a gap of **0.0820**, refusal-minus-correct margin **-0.7047**, and RETAIN distance **2.5222**. Both preserved the frozen base and passed their utility guardrail, but neither established a RETAIN-matched hidden-memory audit.

Accordingly, the final experiment uses the refusal-only adapter solely as a known removable mechanism calibration. It does not treat IDK as an unlearned state and does not claim that successful IDK reversal validates an intact-memory interpretation for GD02.

## Discovery results

Exact FULL activation patching into IDK recovered **70.64%** of the IDK gap at layer 14 (raw **+0.0851** nats/token; 5/5 positive authors).

At that frozen layer, GD02 recovered **2.35%** and RETAIN recovered **1.46%**, giving the prespecified differential **+0.89 percentage points**. This discovery effect was small.

Alpha 1 produced IDK recovery of **62.33%**, RETAIN recovery of **-7.49%**, and worst absolute R_control change of **0.2372 nats/token**. The negative RETAIN effect is a specificity limitation and is not hidden by the differential objective.

## Held-out confirmation

### Reversible calibration

- IDK fractional recovery: **61.44%**.
- Raw recovery: **+0.0846 nats/token**.
- Author-clustered 95% interval: **[+52.23%, +70.45%]**.
- Positive authors: **5/5**.
- Learned-direction rank: **1/6**.

The held-out reversible IDK calibration succeeded and ranked above all five random controls.

### GD02 transfer test

- GD02 fractional recovery: **-69.39%**; raw **-0.4797 nats/token**.
- RETAIN fractional recovery: **-9.39%**; raw **-0.2041 nats/token**.
- Differential `GD02 − RETAIN`: **-60.00 percentage points**.
- Author-clustered 95% interval: **[-85.69, -42.12] percentage points**.
- Positive differential authors: **0/5**.
- Learned-direction rank: **6/6**.

No differential causal recoverability was detected. The learned direction produced a contrary effect: it reduced GD02 target evidence more than RETAIN target evidence.

## Greedy-generation diagnostics

Generation is diagnostic rather than a selection endpoint. IDK ROUGE-L changed from **0.023** to **0.590**, while the refusal phrase heuristic changed from **41/100** to **2/100**. GD02 ROUGE-L changed from **0.532** to **0.321**. These diagnostics agree with successful IDK reversal and contrary GD02 transfer, but they are not substitutes for the frozen likelihood endpoints.

## Interpretation and limitations

The experiment validates the assay on a known reversible suppression state, but it does **not** find that the same direction recovers GD02. This result does not prove that GD02 erased all relevant information. It establishes only that differential recoverability was absent—and strongly contrary—under one layer-14 Q_END additive direction in this model and dataset.

Further limitations are: GD02 was not behavior-matched to RETAIN; only five confirmation authors and five random directions were used; the alpha-1 direction harmed RETAIN utility during discovery; author-level bootstrap intervals are based on five clusters; and generation overlap/refusal measures are simple diagnostics.

## Provenance

- Final states: `0e8ba8a0f4816507da0f83cf98092d0419bd4b759bf53bb95574ae5f1e3e2d6c`
- Causal layer: `c030213ff907a0a8a146c0b9904ff3889f7525f2b34b498a39a6a6198ff237a2`
- Patch-transfer result: `849ce5d4cdada7482b707d76c98ff4a9c20d27add5e0329d39fafa3a4fb7e150`
- Alpha-selection result: `96be1cf979b819778507e046858e6c996fbc32c1cd399aa92b438e252394b33c`
- Confirmation result: `b5e2665254fc6c3ec770a05ec6c3eb7df2cd2abc45b58a44201eed9ecd8cd084`
- Analysis manifest: `26fcc6002fac85add8e45c2b3005acdcc953d8be0f4be3df12e0429b87626be1`
