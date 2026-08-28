# Archived calibration: `idk_refusal_failed`

This directory preserves the completed refusal-only LoRA experiment exactly as
the failed P1 calibration result produced it. It is evidence, not an active
checkpoint source.

## Construction

- Frozen base: public `FULL`
- Trainable state: removable rank-8 LoRA
- Objective: refusal-target SFT plus paired retain-answer SFT
- Data: all 400 forget questions paired with 400 fixed retain questions
- Training: two epochs, 25 optimizer steps
- Evaluated checkpoints: steps 3, 6, 10, 16, and 25

## Formal result

P1 failed for the selected step-25 adapter:

- `FULL − IDK`: `+0.1198518` nats/token; required at least `+0.15`
- `|IDK − RETAIN|`: `2.4843938`; required at most `0.30`
- `R_control` degradation: `0.1129755`; passed the `0.20` guardrail
- adapter off/on/off restoration: exact
- frozen FULL parameter hash: exact before/after match

Step 16 suppressed more strongly (`+0.216923`) but degraded `R_control` by
`0.247765` and remained `2.387323` nats/token from RETAIN.

The refusal loss fell from `5.7420` to `0.3038`, showing that the adapter learned
to generate refusals without sufficiently lowering teacher-forced correct-answer
likelihood. This is the reason for the redesign, not a corrupted run.

## Contents

- `code/`: snapshot of the implementation and configuration that produced it
- `artifacts/`: checkpoints, raw candidate and FULL/RETAIN reference scores,
  frozen splits, activations, logs, selection, reversibility, and formal gates
- `checksums.sha256`: integrity hashes for the archived files

Do not resume the new suppression experiment from these adapters. The replacement
must begin from a fresh LoRA over the original frozen `FULL` checkpoint.
