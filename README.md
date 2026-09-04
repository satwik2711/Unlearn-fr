# Mechanistic Audits of Model Forgetting

Behavioral non-recall does not tell us whether a model has genuinely lost some
information, reorganized it, or merely stopped exposing it. This project asks
whether causal analysis of internal activations can distinguish robust removal
from reversible suppression—and whether that distinction predicts what later
resurfaces under fine-tuning, stronger elicitation, or relearning.

The proposed research uses fine-tuning and knowledge distillation as a
controlled laboratory. Starting from matched models and known exposure
records, it will compare accessible, never-exposed, and reversibly suppressed
targets; localize causal differences with activation patching; test whether
those differences form reusable features or directions; and then evaluate
whether the resulting audit predicts the durability of unlearning.

## Completed pilot

The included TOFU pilot calibrated a residual intervention on a removable
refusal adapter over Llama-3.2-1B-Instruct. A `FULL - IDK` direction recovered
61.4% of suppressed target evidence on held-out authors, but transferred in the
opposite direction to a Gradient Difference checkpoint: GD recovery was
-69.4%, compared with -9.4% for the withheld-data reference. The result does
not establish erasure. It shows that a recovery mechanism valid for one known
suppression state need not generalize to another forgetting intervention.

Two failed pilots informed this design. A Qwen-3.5-2B run on deterministically
aliased TOFU data learned the task but produced too little separation between
`FULL` and `RETAIN` for a clean causal comparison. Two subsequent refusal-based
controls were exactly reversible and preserved utility, but neither matched
the withheld-data reference. These negative results motivated the narrower,
claim-bounded experiment above.

## Repository

- `context/research.md` — research motivation and proposed agenda
- `pivot_experiment/src/` — core experimental implementation
- `pivot_experiment/scripts/` — stage entry points
- `pivot_experiment/configs/` — frozen configurations
- `pivot_experiment/artifacts/analysis/` — compact final report, tables, and figures
- `pivot_experiment/archive/` — small records of the failed calibration gates

Large checkpoints, activation caches, raw sweeps, and logs are intentionally
excluded from the submission repository.
