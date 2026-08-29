#!/usr/bin/env python3
"""Run Chunk 2 matched FULL-to-GD/RETAIN exact Q_END activation patches."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    load_yaml,
    stable_hash,
)
from pivot_experiment.data import load_prepared_rows  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.patching import (  # noqa: E402
    ActivationStore,
    run_receiver_controls,
    run_matched_patching,
    validate_patch_inputs,
)


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--phase",
        choices=("matched", "controls"),
        default="matched",
        help="Run the layer sweep first; controls are authorized by the partial P2 check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every frozen baseline and donor sidecar without loading weights.",
    )
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    models = load_yaml(args.models_config)
    rows = load_prepared_rows(args.artifacts, "discovery")
    if len(rows) != 100:
        raise ValueError(f"Expected 100 discovery rows, found {len(rows)}")
    full, retain, gd, freeze = validate_patch_inputs(
        artifact_root=args.artifacts,
        models_config=models,
        rows=rows,
    )
    candidate_id = models["downstream_gd_candidate"]
    layers = experiment["patching"]["layers"]
    expected_layers = models["expected_architecture"]["decoder_layers"]
    hidden_size = models["expected_architecture"]["hidden_size"]
    if layers != list(range(expected_layers)):
        raise ValueError("Chunk 2 must patch every decoder layer exactly once")
    ActivationStore(full, expected_layers, hidden_size).validate_all(
        [row["example_id"] for row in rows]
    )
    freeze_hash = stable_hash(freeze)
    print(
        f"Frozen GD={candidate_id} ({models['downstream_gd_scope']}); "
        f"100 examples x 16 layers x 2 receivers = 3200 patched scores"
    )
    print("FULL donor activation sidecars and all baselines validated")
    if args.phase == "controls":
        selection_path = args.artifacts / "results" / "p2_layer_selection.json"
        if not selection_path.exists():
            raise SystemExit("Run the matched phase and check_gates.py --through P2 first")
        import json

        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        print(f"Frozen control layer: {selection['selected_layer']}")
    if args.dry_run:
        print("Dry run complete: no tokenizer or model weights loaded")
        return

    tokenizer = load_tokenizer(models)
    output_root = args.artifacts / "interventions"
    receivers = (
        ("gd", candidate_id, gd, candidate_id),
        ("retain", "retain", retain, None),
    )
    for receiver_state, model_key, baselines, record_candidate_id in receivers:
        model, device = load_public_model(model_key, models)
        print(f"Loaded {model_key} on {device}")
        if args.phase == "matched":
            run_matched_patching(
                model=model,
                tokenizer=tokenizer,
                receiver_state=receiver_state,
                receiver_model_spec=models["models"][model_key],
                candidate_id=record_candidate_id,
                rows=rows,
                receiver_baselines=baselines,
                full_donors=full,
                layers=layers,
                hidden_size=hidden_size,
                batch_size=experiment["evaluation"]["sequence_batch_size"],
                split_hash=next(iter(full.values()))["split_hash"],
                freeze_hash=freeze_hash,
                output_path=output_root / f"p2_{receiver_state}_matched.jsonl",
            )
        else:
            run_receiver_controls(
                model=model,
                tokenizer=tokenizer,
                receiver_state=receiver_state,
                receiver_model_spec=models["models"][model_key],
                candidate_id=record_candidate_id,
                rows=rows,
                receiver_baselines=baselines,
                own_activation_rows=gd if receiver_state == "gd" else None,
                full_donors=full,
                selected_layer=selection["selected_layer"],
                layer_selection_hash=stable_hash(selection),
                self_patch_count=experiment["patching"]["self_patch_example_count"],
                total_layers=expected_layers,
                hidden_size=hidden_size,
                batch_size=experiment["evaluation"]["sequence_batch_size"],
                split_hash=next(iter(full.values()))["split_hash"],
                freeze_hash=freeze_hash,
                output_path=output_root / "p2_controls.jsonl",
                all_receiver_models={
                    "gd": models["models"][candidate_id],
                    "retain": models["models"]["retain"],
                },
            )
        release_model(model)
    print(f"P2 {args.phase} phase complete; run check_gates.py --through P2")


if __name__ == "__main__":
    main()
