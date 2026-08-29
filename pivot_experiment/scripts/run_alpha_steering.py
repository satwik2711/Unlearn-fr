#!/usr/bin/env python3
"""Run one resumable IDK/RETAIN alpha-selection steering job."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    load_yaml,
)
from pivot_experiment.data import load_prepared_rows  # noqa: E402
from pivot_experiment.idk_localization import load_final_freeze, resolve_frozen_path  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.patch_transfer import load_causal_layer  # noqa: E402
from pivot_experiment.steering import (  # noqa: E402
    STEERING_RECEIVERS,
    load_direction,
    load_receiver_archives,
    run_alpha_baselines,
    run_alpha_sweep,
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
    parser.add_argument("--state", choices=STEERING_RECEIVERS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    models = load_yaml(args.models_config)
    final_states = load_final_freeze(args.artifacts / "freeze" / "final_states.json")
    causal_layer = load_causal_layer(
        args.artifacts / "freeze" / "causal_layer.json", final_states
    )
    direction_path = args.artifacts / "directions" / "full_minus_idk.safetensors"
    direction, direction_manifest = load_direction(
        direction_path,
        direction_path.with_suffix(".manifest.json"),
        final_states,
        causal_layer,
    )
    rows_by_subset = {
        "discovery": load_prepared_rows(args.artifacts, "discovery"),
        "r_control": load_prepared_rows(args.artifacts, "r_control"),
    }
    if any(len(rows) != 100 for rows in rows_by_subset.values()):
        raise ValueError("Chunk 4 requires 100 discovery and 100 R_control rows")
    expected = {
        subset: {row["example_id"] for row in rows}
        for subset, rows in rows_by_subset.items()
    }
    archived = load_receiver_archives(args.state, final_states, expected)
    alphas = [float(value) for value in experiment["steering"]["alphas"]]
    print(
        f"Chunk 4 {args.state}: 200 baselines + "
        f"{len(alphas)} x 200 steering scores = {len(alphas) * 200}"
    )
    print(
        f"Direction: FULL-IDK at layer {causal_layer['selected_layer']}; "
        f"alphas={alphas}"
    )
    if args.dry_run:
        print("Dry run complete: tokenizer and model weights were not loaded")
        return

    tokenizer = load_tokenizer(models)
    if args.state == "idk":
        base_model, device = load_public_model("full", models)
        adapter_path = resolve_frozen_path(final_states["states"]["idk"]["adapter_path"])
        adapter_id = final_states["states"]["idk"]["adapter_id"]
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            adapter_name=adapter_id,
            is_trainable=False,
        )
        model.set_adapter(adapter_id)
        model.eval()
    else:
        model, device = load_public_model("retain", models)
    print(f"Loaded {args.state} on {device}")
    batch_size = experiment["evaluation"]["sequence_batch_size"]
    baseline_path = args.artifacts / "scores" / f"{args.state}_alpha_baselines.jsonl"
    baselines = run_alpha_baselines(
        model=model,
        tokenizer=tokenizer,
        receiver=args.state,
        rows_by_subset=rows_by_subset,
        archived=archived,
        final_states=final_states,
        causal_layer=causal_layer,
        direction_manifest=direction_manifest,
        batch_size=batch_size,
        output_path=baseline_path,
    )
    run_alpha_sweep(
        model=model,
        tokenizer=tokenizer,
        receiver=args.state,
        rows_by_subset=rows_by_subset,
        baselines=baselines,
        direction=direction,
        direction_manifest=direction_manifest,
        final_states=final_states,
        causal_layer=causal_layer,
        alphas=alphas,
        batch_size=batch_size,
        output_path=args.artifacts / "interventions" / f"{args.state}_alpha_sweep.jsonl",
    )
    release_model(model)
    print(f"Chunk 4 {args.state} alpha sweep complete")


if __name__ == "__main__":
    main()
