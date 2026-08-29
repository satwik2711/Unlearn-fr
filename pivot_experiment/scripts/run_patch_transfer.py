#!/usr/bin/env python3
"""Run one resumable FULL-to-GD02/RETAIN layer sweep for Chunk 3."""

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
)
from pivot_experiment.data import load_prepared_rows  # noqa: E402
from pivot_experiment.idk_localization import load_final_freeze  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.patch_transfer import (  # noqa: E402
    RECEIVERS,
    load_causal_layer,
    load_transfer_inputs,
    run_receiver_baselines,
    run_transfer_sweep,
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
    parser.add_argument("--state", choices=tuple(RECEIVERS), required=True)
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
    rows = load_prepared_rows(args.artifacts, "discovery")
    if len(rows) != 100:
        raise ValueError(f"Expected 100 discovery rows, found {len(rows)}")
    layers = experiment["patching"]["layers"]
    if layers != list(range(final_states["architecture"]["decoder_layers"])):
        raise ValueError("Chunk 3 must retain the prespecified all-layer sweep")
    full_donors, archived, runtime_full = load_transfer_inputs(
        artifact_root=args.artifacts,
        receiver=args.state,
        rows=rows,
        final_states=final_states,
    )
    print(
        f"Chunk 3 {args.state}: 100 receiver baselines + "
        "100 questions x 16 layers = 1600 patched scores"
    )
    print(
        f"Frozen readout layer: {causal_layer['selected_layer']} "
        f"({causal_layer['freeze_hash']})"
    )
    if args.dry_run:
        print("Dry run complete: tokenizer and model weights were not loaded")
        return

    tokenizer = load_tokenizer(models)
    model_key = RECEIVERS[args.state]["model_key"]
    model, device = load_public_model(model_key, models)
    print(f"Loaded {args.state} on {device}")
    batch_size = experiment["evaluation"]["sequence_batch_size"]
    baseline_path = args.artifacts / "scores" / f"{args.state}_runtime_baseline.jsonl"
    baselines = run_receiver_baselines(
        model=model,
        tokenizer=tokenizer,
        receiver=args.state,
        rows=rows,
        archived_rows=archived,
        final_states=final_states,
        causal_layer=causal_layer,
        batch_size=batch_size,
        output_path=baseline_path,
    )
    run_transfer_sweep(
        model=model,
        tokenizer=tokenizer,
        receiver=args.state,
        rows=rows,
        receiver_baselines=baselines,
        runtime_full=runtime_full,
        full_donors=full_donors,
        final_states=final_states,
        causal_layer=causal_layer,
        layers=layers,
        batch_size=batch_size,
        output_path=args.artifacts / "interventions" / f"{args.state}_layer_sweep.jsonl",
    )
    release_model(model)
    print(f"Chunk 3 {args.state} sweep complete")


if __name__ == "__main__":
    main()
