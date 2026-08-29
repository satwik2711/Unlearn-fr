#!/usr/bin/env python3
"""Run the final experiment's IDK-only 16-layer causal localization."""

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
from pivot_experiment.idk_localization import (  # noqa: E402
    freeze_causal_layer,
    load_final_freeze,
    load_localization_records,
    resolve_frozen_path,
    rebase_layer_sweep,
    run_idk_layer_sweep,
    run_runtime_baselines,
    run_self_patch_audit,
)
from pivot_experiment.final_freeze import create_final_states_freeze  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402


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
        "--dry-run",
        action="store_true",
        help="Validate the frozen workload without loading tokenizer or weights.",
    )
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    models = load_yaml(args.models_config)
    create_final_states_freeze(
        experiment_config=experiment,
        models_config=models,
        artifact_root=args.artifacts,
        output_path=args.artifacts / "freeze" / "final_states.json",
    )
    final_states = load_final_freeze(args.artifacts / "freeze" / "final_states.json")
    rows = load_prepared_rows(args.artifacts, "discovery")
    if len(rows) != 100:
        raise ValueError(f"Expected 100 discovery rows, found {len(rows)}")
    full_rows, idk_rows = load_localization_records(
        final_states, {row["example_id"] for row in rows}
    )
    layers = experiment["patching"]["layers"]
    if layers != list(range(final_states["architecture"]["decoder_layers"])):
        raise ValueError("IDK localization must cover all 16 decoder layers")
    adapter_path = resolve_frozen_path(final_states["states"]["idk"]["adapter_path"])
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing frozen IDK adapter at {adapter_path}")
    print(
        "IDK localization: 100 questions x 16 layers = 1600 patched scores; "
        "then 4 self-patch audit rows"
    )
    print(f"Adapter: {final_states['states']['idk']['adapter_id']}")
    print(f"Final states: {final_states['freeze_hash']}")
    if args.dry_run:
        print("Dry run complete: tokenizer and model weights were not loaded")
        return

    tokenizer = load_tokenizer(models)
    base_model, device = load_public_model("full", models)
    adapter_id = final_states["states"]["idk"]["adapter_id"]
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        adapter_name=adapter_id,
        is_trainable=False,
    )
    model.set_adapter(adapter_id)
    model.eval()
    print(f"Loaded FULL + {adapter_id} on {device}")
    batch_size = experiment["evaluation"]["sequence_batch_size"]
    runtime_baseline_path = (
        args.artifacts / "scores" / "idk_runtime_baselines.jsonl"
    )
    runtime_full, runtime_idk = run_runtime_baselines(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        final_states=final_states,
        batch_size=batch_size,
        output_path=runtime_baseline_path,
    )
    sweep_path = args.artifacts / "interventions" / "idk_layer_sweep.jsonl"
    run_idk_layer_sweep(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        full_rows=full_rows,
        idk_rows=idk_rows,
        freeze=final_states,
        layers=layers,
        batch_size=batch_size,
        output_path=sweep_path,
    )
    rebased_sweep_path = (
        args.artifacts / "interventions" / "idk_layer_sweep_rebased.jsonl"
    )
    rebase_layer_sweep(
        raw_sweep_path=sweep_path,
        runtime_baseline_path=runtime_baseline_path,
        runtime_full=runtime_full,
        runtime_idk=runtime_idk,
        final_states=final_states,
        output_path=rebased_sweep_path,
    )
    layer_freeze = freeze_causal_layer(
        sweep_path=rebased_sweep_path,
        final_states=final_states,
        output_path=args.artifacts / "freeze" / "causal_layer.json",
        bootstrap_resamples=experiment["evaluation"]["bootstrap_resamples"],
        seed=experiment["seed"],
    )
    print(
        f"Frozen IDK-selected layer {layer_freeze['selected_layer']} "
        f"(RF={layer_freeze['selected_metrics']['mean_fractional_recovery']:+.4f})"
    )
    run_self_patch_audit(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        idk_rows=idk_rows,
        runtime_idk=runtime_idk,
        final_states=final_states,
        layer_freeze=layer_freeze,
        batch_size=batch_size,
        tolerance=experiment["patching"]["self_patch_tolerance"],
        output_path=args.artifacts / "interventions" / "idk_self_patch_audit.jsonl",
    )
    release_model(model)
    print("Chunk 2 run complete; verify with scripts/check_stages.py --through B")


if __name__ == "__main__":
    main()
