#!/usr/bin/env python3
"""Run one grouped, resumable held-out confirmation state."""

from __future__ import annotations

import argparse
import gc
import json
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
from pivot_experiment.confirmation import (  # noqa: E402
    CONFIRMATION_RECEIVERS,
    load_confirmation_freeze,
    load_confirmation_protocol,
    run_confirmation_generations,
    run_confirmation_scores,
)
from pivot_experiment.data import load_prepared_rows  # noqa: E402
from pivot_experiment.idk_localization import load_final_freeze, resolve_frozen_path  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.patch_transfer import load_causal_layer  # noqa: E402
from pivot_experiment.steering import load_direction  # noqa: E402


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("full", *CONFIRMATION_RECEIVERS), required=True)
    parser.add_argument("--phase", choices=("all", "scores", "generations"), default="all")
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
    confirmation = load_confirmation_freeze(
        args.artifacts / "freeze" / "confirmation.json"
    )
    direction_path = args.artifacts / "directions" / "full_minus_idk.safetensors"
    learned, direction_manifest = load_direction(
        direction_path,
        direction_path.with_suffix(".manifest.json"),
        final_states,
        causal_layer,
    )
    randoms, random_manifest, execution = load_confirmation_protocol(
        confirmation_freeze=confirmation,
        direction_manifest=direction_manifest,
        random_tensor_path=args.artifacts / "directions" / "confirmation_random.safetensors",
        random_manifest_path=args.artifacts / "directions" / "confirmation_random.manifest.json",
        execution_freeze_path=args.artifacts / "freeze" / "confirmation_execution.json",
    )
    rows = load_prepared_rows(args.artifacts, "confirmation")
    if len(rows) != 100 or {row["author_id"] for row in rows} != set(
        confirmation["confirmation_authors"]
    ):
        raise ValueError("Confirmation rows differ from the sealed author split")
    if args.state != "full" and not args.dry_run:
        full_manifest_path = (
            args.artifacts / "confirmation" / "scores" / "full.manifest.json"
        )
        if not full_manifest_path.is_file():
            raise ValueError("Run the FULL confirmation denominator before receiver states")
        full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
        if (
            full_manifest.get("status") != "complete"
            or full_manifest.get("completed_rows") != 100
            or full_manifest.get("config", {}).get("confirmation_freeze_hash")
            != confirmation["confirmation_freeze_hash"]
            or full_manifest.get("config", {}).get("execution_freeze_hash")
            != execution["execution_freeze_hash"]
        ):
            raise ValueError("FULL confirmation denominator is incomplete or incompatible")
    score_count = 100 if args.state == "full" else 700
    generation_count = 100 if args.state == "full" else 200
    print(
        f"Chunk 5 {args.state}: scores={score_count}, generations={generation_count}, "
        f"phase={args.phase}"
    )
    print(
        f"Frozen layer={confirmation['layer']}, alpha={confirmation['selected_alpha']}, "
        f"random_controls={len(randoms)}"
    )
    if args.dry_run:
        print("Dry run complete: confirmation workload opened; model weights not loaded")
        return

    tokenizer = load_tokenizer(models)
    if args.state == "idk":
        base, device = load_public_model("full", models)
        adapter_id = final_states["states"]["idk"]["adapter_id"]
        model = PeftModel.from_pretrained(
            base,
            str(resolve_frozen_path(final_states["states"]["idk"]["adapter_path"])),
            adapter_name=adapter_id,
            is_trainable=False,
        )
        model.set_adapter(adapter_id)
        model.eval()
    else:
        model_key = "gd_02" if args.state == "gd02" else args.state
        model, device = load_public_model(model_key, models)
    print(f"Loaded {args.state} on {device}")
    if args.phase in {"all", "scores"}:
        run_confirmation_scores(
            model=model,
            tokenizer=tokenizer,
            state=args.state,
            rows=rows,
            learned_direction=learned,
            random_directions=randoms,
            final_states=final_states,
            confirmation_freeze=confirmation,
            execution=execution,
            direction_manifest=direction_manifest,
            random_manifest=random_manifest,
            batch_size=experiment["evaluation"]["sequence_batch_size"],
            output_path=args.artifacts / "confirmation" / "scores" / f"{args.state}.jsonl",
        )
    if args.phase in {"all", "generations"}:
        run_confirmation_generations(
            model=model,
            tokenizer=tokenizer,
            state=args.state,
            rows=rows,
            learned_direction=learned,
            final_states=final_states,
            confirmation_freeze=confirmation,
            execution=execution,
            direction_manifest=direction_manifest,
            output_path=args.artifacts / "confirmation" / "generations" / f"{args.state}.jsonl",
        )
    release_model(model)
    print(f"Chunk 5 {args.state} phase={args.phase} complete")


if __name__ == "__main__":
    main()
