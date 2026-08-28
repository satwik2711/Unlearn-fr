#!/usr/bin/env python3
"""Run resumable teacher-forced FULL/RETAIN discovery evaluations."""

from __future__ import annotations

import argparse
import gc
import json
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
from pivot_experiment.evaluate import evaluate_subset  # noqa: E402
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
    parser.add_argument("--state", choices=("full", "retain", "both"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prepared inputs and print work without loading weights.",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        metavar="N",
        help="Evaluate N rows into artifacts/smoke; never counts toward P0.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    models_config = load_yaml(args.models_config)
    split_path = args.artifacts / "splits" / "frozen_splits.json"
    if not split_path.exists():
        raise SystemExit("Prepared splits are missing; run scripts/prepare.py first")
    frozen = json.loads(split_path.read_text(encoding="utf-8"))
    discovery = load_prepared_rows(args.artifacts, "discovery")
    r_control = load_prepared_rows(args.artifacts, "r_control")
    if len(discovery) != 100 or len(r_control) != 100:
        raise ValueError(
            f"Expected 100 discovery and 100 R_control rows; got {len(discovery)}, {len(r_control)}"
        )

    states = ("full", "retain") if args.state == "both" else (args.state,)
    output_root = args.artifacts / ("smoke" if args.smoke else "scores")
    selected_discovery = discovery[: args.smoke] if args.smoke else discovery
    selected_control = r_control[: args.smoke] if args.smoke else r_control
    for state in states:
        print(
            f"{state}: discovery={len(selected_discovery)}, "
            f"R_control={len(selected_control)}, "
            f"capture_FULL_discovery={state == 'full'}"
        )
    if args.dry_run:
        print("Dry run complete: no tokenizer or model weights loaded")
        return

    tokenizer = load_tokenizer(models_config)
    for state in states:
        model, device = load_public_model(state, models_config)
        print(f"Loaded {state} on {device}")
        spec = models_config["models"][state]
        evaluate_subset(
            model=model,
            tokenizer=tokenizer,
            state=state,
            model_spec=spec,
            tokenizer_spec=models_config["models"][models_config["tokenizer_source"]],
            rows=selected_discovery,
            subset="discovery",
            output_path=output_root / f"{state}_discovery.jsonl",
            activation_dir=(
                args.artifacts / "smoke" / "activations"
                if args.smoke
                else args.artifacts / "activations"
            ),
            batch_size=config["evaluation"]["sequence_batch_size"],
            capture_activations=state == "full",
            split_hash=frozen["split_hash"],
        )
        evaluate_subset(
            model=model,
            tokenizer=tokenizer,
            state=state,
            model_spec=spec,
            tokenizer_spec=models_config["models"][models_config["tokenizer_source"]],
            rows=selected_control,
            subset="r_control",
            output_path=output_root / f"{state}_r_control.jsonl",
            activation_dir=args.artifacts / "activations",
            batch_size=config["evaluation"]["sequence_batch_size"],
            capture_activations=False,
            split_hash=frozen["split_hash"],
        )
        release_model(model)
    print("Requested public evaluation completed")


if __name__ == "__main__":
    main()
