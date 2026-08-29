#!/usr/bin/env python3
"""Evaluate one frozen GradDiff candidate on discovery and R_control once."""

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
from pivot_experiment.metrics import FROZEN_PROMPT_DATE  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def require_candidate_order(candidate_id: str, models: dict, artifacts: Path) -> None:
    order = models["gd_candidate_order"]
    if candidate_id not in order:
        raise ValueError(
            f"Unknown GD candidate {candidate_id!r}; frozen order is {order}"
        )
    index = order.index(candidate_id)
    candidate_root = artifacts / "scores" / "gd_candidates"
    if (
        (candidate_root / f"{candidate_id}_discovery.jsonl").exists()
        or (candidate_root / f"{candidate_id}_r_control.jsonl").exists()
    ):
        return
    gate_path = artifacts / "gates" / "p1.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    if gate.get("status") == "PASS":
        selected = gate.get("selected_candidate_id")
        if candidate_id != selected:
            raise ValueError(
                f"P1 already selected {selected}; later GD candidates are sealed"
            )
        return
    if index == 0:
        return
    expected = gate.get("next_candidate_id")
    if gate.get("status") != "BLOCKED" or expected != candidate_id:
        raise ValueError(
            f"Run/check earlier candidates first; P1 has not authorized {candidate_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen inputs and workload without loading model weights.",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        metavar="N",
        help="Evaluate N rows into artifacts/smoke; never counts toward P1.",
    )
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    models = load_yaml(args.models_config)
    require_candidate_order(args.candidate, models, args.artifacts)
    if args.candidate not in models["models"]:
        raise ValueError(f"Missing model specification for {args.candidate}")
    if experiment["evaluation"]["prompt_date"] != FROZEN_PROMPT_DATE:
        raise ValueError(
            "Experiment prompt_date differs from the evaluator's frozen prompt date"
        )
    p0_path = args.artifacts / "gates" / "p0.json"
    if not p0_path.exists() or json.loads(p0_path.read_text()).get("status") != "PASS":
        raise SystemExit("P0 must formally pass before evaluating GD")
    split_path = args.artifacts / "splits" / "frozen_splits.json"
    if not split_path.exists():
        raise SystemExit("Prepared splits are missing; run scripts/prepare.py first")
    frozen = json.loads(split_path.read_text(encoding="utf-8"))
    discovery = load_prepared_rows(args.artifacts, "discovery")
    r_control = load_prepared_rows(args.artifacts, "r_control")
    if len(discovery) != 100 or len(r_control) != 100:
        raise ValueError(
            "Expected 100 discovery and 100 R_control rows; "
            f"got {len(discovery)} and {len(r_control)}"
        )
    selected_discovery = discovery[: args.smoke] if args.smoke else discovery
    selected_control = r_control[: args.smoke] if args.smoke else r_control
    print(
        f"{args.candidate}: discovery={len(selected_discovery)} "
        "(correct + 5 perturbed each, all-layer Q_END cache), "
        f"R_control={len(selected_control)} (correct only)"
    )
    print(f"Frozen prompt date: {FROZEN_PROMPT_DATE}")
    if args.dry_run:
        print("Dry run complete: tokenizer and model weights were not loaded")
        return

    output_root = (
        args.artifacts / "smoke" / "gd_candidates"
        if args.smoke
        else args.artifacts / "scores" / "gd_candidates"
    )
    activation_root = (
        args.artifacts / "smoke" / "activations" / "gd_candidates" / args.candidate
        if args.smoke
        else args.artifacts / "activations" / "gd_candidates" / args.candidate
    )
    tokenizer = load_tokenizer(models)
    model, device = load_public_model(args.candidate, models)
    print(f"Loaded {args.candidate} on {device}")
    model_spec = models["models"][args.candidate]
    tokenizer_spec = models["models"][models["tokenizer_source"]]
    evaluate_subset(
        model=model,
        tokenizer=tokenizer,
        state="gd",
        model_spec=model_spec,
        tokenizer_spec=tokenizer_spec,
        candidate_id=args.candidate,
        rows=selected_discovery,
        subset="discovery",
        output_path=output_root / f"{args.candidate}_discovery.jsonl",
        activation_dir=activation_root,
        batch_size=experiment["evaluation"]["sequence_batch_size"],
        capture_activations=True,
        split_hash=frozen["split_hash"],
    )
    evaluate_subset(
        model=model,
        tokenizer=tokenizer,
        state="gd",
        model_spec=model_spec,
        tokenizer_spec=tokenizer_spec,
        candidate_id=args.candidate,
        rows=selected_control,
        subset="r_control",
        output_path=output_root / f"{args.candidate}_r_control.jsonl",
        activation_dir=activation_root,
        batch_size=experiment["evaluation"]["sequence_batch_size"],
        capture_activations=False,
        split_hash=frozen["split_hash"],
    )
    release_model(model)
    print(
        f"{args.candidate} evaluation complete; run check_gates.py --through P1"
    )


if __name__ == "__main__":
    main()
