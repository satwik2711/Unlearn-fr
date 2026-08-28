#!/usr/bin/env python3
"""Evaluate suppression-IDK checkpoints, select one, and audit reversibility."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    atomic_json,
    load_yaml,
    stable_hash,
)
from pivot_experiment.data import load_prepared_rows  # noqa: E402
from pivot_experiment.evaluate import evaluate_subset  # noqa: E402
from pivot_experiment.idk_suppression import (  # noqa: E402
    file_sha256,
    select_idk_suppression_checkpoint,
)
from pivot_experiment.metrics import score_answers  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.records import read_jsonl, read_unique  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate checkpoints without loading FULL"
    )
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    models_config = load_yaml(args.models_config)
    training_summary_path = (
        args.artifacts / "results" / "idk_suppression_training_summary.json"
    )
    if not training_summary_path.exists():
        raise SystemExit("IDK-suppression training summary is missing")
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if training_summary.get("status") != "complete":
        raise SystemExit("IDK-suppression training is not complete")
    checkpoint_paths = [Path(path) for path in training_summary["checkpoint_paths"]]
    candidates = []
    for path in checkpoint_paths:
        adapter_file = path / "adapter_model.safetensors"
        if not adapter_file.exists() or not (path / "trainer_state.pt").exists():
            raise FileNotFoundError(f"Incomplete IDK checkpoint: {path}")
        candidates.append(
            {
                "adapter_id": path.name,
                "adapter_path": str(path.resolve()),
                "adapter_hash": file_sha256(adapter_file),
            }
        )
    discovery = load_prepared_rows(args.artifacts, "discovery")
    training_pairs = read_jsonl(
        args.artifacts / "data" / "idk_suppression_training.jsonl"
    )
    refusal_by_example = {
        row["forget_example_id"]: row["refusal_answer"] for row in training_pairs
    }
    discovery = [
        {**row, "refusal_answer": refusal_by_example[row["example_id"]]}
        for row in discovery
    ]
    r_control = load_prepared_rows(args.artifacts, "r_control")
    frozen = json.loads(
        (args.artifacts / "splits" / "frozen_splits.json").read_text(encoding="utf-8")
    )
    print(f"IDK-suppression candidates: {[row['adapter_id'] for row in candidates]}")
    print("Each candidate: 100 discovery rows with activations + 100 R_control rows")
    if args.dry_run:
        print("Dry run complete: FULL weights were not loaded")
        return

    tokenizer = load_tokenizer(models_config)
    base, device = load_public_model("full", models_config)
    first, *remaining = candidates
    model = PeftModel.from_pretrained(
        base,
        first["adapter_path"],
        adapter_name=first["adapter_id"],
        is_trainable=False,
    )
    for candidate in remaining:
        model.load_adapter(
            candidate["adapter_path"],
            adapter_name=candidate["adapter_id"],
            is_trainable=False,
        )
    model.eval()
    batch_size = experiment["evaluation"]["sequence_batch_size"]
    base_spec = models_config["models"]["full"]
    tokenizer_spec = models_config["models"][models_config["tokenizer_source"]]
    for candidate in candidates:
        name = candidate["adapter_id"]
        model.set_adapter(name)
        output_root = args.artifacts / "scores" / "idk_suppression_candidates"
        evaluate_subset(
            model=model,
            tokenizer=tokenizer,
            state="idk_suppression",
            model_spec=base_spec,
            tokenizer_spec=tokenizer_spec,
            adapter_spec=candidate,
            rows=discovery,
            subset="discovery",
            output_path=output_root / f"{name}_discovery.jsonl",
            activation_dir=(
                args.artifacts / "activations" / "idk_suppression_candidates"
            ),
            batch_size=batch_size,
            capture_activations=True,
            split_hash=frozen["split_hash"],
        )
        evaluate_subset(
            model=model,
            tokenizer=tokenizer,
            state="idk_suppression",
            model_spec=base_spec,
            tokenizer_spec=tokenizer_spec,
            adapter_spec=candidate,
            rows=r_control,
            subset="r_control",
            output_path=output_root / f"{name}_r_control.jsonl",
            activation_dir=(
                args.artifacts / "activations" / "idk_suppression_candidates"
            ),
            batch_size=batch_size,
            capture_activations=False,
            split_hash=frozen["split_hash"],
        )

    selection = select_idk_suppression_checkpoint(
        experiment, args.artifacts, candidates
    )
    selected_name = selection["selected_adapter_id"]
    model.set_adapter(selected_name)
    audit_rows = discovery[:8]
    pairs = [(row["question"], row["answer"]) for row in audit_rows]
    with model.disable_adapter():
        off_before = score_answers(model, tokenizer, pairs, batch_size)
    on = score_answers(model, tokenizer, pairs, batch_size)
    with model.disable_adapter():
        off_after = score_answers(model, tokenizer, pairs, batch_size)
    full_records = read_unique(args.artifacts / "scores" / "full_discovery.jsonl")
    selected_records = read_unique(
        args.artifacts
        / "scores"
        / "idk_suppression_candidates"
        / f"{selected_name}_discovery.jsonl"
    )
    full_reference = np.array(
        [full_records[row["example_id"]]["mean_target_logprob"] for row in audit_rows]
    )
    idk_reference = np.array(
        [selected_records[row["example_id"]]["mean_target_logprob"] for row in audit_rows]
    )
    off_before_values = np.array([row["mean_target_logprob"] for row in off_before])
    on_values = np.array([row["mean_target_logprob"] for row in on])
    off_after_values = np.array([row["mean_target_logprob"] for row in off_after])
    reversibility = {
        "schema_version": 1,
        "status": "complete",
        "state": "idk_suppression",
        "device": device,
        "selected_adapter_id": selected_name,
        "selected_adapter_hash": selection["selected_adapter_hash"],
        "audit_example_ids": [row["example_id"] for row in audit_rows],
        "off_before_max_abs_from_stored_full": float(
            np.max(np.abs(off_before_values - full_reference))
        ),
        "adapter_on_max_abs_from_stored_idk": float(
            np.max(np.abs(on_values - idk_reference))
        ),
        "off_after_max_abs_from_stored_full": float(
            np.max(np.abs(off_after_values - full_reference))
        ),
        "off_before_after_max_abs": float(
            np.max(np.abs(off_before_values - off_after_values))
        ),
        "base_hash_before": training_summary["base_hash_before"],
        "base_hash_after": training_summary["base_hash_after"],
        "base_hash_match": training_summary["base_hash_match"],
        "audit_rows": [
            {
                "example_id": row["example_id"],
                "stored_full": float(full_reference[index]),
                "off_before": float(off_before_values[index]),
                "adapter_on": float(on_values[index]),
                "stored_idk": float(idk_reference[index]),
                "off_after": float(off_after_values[index]),
            }
            for index, row in enumerate(audit_rows)
        ],
    }
    reversibility["audit_hash"] = stable_hash(reversibility["audit_rows"])
    atomic_json(
        args.artifacts / "results" / "idk_suppression_reversibility.json",
        reversibility,
    )
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(
        f"Selected {selected_name}; suppression-IDK evaluation and "
        "reversibility audit complete"
    )


if __name__ == "__main__":
    main()
