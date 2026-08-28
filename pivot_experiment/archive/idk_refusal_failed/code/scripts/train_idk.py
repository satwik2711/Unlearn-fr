#!/usr/bin/env python3
"""Train the removable balanced-refusal IDK LoRA over frozen FULL."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

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
from pivot_experiment.data import prepare_idk_training_pairs  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.records import read_jsonl  # noqa: E402
from pivot_experiment.training import balanced_pair_loss, exact_base_parameter_hash  # noqa: E402


DEFAULT_IDK_CONFIG = PROJECT_ROOT / "configs" / "idk_lora.yaml"


def move_optimizer_state(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    model,
    optimizer,
    checkpoint_root: Path,
    optimizer_step: int,
    consumed_pairs: int,
    trajectory: list[dict],
    run_hash: str,
    base_hash_before: str,
) -> Path:
    destination = checkpoint_root / f"step-{optimizer_step:06d}"
    if destination.exists():
        state = torch.load(destination / "trainer_state.pt", map_location="cpu", weights_only=False)
        if state.get("run_hash") != run_hash or state.get("consumed_pairs") != consumed_pairs:
            raise ValueError(f"Existing checkpoint conflicts with current run: {destination}")
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"Incomplete temporary checkpoint requires inspection: {temporary}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save(
        {
            "schema_version": 1,
            "run_hash": run_hash,
            "optimizer_step": optimizer_step,
            "consumed_pairs": consumed_pairs,
            "trajectory": trajectory,
            "optimizer_state": optimizer.state_dict(),
            "base_hash_before": base_hash_before,
        },
        temporary / "trainer_state.pt",
    )
    temporary.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--idk-config", type=Path, default=DEFAULT_IDK_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate workload without loading FULL"
    )
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    models_config = load_yaml(args.models_config)
    idk = load_yaml(args.idk_config)
    p0_path = args.artifacts / "gates" / "p0.json"
    if not p0_path.exists() or json.loads(p0_path.read_text()).get("status") != "PASS":
        raise SystemExit("P0 must formally pass before IDK training")
    training_data = prepare_idk_training_pairs(idk, args.artifacts)
    pairs = read_jsonl(args.artifacts / "data" / "idk_training.jsonl")
    training = idk["training"]
    total_pair_presentations = len(pairs) * training["max_epochs"]
    effective = training["effective_batch_pairs"]
    micro = training["micro_batch_pairs"]
    if effective % micro:
        raise ValueError("effective_batch_pairs must be divisible by micro_batch_pairs")
    total_optimizer_steps = math.ceil(total_pair_presentations / effective)
    checkpoints = training["checkpoint_optimizer_steps"]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != total_optimizer_steps:
        raise ValueError(
            f"Checkpoint steps must be unique/sorted and end at {total_optimizer_steps}"
        )
    run_config = {
        "experiment": experiment,
        "models": models_config,
        "idk": idk,
        "training_data_hash": training_data["pairs_hash"],
    }
    run_hash = stable_hash(run_config)
    print(
        f"IDK workload: {len(pairs)} pairs x {training['max_epochs']} epochs, "
        f"effective batch {effective}, {total_optimizer_steps} optimizer steps"
    )
    print(f"Checkpoints: {checkpoints}")
    if args.dry_run:
        print("Dry run complete: FULL weights were not loaded")
        return

    random.seed(idk["seed"])
    np.random.seed(idk["seed"])
    torch.manual_seed(idk["seed"])
    tokenizer = load_tokenizer(models_config)
    base, device = load_public_model(idk["base_state"], models_config)
    lora = idk["lora"]
    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora["r"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                bias=lora["bias"],
                target_modules=lora["target_modules"],
            ),
        )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    base_hash_before = exact_base_parameter_hash(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    consumed_pairs = 0
    optimizer_step = 0
    trajectory: list[dict] = []
    if args.resume:
        state = torch.load(args.resume / "trainer_state.pt", map_location="cpu", weights_only=False)
        if state["run_hash"] != run_hash:
            raise ValueError("Resume checkpoint belongs to a different frozen run")
        if state["base_hash_before"] != base_hash_before:
            raise ValueError("Resume base-parameter hash differs")
        optimizer.load_state_dict(state["optimizer_state"])
        move_optimizer_state(optimizer, next(model.parameters()).device)
        consumed_pairs = state["consumed_pairs"]
        optimizer_step = state["optimizer_step"]
        trajectory = state["trajectory"]

    ordered_pairs = []
    for epoch in range(training["max_epochs"]):
        indices = list(range(len(pairs)))
        random.Random(idk["seed"] + epoch).shuffle(indices)
        ordered_pairs.extend(pairs[index] for index in indices)
    checkpoint_root = args.artifacts / "checkpoints" / "idk"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_pairs = 0
    objective_sum = forget_sum = retain_sum = 0.0
    while consumed_pairs < len(ordered_pairs):
        batch = ordered_pairs[consumed_pairs : consumed_pairs + micro]
        objective, forget_loss, retain_loss = balanced_pair_loss(
            model, tokenizer, batch, training["retain_lambda"]
        )
        pair_count = len(batch)
        (objective * pair_count).backward()
        accumulated_pairs += pair_count
        objective_sum += float(objective.detach().cpu()) * pair_count
        forget_sum += float(forget_loss.detach().cpu()) * pair_count
        retain_sum += float(retain_loss.detach().cpu()) * pair_count
        consumed_pairs += pair_count
        should_step = accumulated_pairs >= effective or consumed_pairs == len(ordered_pairs)
        if not should_step:
            continue
        for _, parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_pairs)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], training["max_grad_norm"]
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        record = {
            "optimizer_step": optimizer_step,
            "consumed_pair_presentations": consumed_pairs,
            "objective": objective_sum / accumulated_pairs,
            "forget_refusal_loss": forget_sum / accumulated_pairs,
            "retain_loss": retain_sum / accumulated_pairs,
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        trajectory.append(record)
        atomic_json(args.artifacts / "results" / "idk_training_trajectory.json", trajectory)
        print(
            f"IDK step {optimizer_step}/{total_optimizer_steps}: "
            f"loss={record['objective']:.4f} "
            f"forget={record['forget_refusal_loss']:.4f} retain={record['retain_loss']:.4f}"
        )
        accumulated_pairs = 0
        objective_sum = forget_sum = retain_sum = 0.0
        if optimizer_step in checkpoints:
            saved = save_checkpoint(
                model,
                optimizer,
                checkpoint_root,
                optimizer_step,
                consumed_pairs,
                trajectory,
                run_hash,
                base_hash_before,
            )
            print(f"Saved {saved}")

    base_hash_after = exact_base_parameter_hash(model)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "run_hash": run_hash,
        "device": device,
        "total_optimizer_steps": optimizer_step,
        "total_pair_presentations": consumed_pairs,
        "checkpoint_steps": checkpoints,
        "checkpoint_paths": [
            str((checkpoint_root / f"step-{step:06d}").resolve()) for step in checkpoints
        ],
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_hash_match": base_hash_before == base_hash_after,
    }
    atomic_json(args.artifacts / "results" / "idk_training_summary.json", summary)
    if not summary["base_hash_match"]:
        raise RuntimeError("Frozen FULL base parameters changed during IDK training")
    print("IDK training complete; exact base-parameter hash is unchanged")


if __name__ == "__main__":
    main()
