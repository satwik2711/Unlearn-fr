#!/usr/bin/env python3
"""Run the fixed layer-10 post-hoc steering robustness check.

This analysis is intentionally separate from the frozen layer-14 experiment.
It reuses discovery activation caches to build a FULL-minus-IDK direction, then
scores that direction and five norm-matched random controls on the already-open
confirmation authors.  The result is exploratory/post-hoc, not a second held-out
confirmation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file

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
from pivot_experiment.final_freeze import file_sha256  # noqa: E402
from pivot_experiment.idk_localization import (  # noqa: E402
    clustered_interval,
    load_final_freeze,
    resolve_frozen_path,
)
from pivot_experiment.metrics import encode_answer, score_encoded_batch  # noqa: E402
from pivot_experiment.models import load_public_model, load_tokenizer  # noqa: E402
from pivot_experiment.patching import ActivationStore  # noqa: E402
from pivot_experiment.records import append_jsonl, read_jsonl, read_unique  # noqa: E402


MID_LAYER = 10
ALPHA = 1.0
RANDOM_SEEDS = (1042, 2042, 3042, 4042, 5042)
STATES = ("idk", "gd02", "retain")


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def output_root(artifacts: Path) -> Path:
    return artifacts / "posthoc_midlayer"


def prepare(artifacts: Path) -> dict[str, Any]:
    final_states = load_final_freeze(artifacts / "freeze" / "final_states.json")
    rows = load_prepared_rows(artifacts, "discovery")
    expected_ids = {row["example_id"] for row in rows}
    full_rows = read_unique(
        resolve_frozen_path(final_states["states"]["full"]["discovery_scores"])
    )
    idk_rows = read_unique(
        resolve_frozen_path(final_states["states"]["idk"]["discovery_scores"])
    )
    if set(full_rows) != expected_ids or set(idk_rows) != expected_ids:
        raise ValueError("Mid-layer direction inputs differ from discovery")

    architecture = final_states["architecture"]
    full_store = ActivationStore(
        full_rows,
        architecture["decoder_layers"],
        architecture["hidden_size"],
        resolve_frozen_path(final_states["states"]["full"]["activations"]["directory"]),
    )
    idk_store = ActivationStore(
        idk_rows,
        architecture["decoder_layers"],
        architecture["hidden_size"],
        resolve_frozen_path(final_states["states"]["idk"]["activations"]["directory"]),
    )
    ordered_ids = sorted(expected_ids)
    differences = []
    provenance = []
    for example_id in ordered_ids:
        full = full_rows[example_id]
        idk = idk_rows[example_id]
        if (
            full["prompt_hash"] != idk["prompt_hash"]
            or full["author_id"] != idk["author_id"]
            or full["split_hash"] != final_states["split"]["split_hash"]
            or idk["split_hash"] != final_states["split"]["split_hash"]
        ):
            raise ValueError(f"Mid-layer provenance drift at {example_id}")
        differences.append(
            full_store.vector(example_id, MID_LAYER).float()
            - idk_store.vector(example_id, MID_LAYER).float()
        )
        provenance.append((example_id, full["prompt_hash"], full["author_id"]))
    learned = torch.stack(differences).mean(dim=0).contiguous()
    norm = float(torch.linalg.vector_norm(learned).item())
    tensors = {"learned": learned}
    random_records = []
    for seed in RANDOM_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vector = torch.randn(learned.shape, generator=generator, dtype=torch.float32)
        vector = (vector / torch.linalg.vector_norm(vector) * norm).contiguous()
        name = f"random_{seed}"
        tensors[name] = vector
        random_records.append(
            {"condition": name, "seed": seed, "tensor_sha256": tensor_hash(vector)}
        )

    root = output_root(artifacts)
    tensor_path = root / "directions" / "full_minus_idk_layer10.safetensors"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor_path.exists():
        existing = load_file(tensor_path, device="cpu")
        if set(existing) != set(tensors) or any(
            not torch.equal(existing[key].float(), value)
            for key, value in tensors.items()
        ):
            raise ValueError("Refusing to overwrite changed mid-layer directions")
    else:
        temporary = tensor_path.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary)
        temporary.replace(tensor_path)

    layer_rows = read_jsonl(artifacts / "analysis" / "tables" / "layer_curves.jsonl")
    layer_summary = next(
        row
        for row in layer_rows
        if row.get("state") == "idk" and row.get("layer") == MID_LAYER
    )
    payload = {
        "schema_version": 1,
        "status": "FROZEN_POSTHOC",
        "analysis_role": "posthoc_midnetwork_robustness_check",
        "confirmation_previously_opened": True,
        "fresh_confirmation_claim_allowed": False,
        "selection_rule": "fixed_center_of_proposed_layers_9_to_11",
        "layer": MID_LAYER,
        "alpha": ALPHA,
        "token_position": "Q_END",
        "direction": "mean_FULL_minus_IDK_on_discovery",
        "example_count": len(ordered_ids),
        "source_examples_hash": stable_hash(provenance),
        "direction_norm": norm,
        "learned_tensor_sha256": tensor_hash(learned),
        "random_controls": random_records,
        "tensor_file": tensor_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "tensor_file_sha256": file_sha256(tensor_path),
        "final_states_hash": final_states["freeze_hash"],
        "existing_exact_patch_layer10": layer_summary,
        "no_layer_or_alpha_search": True,
        "generation": False,
    }
    protocol = {**payload, "protocol_hash": stable_hash(payload)}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Refusing to overwrite changed mid-layer protocol")
    else:
        atomic_json(protocol_path, protocol)
    return protocol


def load_protocol(artifacts: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    root = output_root(artifacts)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    tensor_path = PROJECT_ROOT / protocol["tensor_file"]
    tensors = {
        key: value.float()
        for key, value in load_file(tensor_path, device="cpu").items()
    }
    if (
        stable_hash(payload) != protocol.get("protocol_hash")
        or file_sha256(tensor_path) != protocol.get("tensor_file_sha256")
        or tensor_hash(tensors["learned"]) != protocol.get("learned_tensor_sha256")
        or protocol.get("layer") != MID_LAYER
        or protocol.get("alpha") != ALPHA
    ):
        raise ValueError("Mid-layer protocol is invalid or changed")
    return protocol, tensors


def load_state_model(state: str, models: dict, final_states: dict):
    if state == "idk":
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
        return model, device
    return load_public_model("gd_02" if state == "gd02" else state, models)


def score_state(
    state: str,
    artifacts: Path,
    experiment: dict,
    models: dict,
) -> None:
    protocol, tensors = load_protocol(artifacts)
    final_states = load_final_freeze(artifacts / "freeze" / "final_states.json")
    rows = load_prepared_rows(artifacts, "confirmation")
    if len(rows) != 100:
        raise ValueError("Expected 100 existing confirmation rows")
    original = read_jsonl(artifacts / "confirmation" / "scores" / f"{state}.jsonl")
    baselines = {
        row["example_id"]: row for row in original if row["condition"] == "baseline"
    }
    if set(baselines) != {row["example_id"] for row in rows}:
        raise ValueError(f"Original {state} confirmation baseline is incomplete")

    output_path = output_root(artifacts) / "scores" / f"{state}.jsonl"
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate post-hoc scores for {state}")
    expected = len(rows) * len(tensors)
    print(f"{state}/midlayer_sanity: {len(completed)}/{expected} already complete")
    model, device = load_state_model(state, models, final_states)
    tokenizer = load_tokenizer(models)
    print(f"Loaded {state} on {device}; fixed layer={MID_LAYER}, alpha={ALPHA}")
    batch_size = int(experiment["evaluation"]["sequence_batch_size"])
    for condition, direction in sorted(tensors.items()):
        pending = [
            row for row in rows if f"{condition}:{row['example_id']}" not in completed
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            encoded = [
                encode_answer(tokenizer, row["question"], row["answer"])
                for row in batch
            ]
            scores, _ = score_encoded_batch(
                model,
                tokenizer,
                encoded,
                steer_layer=MID_LAYER,
                steer_direction=direction,
                steer_alpha=ALPHA,
            )
            records = []
            for prepared, score in zip(batch, scores, strict=True):
                baseline = baselines[prepared["example_id"]]
                if score["prompt_hash"] != baseline["prompt_hash"]:
                    raise ValueError(
                        f"Prompt drift at {state}/{prepared['example_id']}"
                    )
                records.append(
                    {
                        "schema_version": 1,
                        "cell_id": f"{condition}:{prepared['example_id']}",
                        "state": state,
                        "subset": "previously_opened_confirmation_posthoc",
                        "author_id": prepared["author_id"],
                        "example_id": prepared["example_id"],
                        "condition": condition,
                        "layer": MID_LAYER,
                        "alpha": ALPHA,
                        "mean_target_logprob": score["mean_target_logprob"],
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "direction_tensor_sha256": tensor_hash(direction),
                        "protocol_hash": protocol["protocol_hash"],
                        "final_states_hash": final_states["freeze_hash"],
                    }
                )
            append_jsonl(output_path, records)
            completed.update(row["cell_id"] for row in records)
            if len(completed) % 100 == 0 or len(completed) == expected:
                print(f"{state}/midlayer_sanity: {len(completed)}/{expected}")
    release_model(model)
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} post-hoc scores for {state}")


def condition_metric(
    rows: list[dict],
    baselines: dict[str, dict],
    full: dict[str, dict],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["author_id"]].append(row)
    if len(grouped) != 5 or any(len(values) != 20 for values in grouped.values()):
        raise ValueError("Post-hoc condition lacks five complete authors")
    raw = {}
    headroom = {}
    for author, values in sorted(grouped.items()):
        raw[author] = float(
            np.mean(
                [
                    row["mean_target_logprob"]
                    - baselines[row["example_id"]]["mean_target_logprob"]
                    for row in values
                ]
            )
        )
        headroom[author] = float(
            np.mean(
                [
                    full[row["example_id"]]["mean_target_logprob"]
                    - baselines[row["example_id"]]["mean_target_logprob"]
                    for row in values
                ]
            )
        )
    unstable = {author: value for author, value in headroom.items() if value <= 0.02}
    rf = {
        author: raw[author] / headroom[author]
        for author in raw
        if author not in unstable
    }
    values = list(rf.values())
    return {
        "mean_raw_recovery": float(np.mean(list(raw.values()))),
        "mean_fractional_recovery": float(np.mean(values)) if values else None,
        "author_fractional_recovery": rf,
        "positive_authors": sum(value > 0 for value in values),
        "fractional_author_clustered_ci_95": clustered_interval(
            values, resamples, seed
        ),
        "unstable_authors": unstable,
    }


def finalize(artifacts: Path, experiment: dict) -> dict[str, Any]:
    protocol, tensors = load_protocol(artifacts)
    full_rows = read_jsonl(artifacts / "confirmation" / "scores" / "full.jsonl")
    full = {
        row["example_id"]: row for row in full_rows if row["condition"] == "baseline"
    }
    conditions = sorted(tensors)
    metrics = {}
    source_hashes = {}
    for state in STATES:
        original_path = artifacts / "confirmation" / "scores" / f"{state}.jsonl"
        score_path = output_root(artifacts) / "scores" / f"{state}.jsonl"
        original = read_jsonl(original_path)
        baselines = {
            row["example_id"]: row for row in original if row["condition"] == "baseline"
        }
        scored = read_jsonl(score_path)
        if len(scored) != 100 * len(conditions):
            raise ValueError(f"Incomplete post-hoc scores for {state}")
        metrics[state] = {}
        for condition in conditions:
            selected = [row for row in scored if row["condition"] == condition]
            metrics[state][condition] = condition_metric(
                selected,
                baselines,
                full,
                int(experiment["evaluation"]["bootstrap_resamples"]),
                int(experiment["seed"]),
            )
        source_hashes[state] = {
            "posthoc_scores_sha256": file_sha256(score_path),
            "original_baselines_sha256": file_sha256(original_path),
        }

    transfer = {}
    for condition in conditions:
        gd = metrics["gd02"][condition]["author_fractional_recovery"]
        retain = metrics["retain"][condition]["author_fractional_recovery"]
        common = sorted(set(gd) & set(retain))
        author_values = {author: gd[author] - retain[author] for author in common}
        values = list(author_values.values())
        transfer[condition] = {
            "mean_differential_fractional_recovery": float(np.mean(values)),
            "author_differential": author_values,
            "positive_authors": sum(value > 0 for value in values),
            "author_clustered_ci_95": clustered_interval(
                values,
                int(experiment["evaluation"]["bootstrap_resamples"]),
                int(experiment["seed"]),
            ),
        }
    random_conditions = [
        condition for condition in conditions if condition != "learned"
    ]
    idk_value = metrics["idk"]["learned"]["mean_fractional_recovery"]
    transfer_value = transfer["learned"]["mean_differential_fractional_recovery"]
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_POSTHOC",
        "analysis_role": "posthoc_midnetwork_robustness_check",
        "fresh_confirmation_claim_allowed": False,
        "layer": MID_LAYER,
        "alpha": ALPHA,
        "protocol_hash": protocol["protocol_hash"],
        "idk_learned": metrics["idk"]["learned"],
        "gd02_learned": metrics["gd02"]["learned"],
        "retain_learned": metrics["retain"]["learned"],
        "learned_transfer": transfer["learned"],
        "idk_learned_rank_among_six": 1
        + sum(
            metrics["idk"][condition]["mean_fractional_recovery"] > idk_value
            for condition in random_conditions
        ),
        "transfer_learned_rank_among_six": 1
        + sum(
            transfer[condition]["mean_differential_fractional_recovery"]
            > transfer_value
            for condition in random_conditions
        ),
        "all_condition_metrics": metrics,
        "all_transfer_conditions": transfer,
        "source_hashes": source_hashes,
        "interpretation_rule": (
            "A negative or null learned GD02-minus-RETAIN differential strengthens only the "
            "assay-specific non-transfer robustness claim; it cannot establish erasure."
        ),
    }
    result = {**payload, "result_hash": stable_hash(payload)}
    output_path = output_root(artifacts) / "result.json"
    if output_path.exists():
        if json.loads(output_path.read_text(encoding="utf-8")) != result:
            raise ValueError("Refusing to overwrite changed post-hoc result")
    else:
        atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "finalize", "all"))
    parser.add_argument("--state", choices=STATES)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    experiment = load_yaml(args.config)
    models = load_yaml(args.models_config)

    if args.phase in {"prepare", "all"}:
        protocol = prepare(args.artifacts)
        print(
            f"Prepared post-hoc layer {protocol['layer']} direction: "
            f"norm={protocol['direction_norm']:.6f}, alpha={protocol['alpha']:g}"
        )
    if args.phase == "score":
        if args.state is None:
            parser.error("score requires --state")
        score_state(args.state, args.artifacts, experiment, models)
    elif args.phase == "all":
        for state in STATES:
            score_state(state, args.artifacts, experiment, models)
        result = finalize(args.artifacts, experiment)
        print(
            f"Layer-10 post-hoc complete: IDK RF={result['idk_learned']['mean_fractional_recovery']:+.4f}, "
            f"transfer={result['learned_transfer']['mean_differential_fractional_recovery']:+.4f}"
        )
    elif args.phase == "finalize":
        result = finalize(args.artifacts, experiment)
        print(
            f"Layer-10 post-hoc complete: IDK RF={result['idk_learned']['mean_fractional_recovery']:+.4f}, "
            f"transfer={result['learned_transfer']['mean_differential_fractional_recovery']:+.4f}"
        )


if __name__ == "__main__":
    main()
