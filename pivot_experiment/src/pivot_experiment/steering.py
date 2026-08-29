"""IDK-calibrated direction construction and discovery-only alpha selection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .config import atomic_json, stable_hash
from .final_freeze import file_sha256
from .idk_localization import resolve_frozen_path
from .metrics import EVALUATOR_VERSION, FROZEN_PROMPT_DATE, encode_answer, score_encoded_batch
from .patching import ActivationStore
from .records import append_jsonl, initialize_manifest, read_jsonl, read_unique


STEERING_RECEIVERS = ("idk", "retain")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_direction(
    *,
    rows: list[dict],
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    tensor_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Build mean FULL-minus-IDK Q_END direction without loading a model."""

    expected_ids = {row["example_id"] for row in rows}
    full_rows = read_unique(
        resolve_frozen_path(final_states["states"]["full"]["discovery_scores"])
    )
    idk_rows = read_unique(
        resolve_frozen_path(final_states["states"]["idk"]["discovery_scores"])
    )
    if set(full_rows) != expected_ids or set(idk_rows) != expected_ids:
        raise ValueError("Direction inputs differ from the frozen discovery subset")
    architecture = final_states["architecture"]
    layer = causal_layer["selected_layer"]
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
    source_rows = []
    for example_id in ordered_ids:
        full = full_rows[example_id]
        idk = idk_rows[example_id]
        if (
            full["prompt_hash"] != idk["prompt_hash"]
            or full["author_id"] != idk["author_id"]
            or full["activation_shape"] != idk["activation_shape"]
            or full["split_hash"] != final_states["split"]["split_hash"]
            or idk["split_hash"] != final_states["split"]["split_hash"]
        ):
            raise ValueError(f"Direction provenance drift at {example_id}")
        differences.append(
            full_store.vector(example_id, layer).float()
            - idk_store.vector(example_id, layer).float()
        )
        source_rows.append(
            {
                "example_id": example_id,
                "author_id": full["author_id"],
                "prompt_hash": full["prompt_hash"],
                "full_activation_file": Path(full["activation_file"]).name,
                "full_activation_row": full["activation_row"],
                "idk_activation_file": Path(idk["activation_file"]).name,
                "idk_activation_row": idk["activation_row"],
            }
        )
    direction = torch.stack(differences).mean(dim=0).contiguous()
    if direction.shape != (architecture["hidden_size"],) or not torch.isfinite(direction).all():
        raise ValueError("Constructed direction has invalid shape or values")
    tensor_hash = _tensor_sha256(direction)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor_path.exists():
        existing = load_file(tensor_path, device="cpu").get("direction")
        if existing is None or not torch.equal(existing.float(), direction):
            raise ValueError(f"Refusing to overwrite changed direction: {tensor_path}")
    else:
        temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
        save_file({"direction": direction}, temporary)
        temporary.replace(tensor_path)
    payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "scientific_gate": False,
        "name": "full_minus_idk",
        "sign": "FULL_minus_IDK",
        "positive_alpha_meaning": "move_receiver_Q_END_toward_FULL_from_IDK",
        "layer": layer,
        "token_position": "Q_END",
        "source_state_full": final_states["states"]["full"]["model"],
        "source_state_idk": {
            "base_model": final_states["states"]["idk"]["base_model"],
            "adapter_id": final_states["states"]["idk"]["adapter_id"],
            "adapter_hash": final_states["states"]["idk"]["adapter_hash"],
        },
        "example_count": len(ordered_ids),
        "source_examples_hash": stable_hash(source_rows),
        "source_example_ids_hash": stable_hash(ordered_ids),
        "prompt_hashes_hash": stable_hash(
            [full_rows[example_id]["prompt_hash"] for example_id in ordered_ids]
        ),
        "hidden_size": architecture["hidden_size"],
        "dtype": "float32",
        "l2_norm": float(torch.linalg.vector_norm(direction).item()),
        "tensor_sha256": tensor_hash,
        "file_sha256": file_sha256(tensor_path),
        "tensor_name": "direction",
        "tensor_path": tensor_path.resolve().relative_to(
            resolve_frozen_path(".")
        ).as_posix(),
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "gd02_activation_or_score_read": False,
        "confirmation_read": False,
    }
    result = {**payload, "direction_freeze_hash": stable_hash(payload)}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(f"Refusing to overwrite changed direction manifest: {manifest_path}")
        return existing
    atomic_json(manifest_path, result)
    return result


def load_direction(
    tensor_path: Path,
    manifest_path: Path,
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not tensor_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Missing frozen steering direction or manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        key: value for key, value in manifest.items() if key != "direction_freeze_hash"
    }
    direction = load_file(tensor_path, device="cpu").get("direction")
    if (
        direction is None
        or manifest.get("status") != "FROZEN"
        or stable_hash(payload) != manifest.get("direction_freeze_hash")
        or manifest.get("final_states_hash") != final_states["freeze_hash"]
        or manifest.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
        or manifest.get("layer") != causal_layer["selected_layer"]
        or manifest.get("tensor_sha256") != _tensor_sha256(direction)
        or manifest.get("file_sha256") != file_sha256(tensor_path)
        or tuple(direction.shape) != (final_states["architecture"]["hidden_size"],)
    ):
        raise ValueError("Frozen steering direction is invalid or has changed")
    return direction.float(), manifest


def load_receiver_archives(
    receiver: str, final_states: dict[str, Any], expected: dict[str, set[str]]
) -> dict[str, dict[str, dict]]:
    if receiver not in STEERING_RECEIVERS:
        raise ValueError(f"Unknown steering receiver: {receiver}")
    state = final_states["states"][receiver]
    result = {
        "discovery": read_unique(resolve_frozen_path(state["discovery_scores"])),
        "r_control": read_unique(resolve_frozen_path(state["r_control_scores"])),
    }
    expected_state = receiver
    for subset, records in result.items():
        if set(records) != expected[subset]:
            raise ValueError(f"{receiver}/{subset} differs from the frozen workload")
        for example_id, row in records.items():
            if (
                row.get("state") != expected_state
                or row.get("subset") != subset
                or row.get("split_hash") != final_states["split"]["split_hash"]
            ):
                raise ValueError(f"Invalid archived row at {receiver}/{example_id}")
    return result


def _receiver_model_spec(receiver: str, final_states: dict[str, Any]) -> dict[str, str]:
    state = final_states["states"][receiver]
    return state["base_model"] if receiver == "idk" else state["model"]


def run_alpha_baselines(
    *,
    model,
    tokenizer,
    receiver: str,
    rows_by_subset: dict[str, list[dict]],
    archived: dict[str, dict[str, dict]],
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    direction_manifest: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> dict[str, dict]:
    model_spec = _receiver_model_spec(receiver, final_states)
    workload = [
        row for subset in ("discovery", "r_control") for row in rows_by_subset[subset]
    ]
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "receiver": receiver,
        "model": model_spec,
        "adapter_id": final_states["states"][receiver].get("adapter_id"),
        "adapter_hash": final_states["states"][receiver].get("adapter_hash"),
        "subsets": ["discovery", "r_control"],
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in workload]
        ),
        "intervention": "none",
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "split_hash": final_states["split"]["split_hash"],
        "batch_size": batch_size,
        "generation": False,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate {receiver} alpha-baseline cells")
    expected_rows = len(workload)
    for subset in ("discovery", "r_control"):
        pending = [
            row
            for row in rows_by_subset[subset]
            if f"{subset}:{row['example_id']}" not in completed
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            encoded = [
                encode_answer(tokenizer, row["question"], row["answer"])
                for row in batch
            ]
            scores, _ = score_encoded_batch(model, tokenizer, encoded)
            records = []
            for prepared, score in zip(batch, scores, strict=True):
                example_id = prepared["example_id"]
                if score["prompt_hash"] != archived[subset][example_id]["prompt_hash"]:
                    raise ValueError(f"Prompt drift at {receiver}/{subset}/{example_id}")
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": f"{subset}:{example_id}",
                        "state": receiver,
                        "subset": subset,
                        "author_id": prepared["author_id"],
                        "example_id": example_id,
                        "intervention": "none",
                        "model_id": model_spec["repo_id"],
                        "model_revision": model_spec["revision"],
                        "adapter_id": final_states["states"][receiver].get("adapter_id"),
                        "mean_target_logprob": score["mean_target_logprob"],
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "split_hash": final_states["split"]["split_hash"],
                        "final_states_hash": final_states["freeze_hash"],
                        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
                        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
                    }
                )
            append_jsonl(output_path, records)
            completed.update(record["cell_id"] for record in records)
            manifest.update(status="running", completed_rows=len(completed), expected_rows=expected_rows)
            atomic_json(manifest_path, manifest)
            print(f"{receiver}/alpha_baselines: {len(completed)}/{expected_rows}")
    if len(completed) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} {receiver} alpha baselines")
    manifest.update(status="complete", completed_rows=expected_rows, expected_rows=expected_rows)
    atomic_json(manifest_path, manifest)
    result = read_unique(output_path, key="cell_id")
    for record in result.values():
        subset = record.get("subset")
        example_id = record.get("example_id")
        if (
            subset not in {"discovery", "r_control"}
            or example_id not in archived[subset]
            or record.get("state") != receiver
            or record.get("intervention") != "none"
            or record.get("model_id") != model_spec["repo_id"]
            or record.get("model_revision") != model_spec["revision"]
            or record.get("prompt_hash") != archived[subset][example_id]["prompt_hash"]
            or record.get("final_states_hash") != final_states["freeze_hash"]
            or record.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
            or record.get("direction_freeze_hash")
            != direction_manifest["direction_freeze_hash"]
        ):
            raise ValueError(f"Invalid resumed alpha baseline: {record.get('cell_id')}")
    return result


def run_alpha_sweep(
    *,
    model,
    tokenizer,
    receiver: str,
    rows_by_subset: dict[str, list[dict]],
    baselines: dict[str, dict],
    direction: torch.Tensor,
    direction_manifest: dict[str, Any],
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    alphas: list[float],
    batch_size: int,
    output_path: Path,
) -> None:
    model_spec = _receiver_model_spec(receiver, final_states)
    baseline_manifest = json.loads(
        (output_path.parents[1] / "scores" / f"{receiver}_alpha_baselines.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    workload = [
        row for subset in ("discovery", "r_control") for row in rows_by_subset[subset]
    ]
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "receiver": receiver,
        "model": model_spec,
        "adapter_id": final_states["states"][receiver].get("adapter_id"),
        "adapter_hash": final_states["states"][receiver].get("adapter_hash"),
        "intervention": "add_full_minus_idk_at_q_end",
        "layer": causal_layer["selected_layer"],
        "token_position": "Q_END",
        "alphas": alphas,
        "subsets": ["discovery", "r_control"],
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in workload]
        ),
        "baseline_manifest_hash": stable_hash(baseline_manifest),
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "direction_tensor_sha256": direction_manifest["tensor_sha256"],
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "split_hash": final_states["split"]["split_hash"],
        "batch_size": batch_size,
        "generation": False,
        "gd02_read": False,
        "confirmation_read": False,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate {receiver} alpha-sweep cells")
    expected_rows = len(workload) * len(alphas)
    print(f"{receiver}/alpha_sweep: {len(completed)}/{expected_rows} already complete")
    for alpha in alphas:
        for subset in ("discovery", "r_control"):
            pending = [
                row
                for row in rows_by_subset[subset]
                if f"{subset}:{row['example_id']}:alpha-{alpha:g}" not in completed
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
                    steer_layer=causal_layer["selected_layer"],
                    steer_direction=direction,
                    steer_alpha=alpha,
                )
                records = []
                for prepared, score in zip(batch, scores, strict=True):
                    example_id = prepared["example_id"]
                    baseline = baselines[f"{subset}:{example_id}"]
                    if score["prompt_hash"] != baseline["prompt_hash"]:
                        raise ValueError(f"Prompt drift during {receiver}/{subset}/{example_id}")
                    records.append(
                        {
                            "schema_version": 1,
                            "run_id": config_hash,
                            "cell_id": f"{subset}:{example_id}:alpha-{alpha:g}",
                            "state": receiver,
                            "subset": subset,
                            "author_id": prepared["author_id"],
                            "example_id": example_id,
                            "intervention": "add_full_minus_idk_at_q_end",
                            "layer": causal_layer["selected_layer"],
                            "token_position": "Q_END",
                            "alpha": alpha,
                            "model_id": model_spec["repo_id"],
                            "model_revision": model_spec["revision"],
                            "adapter_id": final_states["states"][receiver].get("adapter_id"),
                            "mean_target_logprob": score["mean_target_logprob"],
                            "baseline_mean_target_logprob": baseline["mean_target_logprob"],
                            "raw_change": score["mean_target_logprob"]
                            - baseline["mean_target_logprob"],
                            "token_count": score["token_count"],
                            "prompt_hash": score["prompt_hash"],
                            "split_hash": final_states["split"]["split_hash"],
                            "final_states_hash": final_states["freeze_hash"],
                            "causal_layer_freeze_hash": causal_layer["freeze_hash"],
                            "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
                        }
                    )
                append_jsonl(output_path, records)
                completed.update(record["cell_id"] for record in records)
                manifest.update(status="running", completed_rows=len(completed), expected_rows=expected_rows)
                atomic_json(manifest_path, manifest)
                print(f"{receiver}/alpha_sweep: {len(completed)}/{expected_rows}")
    if len(completed) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} {receiver} alpha cells")
    manifest.update(status="complete", completed_rows=expected_rows, expected_rows=expected_rows)
    atomic_json(manifest_path, manifest)


def _discovery_recovery(
    *,
    alpha_rows: list[dict],
    baselines: dict[str, dict],
    full_baselines: dict[str, dict],
) -> dict[str, Any]:
    by_author: dict[str, list[dict]] = defaultdict(list)
    for row in alpha_rows:
        by_author[row["author_id"]].append(row)
    if len(by_author) != 5 or any(len(rows) != 20 for rows in by_author.values()):
        raise ValueError("Alpha condition lacks five complete discovery authors")
    author_raw = {}
    author_headroom = {}
    for author, rows in sorted(by_author.items()):
        author_raw[author] = float(np.mean([row["raw_change"] for row in rows]))
        author_headroom[author] = float(
            np.mean(
                [
                    full_baselines[row["example_id"]]["mean_target_logprob"]
                    - baselines[f"discovery:{row['example_id']}"]["mean_target_logprob"]
                    for row in rows
                ]
            )
        )
    unstable = {author: value for author, value in author_headroom.items() if value <= 0.02}
    author_rf = {
        author: author_raw[author] / author_headroom[author]
        for author in author_raw
        if author not in unstable
    }
    return {
        "mean_raw_recovery": float(np.mean(list(author_raw.values()))),
        "mean_fractional_recovery": (
            float(np.mean(list(author_rf.values()))) if author_rf else None
        ),
        "author_raw_recovery": author_raw,
        "author_headroom": author_headroom,
        "author_fractional_recovery": author_rf,
        "unstable_authors": unstable,
        "positive_authors": sum(value > 0 for value in author_rf.values()),
    }


def _state_revision_freeze(final_states: dict[str, Any]) -> dict[str, Any]:
    return {
        "full": final_states["states"]["full"]["model"],
        "idk": {
            "base_model": final_states["states"]["idk"]["base_model"],
            "adapter_id": final_states["states"]["idk"]["adapter_id"],
            "adapter_hash": final_states["states"]["idk"]["adapter_hash"],
        },
        "gd02": {
            "model": final_states["states"]["gd02"]["model"],
            "candidate_id": final_states["states"]["gd02"]["candidate_id"],
        },
        "retain": final_states["states"]["retain"]["model"],
    }


def finalize_alpha_selection(
    *,
    artifact_root: Path,
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    direction_manifest: dict[str, Any],
    alphas: list[float],
    beta: float,
    random_seeds: list[int],
    bootstrap_resamples: int,
    analysis_output: Path,
    confirmation_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if final_states.get("confirmation", {}).get("status") != "sealed":
        raise ValueError("Confirmation was opened before alpha selection")
    full_rows = read_jsonl(artifact_root / "scores" / "idk_runtime_baselines.jsonl")
    full_baselines = {
        row["example_id"]: row for row in full_rows if row.get("state") == "full"
    }
    if len(full_baselines) != 100:
        raise ValueError("Missing current-evaluator FULL discovery baselines")
    state_data = {}
    source_hashes = {}
    for receiver in STEERING_RECEIVERS:
        baseline_path = artifact_root / "scores" / f"{receiver}_alpha_baselines.jsonl"
        sweep_path = artifact_root / "interventions" / f"{receiver}_alpha_sweep.jsonl"
        baseline_manifest_path = baseline_path.with_suffix(".manifest.json")
        sweep_manifest_path = sweep_path.with_suffix(".manifest.json")
        if not all(
            path.is_file()
            for path in (baseline_path, sweep_path, baseline_manifest_path, sweep_manifest_path)
        ):
            raise FileNotFoundError(f"Missing {receiver} alpha-selection artifacts")
        baseline_rows = read_jsonl(baseline_path)
        sweep_rows = read_jsonl(sweep_path)
        baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        sweep_manifest = json.loads(sweep_manifest_path.read_text(encoding="utf-8"))
        if (
            len(baseline_rows) != 200
            or len(sweep_rows) != 600
            or baseline_manifest.get("status") != "complete"
            or baseline_manifest.get("completed_rows") != 200
            or sweep_manifest.get("status") != "complete"
            or sweep_manifest.get("completed_rows") != 600
            or sweep_manifest.get("config", {}).get("alphas") != alphas
            or sweep_manifest.get("config", {}).get("direction_freeze_hash")
            != direction_manifest["direction_freeze_hash"]
            or sweep_manifest.get("config", {}).get("causal_layer_freeze_hash")
            != causal_layer["freeze_hash"]
            or baseline_manifest.get("config", {}).get("final_states_hash")
            != final_states["freeze_hash"]
            or baseline_manifest.get("config", {}).get("direction_freeze_hash")
            != direction_manifest["direction_freeze_hash"]
            or baseline_manifest.get("config", {}).get("causal_layer_freeze_hash")
            != causal_layer["freeze_hash"]
            or any(row.get("subset") not in {"discovery", "r_control"} for row in sweep_rows)
            or any(row.get("state") != receiver for row in sweep_rows)
        ):
            raise ValueError(f"Invalid or incomplete {receiver} alpha-selection artifacts")
        model_spec = _receiver_model_spec(receiver, final_states)
        for row in baseline_rows:
            if (
                row.get("state") != receiver
                or row.get("intervention") != "none"
                or row.get("model_id") != model_spec["repo_id"]
                or row.get("model_revision") != model_spec["revision"]
                or row.get("final_states_hash") != final_states["freeze_hash"]
                or row.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
                or row.get("direction_freeze_hash")
                != direction_manifest["direction_freeze_hash"]
            ):
                raise ValueError(f"Invalid {receiver} alpha baseline: {row.get('cell_id')}")
        for row in sweep_rows:
            if (
                row.get("state") != receiver
                or row.get("intervention") != "add_full_minus_idk_at_q_end"
                or row.get("layer") != causal_layer["selected_layer"]
                or row.get("token_position") != "Q_END"
                or row.get("alpha") not in alphas
                or row.get("model_id") != model_spec["repo_id"]
                or row.get("model_revision") != model_spec["revision"]
                or row.get("final_states_hash") != final_states["freeze_hash"]
                or row.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
                or row.get("direction_freeze_hash")
                != direction_manifest["direction_freeze_hash"]
            ):
                raise ValueError(f"Invalid {receiver} alpha row: {row.get('cell_id')}")
        baselines = {row["cell_id"]: row for row in baseline_rows}
        if len(baselines) != 200 or len({row["cell_id"] for row in sweep_rows}) != 600:
            raise ValueError(f"Duplicate {receiver} alpha-selection cells")
        state_data[receiver] = {
            "baselines": baselines,
            "sweep": sweep_rows,
        }
        source_hashes[receiver] = {
            "baseline_rows_hash": stable_hash(baseline_rows),
            "baseline_manifest_hash": stable_hash(baseline_manifest),
            "sweep_rows_hash": stable_hash(sweep_rows),
            "sweep_manifest_hash": stable_hash(sweep_manifest),
        }
    alpha_table = []
    for alpha in alphas:
        conditions = {}
        utility_changes = {}
        for receiver in STEERING_RECEIVERS:
            rows = state_data[receiver]["sweep"]
            discovery = [
                row for row in rows if row["subset"] == "discovery" and row["alpha"] == alpha
            ]
            r_control = [
                row for row in rows if row["subset"] == "r_control" and row["alpha"] == alpha
            ]
            if len(discovery) != 100 or len(r_control) != 100:
                raise ValueError(f"Incomplete alpha={alpha} condition for {receiver}")
            conditions[receiver] = _discovery_recovery(
                alpha_rows=discovery,
                baselines=state_data[receiver]["baselines"],
                full_baselines=full_baselines,
            )
            utility_changes[receiver] = float(np.mean([row["raw_change"] for row in r_control]))
        if (
            conditions["idk"]["mean_fractional_recovery"] is None
            or conditions["retain"]["mean_fractional_recovery"] is None
        ):
            raise ValueError(f"No stable discovery authors at alpha={alpha}")
        utility_penalty = max(abs(value) for value in utility_changes.values())
        objective = (
            conditions["idk"]["mean_fractional_recovery"]
            - conditions["retain"]["mean_fractional_recovery"]
            - beta * utility_penalty
        )
        alpha_table.append(
            {
                "alpha": alpha,
                "idk": conditions["idk"],
                "retain": conditions["retain"],
                "r_control_raw_change": utility_changes,
                "utility_penalty": utility_penalty,
                "beta": beta,
                "objective_j": objective,
            }
        )
    selected = max(alpha_table, key=lambda row: (row["objective_j"], -row["alpha"]))
    analysis_payload = {
        "schema_version": 1,
        "stage": "alpha_selection",
        "status": "COMPLETE",
        "scientific_gate": False,
        "selection_source": "IDK_and_RETAIN_discovery_plus_R_control_only",
        "gd02_read": False,
        "confirmation_read": False,
        "selection_rule": "argmax_J_tie_smaller_alpha",
        "alphas": alphas,
        "beta": beta,
        "selected_alpha": selected["alpha"],
        "selected_row": selected,
        "alpha_table": alpha_table,
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "source_hashes": source_hashes,
    }
    analysis = {**analysis_payload, "result_hash": stable_hash(analysis_payload)}
    if analysis_output.exists():
        existing = json.loads(analysis_output.read_text(encoding="utf-8"))
        if existing != analysis:
            raise ValueError(f"Refusing to overwrite changed alpha result: {analysis_output}")
        analysis = existing
    else:
        atomic_json(analysis_output, analysis)
    prompt_hashes = {
        receiver: stable_hash(
            sorted(
                row["prompt_hash"]
                for row in state_data[receiver]["baselines"].values()
            )
        )
        for receiver in STEERING_RECEIVERS
    }
    confirmation_payload = {
        "schema_version": 1,
        "stage": "confirmation_freeze",
        "status": "FROZEN",
        "confirmation_status_before_freeze": "sealed",
        "state_revisions": _state_revision_freeze(final_states),
        "split_hash": final_states["split"]["split_hash"],
        "discovery_authors": final_states["split"]["discovery_authors"],
        "confirmation_authors": final_states["split"]["confirmation_authors"],
        "r_control_authors": final_states["split"]["r_control_authors"],
        "prompt_hashes": prompt_hashes,
        "prompt_date": FROZEN_PROMPT_DATE,
        "evaluator_version": EVALUATOR_VERSION,
        "layer": causal_layer["selected_layer"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "direction": {
            "name": direction_manifest["name"],
            "sign": direction_manifest["sign"],
            "l2_norm": direction_manifest["l2_norm"],
            "tensor_sha256": direction_manifest["tensor_sha256"],
            "file_sha256": direction_manifest["file_sha256"],
            "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        },
        "selected_alpha": selected["alpha"],
        "alpha_selection_result_hash": analysis["result_hash"],
        "alpha_table": alpha_table,
        "beta": beta,
        "random_seeds": random_seeds,
        "generation_conditions": ["full", "unsteered_receiver", "learned_direction"],
        "random_direction_generation": False,
        "bootstrap_resamples": bootstrap_resamples,
        "final_states_hash": final_states["freeze_hash"],
        "gd02_used_for_layer_direction_or_alpha": False,
        "confirmation_read_before_freeze": False,
    }
    confirmation = {
        **confirmation_payload,
        "confirmation_freeze_hash": stable_hash(confirmation_payload),
    }
    if confirmation_output.exists():
        existing = json.loads(confirmation_output.read_text(encoding="utf-8"))
        if existing != confirmation:
            raise ValueError(
                f"Refusing to overwrite changed confirmation freeze: {confirmation_output}"
            )
        confirmation = existing
    else:
        atomic_json(confirmation_output, confirmation)
    return analysis, confirmation


def audit_chunk4(
    *,
    artifact_root: Path,
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
) -> dict[str, Any]:
    tensor_path = artifact_root / "directions" / "full_minus_idk.safetensors"
    direction_path = tensor_path.with_suffix(".manifest.json")
    analysis_path = artifact_root / "results" / "alpha_selection.json"
    confirmation_path = artifact_root / "freeze" / "confirmation.json"
    required = (tensor_path, direction_path, analysis_path, confirmation_path)
    if not all(path.is_file() for path in required):
        return {
            "stage": "D",
            "status": "INCOMPLETE",
            "reason": "direction, alpha-selection result, or confirmation freeze is missing",
        }
    _, direction = load_direction(
        tensor_path, direction_path, final_states, causal_layer
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis_payload = {key: value for key, value in analysis.items() if key != "result_hash"}
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    confirmation_payload = {
        key: value
        for key, value in confirmation.items()
        if key != "confirmation_freeze_hash"
    }
    if (
        analysis.get("status") != "COMPLETE"
        or stable_hash(analysis_payload) != analysis.get("result_hash")
        or analysis.get("direction_freeze_hash") != direction["direction_freeze_hash"]
        or analysis.get("gd02_read") is not False
        or analysis.get("confirmation_read") is not False
        or confirmation.get("status") != "FROZEN"
        or stable_hash(confirmation_payload)
        != confirmation.get("confirmation_freeze_hash")
        or confirmation.get("alpha_selection_result_hash") != analysis["result_hash"]
        or confirmation.get("selected_alpha") != analysis["selected_alpha"]
        or confirmation.get("gd02_used_for_layer_direction_or_alpha") is not False
        or confirmation.get("confirmation_read_before_freeze") is not False
    ):
        raise ValueError("Chunk 4 freeze chain is invalid or inconsistent")
    for receiver, hashes in analysis["source_hashes"].items():
        baseline_path = artifact_root / "scores" / f"{receiver}_alpha_baselines.jsonl"
        sweep_path = artifact_root / "interventions" / f"{receiver}_alpha_sweep.jsonl"
        baseline_manifest = baseline_path.with_suffix(".manifest.json")
        sweep_manifest = sweep_path.with_suffix(".manifest.json")
        if (
            stable_hash(read_jsonl(baseline_path)) != hashes["baseline_rows_hash"]
            or stable_hash(json.loads(baseline_manifest.read_text(encoding="utf-8")))
            != hashes["baseline_manifest_hash"]
            or stable_hash(read_jsonl(sweep_path)) != hashes["sweep_rows_hash"]
            or stable_hash(json.loads(sweep_manifest.read_text(encoding="utf-8")))
            != hashes["sweep_manifest_hash"]
        ):
            raise ValueError(f"{receiver} alpha artifacts changed after confirmation freeze")
    return {
        "stage": "D",
        "status": "COMPLETE",
        "selected_alpha": analysis["selected_alpha"],
        "selected_objective": analysis["selected_row"]["objective_j"],
        "selected_idk_rf": analysis["selected_row"]["idk"]["mean_fractional_recovery"],
        "selected_retain_rf": analysis["selected_row"]["retain"]["mean_fractional_recovery"],
        "selected_utility_penalty": analysis["selected_row"]["utility_penalty"],
        "direction_norm": direction["l2_norm"],
        "direction_freeze_hash": direction["direction_freeze_hash"],
        "confirmation_freeze_hash": confirmation["confirmation_freeze_hash"],
    }
