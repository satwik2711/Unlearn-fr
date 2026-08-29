"""Frozen-layer transfer patching for GD02 and RETAIN."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import atomic_json, stable_hash
from .idk_localization import clustered_interval, resolve_frozen_path
from .metrics import EVALUATOR_VERSION, FROZEN_PROMPT_DATE, encode_answer, score_encoded_batch
from .patching import ActivationStore
from .records import append_jsonl, initialize_manifest, read_jsonl, read_unique


RECEIVERS = {
    "gd02": {"model_key": "gd_02", "record_state": "gd"},
    "retain": {"model_key": "retain", "record_state": "retain"},
}


def load_causal_layer(path: Path, final_states: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing causal-layer freeze: {path}")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in freeze.items() if key != "freeze_hash"}
    if (
        freeze.get("status") != "FROZEN"
        or stable_hash(payload) != freeze.get("freeze_hash")
        or freeze.get("final_states_hash") != final_states["freeze_hash"]
        or freeze.get("selection_source") != "IDK discovery only"
        or freeze.get("gd02_read_during_selection") is not False
    ):
        raise ValueError("Causal-layer freeze is invalid or not IDK-only")
    return freeze


def load_transfer_inputs(
    *,
    artifact_root: Path,
    receiver: str,
    rows: list[dict],
    final_states: dict[str, Any],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    if receiver not in RECEIVERS:
        raise ValueError(f"Unknown transfer receiver: {receiver}")
    expected_ids = {row["example_id"] for row in rows}
    full_donors = read_unique(
        resolve_frozen_path(final_states["states"]["full"]["discovery_scores"])
    )
    archived = read_unique(
        resolve_frozen_path(final_states["states"][receiver]["discovery_scores"])
    )
    runtime_rows = read_jsonl(
        artifact_root / "scores" / "idk_runtime_baselines.jsonl"
    )
    runtime_full = {
        row["example_id"]: row for row in runtime_rows if row.get("state") == "full"
    }
    if set(full_donors) != expected_ids or set(archived) != expected_ids:
        raise ValueError(f"{receiver} inputs differ from the frozen discovery subset")
    if set(runtime_full) != expected_ids:
        raise ValueError("Chunk 2 current-runtime FULL baselines are incomplete")
    runtime_manifest_path = (
        artifact_root / "scores" / "idk_runtime_baselines.manifest.json"
    )
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    if (
        runtime_manifest.get("status") != "complete"
        or runtime_manifest.get("completed_rows") != 200
        or runtime_manifest.get("config", {}).get("final_states_hash")
        != final_states["freeze_hash"]
    ):
        raise ValueError("Chunk 2 runtime-baseline provenance is invalid")
    expected_state = RECEIVERS[receiver]["record_state"]
    for example_id in expected_ids:
        donor = full_donors[example_id]
        baseline = archived[example_id]
        runtime = runtime_full[example_id]
        if (
            baseline.get("state") != expected_state
            or donor["prompt_hash"] != baseline["prompt_hash"]
            or donor["prompt_hash"] != runtime["prompt_hash"]
            or baseline["author_id"] != donor["author_id"]
            or runtime.get("final_states_hash") != final_states["freeze_hash"]
        ):
            raise ValueError(f"Frozen transfer provenance drift at {receiver}/{example_id}")
    return full_donors, archived, runtime_full


def run_receiver_baselines(
    *,
    model,
    tokenizer,
    receiver: str,
    rows: list[dict],
    archived_rows: dict[str, dict],
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> dict[str, dict]:
    state_spec = final_states["states"][receiver]
    model_spec = state_spec["model"]
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "receiver": receiver,
        "model": model_spec,
        "candidate_id": state_spec.get("candidate_id"),
        "intervention": "none",
        "subset": "discovery_runtime_baseline",
        "example_ids": [row["example_id"] for row in rows],
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in rows]
        ),
        "split_hash": final_states["split"]["split_hash"],
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "batch_size": batch_size,
        "generation": False,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["example_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate {receiver} runtime baseline rows")
    expected = len(rows)
    pending = [row for row in rows if row["example_id"] not in completed]
    print(f"{receiver}/runtime_baseline: {len(completed)}/{expected} already complete")
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        encoded = [
            encode_answer(tokenizer, row["question"], row["answer"]) for row in batch
        ]
        scores, _ = score_encoded_batch(model, tokenizer, encoded)
        records = []
        for prepared, score in zip(batch, scores, strict=True):
            example_id = prepared["example_id"]
            archived = archived_rows[example_id]
            if score["prompt_hash"] != archived["prompt_hash"]:
                raise ValueError(f"Prompt drift at {receiver}/{example_id}")
            records.append(
                {
                    "schema_version": 1,
                    "run_id": config_hash,
                    "state": receiver,
                    "subset": "discovery_runtime_baseline",
                    "author_id": prepared["author_id"],
                    "example_id": example_id,
                    "intervention": "none",
                    "model_id": model_spec["repo_id"],
                    "model_revision": model_spec["revision"],
                    "candidate_id": state_spec.get("candidate_id"),
                    "mean_target_logprob": score["mean_target_logprob"],
                    "token_count": score["token_count"],
                    "prompt_hash": score["prompt_hash"],
                    "split_hash": final_states["split"]["split_hash"],
                    "final_states_hash": final_states["freeze_hash"],
                    "causal_layer_freeze_hash": causal_layer["freeze_hash"],
                }
            )
        append_jsonl(output_path, records)
        completed.update(record["example_id"] for record in records)
        manifest.update(status="running", completed_rows=len(completed), expected_rows=expected)
        atomic_json(manifest_path, manifest)
        print(f"{receiver}/runtime_baseline: {len(completed)}/{expected}")
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} {receiver} runtime baselines")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)
    result = read_unique(output_path)
    for example_id, record in result.items():
        if (
            record.get("state") != receiver
            or record.get("final_states_hash") != final_states["freeze_hash"]
            or record.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
            or record.get("prompt_hash") != archived_rows[example_id]["prompt_hash"]
            or record.get("model_id") != model_spec["repo_id"]
            or record.get("model_revision") != model_spec["revision"]
        ):
            raise ValueError(f"Invalid resumed baseline at {receiver}/{example_id}")
    return result


def run_transfer_sweep(
    *,
    model,
    tokenizer,
    receiver: str,
    rows: list[dict],
    receiver_baselines: dict[str, dict],
    runtime_full: dict[str, dict],
    full_donors: dict[str, dict],
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    layers: list[int],
    batch_size: int,
    output_path: Path,
) -> None:
    architecture = final_states["architecture"]
    donor_store = ActivationStore(
        full_donors,
        architecture["decoder_layers"],
        architecture["hidden_size"],
        resolve_frozen_path(final_states["states"]["full"]["activations"]["directory"]),
    )
    example_ids = [row["example_id"] for row in rows]
    donor_store.validate_all(example_ids)
    state_spec = final_states["states"][receiver]
    model_spec = state_spec["model"]
    baseline_manifest = json.loads(
        (output_path.parents[1] / "scores" / f"{receiver}_runtime_baseline.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "receiver": receiver,
        "model": model_spec,
        "candidate_id": state_spec.get("candidate_id"),
        "donor_state": "full",
        "intervention": "matched_full_q_end_patch",
        "token_position": "Q_END",
        "layers": layers,
        "example_ids": example_ids,
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in rows]
        ),
        "split_hash": final_states["split"]["split_hash"],
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "receiver_runtime_baseline_manifest_hash": stable_hash(baseline_manifest),
        "full_runtime_baseline_source": "chunk2_idk_runtime_baselines",
        "batch_size": batch_size,
        "generation": False,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate {receiver} patch cells")
    expected = len(rows) * len(layers)
    print(f"{receiver}/layer_sweep: {len(completed)}/{expected} already complete")
    for layer in layers:
        pending = [
            row
            for row in rows
            if f"{row['example_id']}:layer-{layer:02d}" not in completed
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            encoded = [
                encode_answer(tokenizer, row["question"], row["answer"])
                for row in batch
            ]
            donors = torch.stack(
                [donor_store.vector(row["example_id"], layer) for row in batch]
            )
            scores, _ = score_encoded_batch(
                model,
                tokenizer,
                encoded,
                patch_layer=layer,
                patch_q_end_values=donors,
            )
            records = []
            for prepared, score in zip(batch, scores, strict=True):
                example_id = prepared["example_id"]
                receiver_row = receiver_baselines[example_id]
                full_row = runtime_full[example_id]
                if (
                    score["prompt_hash"] != receiver_row["prompt_hash"]
                    or score["prompt_hash"] != full_row["prompt_hash"]
                ):
                    raise ValueError(f"Prompt drift during {receiver}/{example_id}")
                raw = score["mean_target_logprob"] - receiver_row["mean_target_logprob"]
                headroom = full_row["mean_target_logprob"] - receiver_row["mean_target_logprob"]
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": f"{example_id}:layer-{layer:02d}",
                        "state": receiver,
                        "subset": "discovery",
                        "author_id": prepared["author_id"],
                        "example_id": example_id,
                        "intervention": "matched_full_q_end_patch",
                        "layer": layer,
                        "token_position": "Q_END",
                        "donor_state": "full",
                        "donor_example_id": example_id,
                        "donor_activation_file": Path(full_donors[example_id]["activation_file"]).name,
                        "donor_activation_row": full_donors[example_id]["activation_row"],
                        "model_id": model_spec["repo_id"],
                        "model_revision": model_spec["revision"],
                        "candidate_id": state_spec.get("candidate_id"),
                        "mean_target_logprob": score["mean_target_logprob"],
                        "baseline_mean_target_logprob": receiver_row["mean_target_logprob"],
                        "full_mean_target_logprob": full_row["mean_target_logprob"],
                        "raw_recovery": raw,
                        "headroom": headroom,
                        "question_fractional_recovery_diagnostic": (
                            raw / headroom if abs(headroom) > 1e-12 else None
                        ),
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "split_hash": final_states["split"]["split_hash"],
                        "final_states_hash": final_states["freeze_hash"],
                        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
                    }
                )
            append_jsonl(output_path, records)
            completed.update(record["cell_id"] for record in records)
            manifest.update(status="running", completed_rows=len(completed), expected_rows=expected)
            atomic_json(manifest_path, manifest)
            print(f"{receiver}/layer_sweep: {len(completed)}/{expected}")
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} {receiver} patch cells")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)


def _layer_metrics(rows: list[dict], layers: list[int]) -> list[dict]:
    metrics = []
    for layer in layers:
        by_author: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["layer"] == layer:
                by_author[row["author_id"]].append(row)
        if len(by_author) != 5 or any(len(values) != 20 for values in by_author.values()):
            raise ValueError(f"Layer {layer} lacks five complete discovery authors")
        author_raw = {
            author: float(np.mean([row["raw_recovery"] for row in values]))
            for author, values in sorted(by_author.items())
        }
        author_headroom = {
            author: float(np.mean([row["headroom"] for row in values]))
            for author, values in sorted(by_author.items())
        }
        unstable = {author: value for author, value in author_headroom.items() if value <= 0.02}
        author_rf = {
            author: author_raw[author] / author_headroom[author]
            for author in author_raw
            if author not in unstable
        }
        metrics.append(
            {
                "layer": layer,
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
        )
    return metrics


def finalize_patch_transfer(
    *,
    artifact_root: Path,
    final_states: dict[str, Any],
    causal_layer: dict[str, Any],
    layers: list[int],
    bootstrap_resamples: int,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    receiver_metrics: dict[str, list[dict]] = {}
    source_hashes = {}
    for receiver in RECEIVERS:
        path = artifact_root / "interventions" / f"{receiver}_layer_sweep.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed {receiver} sweep")
        rows = read_jsonl(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            len(rows) != 1600
            or manifest.get("status") != "complete"
            or manifest.get("completed_rows") != 1600
            or manifest.get("config", {}).get("final_states_hash")
            != final_states["freeze_hash"]
            or manifest.get("config", {}).get("causal_layer_freeze_hash")
            != causal_layer["freeze_hash"]
            or any(row.get("subset") != "discovery" for row in rows)
        ):
            raise ValueError(f"Invalid or incomplete {receiver} transfer sweep")
        model_spec = final_states["states"][receiver]["model"]
        for row in rows:
            if (
                row.get("state") != receiver
                or row.get("intervention") != "matched_full_q_end_patch"
                or row.get("token_position") != "Q_END"
                or row.get("donor_state") != "full"
                or row.get("layer") not in layers
                or row.get("final_states_hash") != final_states["freeze_hash"]
                or row.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
                or row.get("model_id") != model_spec["repo_id"]
                or row.get("model_revision") != model_spec["revision"]
            ):
                raise ValueError(f"Invalid {receiver} transfer row: {row.get('cell_id')}")
        if len({row["cell_id"] for row in rows}) != 1600:
            raise ValueError(f"Duplicate {receiver} transfer cells")
        receiver_metrics[receiver] = _layer_metrics(rows, layers)
        source_hashes[receiver] = {
            "rows_hash": stable_hash(rows),
            "manifest_hash": stable_hash(manifest),
        }
    selected_layer = causal_layer["selected_layer"]
    gd = next(
        metric for metric in receiver_metrics["gd02"] if metric["layer"] == selected_layer
    )
    retain = next(
        metric for metric in receiver_metrics["retain"] if metric["layer"] == selected_layer
    )
    common_authors = sorted(
        set(gd["author_fractional_recovery"])
        & set(retain["author_fractional_recovery"])
    )
    differentials = {
        author: gd["author_fractional_recovery"][author]
        - retain["author_fractional_recovery"][author]
        for author in common_authors
    }
    if not differentials:
        raise ValueError("No stable authors remain for the frozen-layer comparison")
    c_patch = float(np.mean(list(differentials.values())))
    payload = {
        "schema_version": 1,
        "stage": "patch_transfer",
        "status": "COMPLETE",
        "scientific_gate": False,
        "selected_layer": selected_layer,
        "layer_selection_source": "frozen_IDK_discovery_only",
        "causal_layer_freeze_hash": causal_layer["freeze_hash"],
        "final_states_hash": final_states["freeze_hash"],
        "primary_estimand": "RF_gd02_minus_RF_retain_at_frozen_layer",
        "c_patch": c_patch,
        "gd02_at_frozen_layer": gd,
        "retain_at_frozen_layer": retain,
        "author_differentials": differentials,
        "stable_authors": common_authors,
        "stable_author_count": len(common_authors),
        "author_clustered_ci_95": clustered_interval(
            list(differentials.values()), bootstrap_resamples, seed
        ),
        "descriptive_layer_curves": receiver_metrics,
        "source_hashes": source_hashes,
        "confirmation_read": False,
        "interpretation": (
            "positive_differential_transfer" if c_patch > 0 else
            "contrary_differential_transfer" if c_patch < 0 else "null_differential_transfer"
        ),
    }
    result = {**payload, "result_hash": stable_hash(payload)}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(f"Refusing to overwrite changed transfer result: {output_path}")
        return existing
    atomic_json(output_path, result)
    return result


def audit_chunk3(
    *, artifact_root: Path, final_states: dict[str, Any], causal_layer: dict[str, Any]
) -> dict[str, Any]:
    path = artifact_root / "results" / "patch_transfer.json"
    if not path.is_file():
        return {"stage": "C", "status": "INCOMPLETE", "reason": "patch_transfer.json is missing"}
    result = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in result.items() if key != "result_hash"}
    if (
        result.get("status") != "COMPLETE"
        or stable_hash(payload) != result.get("result_hash")
        or result.get("final_states_hash") != final_states["freeze_hash"]
        or result.get("causal_layer_freeze_hash") != causal_layer["freeze_hash"]
        or result.get("selected_layer") != causal_layer["selected_layer"]
        or result.get("confirmation_read") is not False
    ):
        raise ValueError("Chunk 3 result is inconsistent with its frozen inputs")
    for receiver, hashes in result["source_hashes"].items():
        sweep = artifact_root / "interventions" / f"{receiver}_layer_sweep.jsonl"
        manifest = sweep.with_suffix(".manifest.json")
        if (
            stable_hash(read_jsonl(sweep)) != hashes["rows_hash"]
            or stable_hash(json.loads(manifest.read_text(encoding="utf-8")))
            != hashes["manifest_hash"]
        ):
            raise ValueError(f"{receiver} sweep changed after transfer finalization")
    return result
