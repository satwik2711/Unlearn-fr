"""IDK-only exact patch sweep, layer freeze, and self-patch audit."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PROJECT_ROOT, atomic_json, stable_hash
from .metrics import EVALUATOR_VERSION, FROZEN_PROMPT_DATE, encode_answer, score_encoded_batch
from .patching import ActivationStore
from .records import (
    append_jsonl,
    atomic_jsonl,
    initialize_manifest,
    read_jsonl,
    read_unique,
)


def resolve_frozen_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Frozen path escapes project root: {relative_path}") from error
    return path


def load_final_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing final state freeze: {path}")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = freeze.get("freeze_hash")
    payload = {key: value for key, value in freeze.items() if key != "freeze_hash"}
    if freeze.get("status") != "COMPLETE" or stable_hash(payload) != claimed_hash:
        raise ValueError("Final state freeze is incomplete or has changed")
    if freeze.get("confirmation", {}).get("status") != "sealed":
        raise ValueError("Confirmation is not sealed")
    return freeze


def load_localization_records(
    freeze: dict[str, Any],
    expected_ids: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    full = read_unique(resolve_frozen_path(freeze["states"]["full"]["discovery_scores"]))
    idk = read_unique(resolve_frozen_path(freeze["states"]["idk"]["discovery_scores"]))
    if set(full) != expected_ids or set(idk) != expected_ids:
        raise ValueError("IDK localization rows differ from the frozen discovery subset")
    for example_id in expected_ids:
        left, right = full[example_id], idk[example_id]
        if (
            left["prompt_hash"] != right["prompt_hash"]
            or left["token_count"] != right["token_count"]
            or left["author_id"] != right["author_id"]
            or left["split_hash"] != freeze["split"]["split_hash"]
            or right["split_hash"] != freeze["split"]["split_hash"]
        ):
            raise ValueError(f"FULL/IDK localization drift at {example_id}")
    return full, idk


def run_idk_layer_sweep(
    *,
    model,
    tokenizer,
    rows: list[dict],
    full_rows: dict[str, dict],
    idk_rows: dict[str, dict],
    freeze: dict[str, Any],
    layers: list[int],
    batch_size: int,
    output_path: Path,
) -> None:
    architecture = freeze["architecture"]
    donor_store = ActivationStore(
        full_rows,
        architecture["decoder_layers"],
        architecture["hidden_size"],
        resolve_frozen_path(freeze["states"]["full"]["activations"]["directory"]),
    )
    example_ids = [row["example_id"] for row in rows]
    donor_store.validate_all(example_ids)
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "final_states_hash": freeze["freeze_hash"],
        "receiver_state": "idk",
        "base_model": freeze["states"]["idk"]["base_model"],
        "adapter_id": freeze["states"]["idk"]["adapter_id"],
        "adapter_hash": freeze["states"]["idk"]["adapter_hash"],
        "donor_state": "full",
        "intervention": "matched_full_q_end_patch",
        "token_position": "Q_END",
        "layers": layers,
        "example_ids": example_ids,
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in rows]
        ),
        "split_hash": freeze["split"]["split_hash"],
        "batch_size": batch_size,
        "generation": False,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate IDK patch cells in {output_path}")
    expected = len(rows) * len(layers)
    print(f"idk/layer_sweep: {len(completed)}/{expected} already complete")

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
                baseline = idk_rows[example_id]
                full = full_rows[example_id]
                if (
                    score["prompt_hash"] != baseline["prompt_hash"]
                    or score["prompt_hash"] != full["prompt_hash"]
                ):
                    raise ValueError(f"Prompt drift during IDK patching: {example_id}")
                headroom = full["mean_target_logprob"] - baseline["mean_target_logprob"]
                effect = score["mean_target_logprob"] - baseline["mean_target_logprob"]
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": f"{example_id}:layer-{layer:02d}",
                        "state": "idk",
                        "model_id": freeze["states"]["idk"]["base_model"]["repo_id"],
                        "model_revision": freeze["states"]["idk"]["base_model"]["revision"],
                        "adapter_id": freeze["states"]["idk"]["adapter_id"],
                        "adapter_hash": freeze["states"]["idk"]["adapter_hash"],
                        "subset": "discovery",
                        "author_id": prepared["author_id"],
                        "example_id": example_id,
                        "intervention": "matched_full_q_end_patch",
                        "layer": layer,
                        "token_position": "Q_END",
                        "donor_state": "full",
                        "donor_example_id": example_id,
                        "donor_activation_file": Path(full["activation_file"]).name,
                        "donor_activation_row": full["activation_row"],
                        "mean_target_logprob": score["mean_target_logprob"],
                        "baseline_mean_target_logprob": baseline["mean_target_logprob"],
                        "full_mean_target_logprob": full["mean_target_logprob"],
                        "raw_recovery": effect,
                        "headroom": headroom,
                        "question_fractional_recovery_diagnostic": effect / headroom,
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "split_hash": freeze["split"]["split_hash"],
                        "final_states_hash": freeze["freeze_hash"],
                    }
                )
            append_jsonl(output_path, records)
            completed.update(record["cell_id"] for record in records)
            manifest.update(
                status="running",
                completed_rows=len(completed),
                expected_rows=expected,
            )
            atomic_json(manifest_path, manifest)
            print(f"idk/layer_sweep: {len(completed)}/{expected}")

    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} IDK patch cells, found {len(completed)}")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)


def run_runtime_baselines(
    *,
    model,
    tokenizer,
    rows: list[dict],
    final_states: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Score FULL-off and IDK-on once in the intervention runtime."""

    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "final_states_hash": final_states["freeze_hash"],
        "states": ["full", "idk"],
        "adapter_id": final_states["states"]["idk"]["adapter_id"],
        "intervention": "none",
        "subset": "discovery_runtime_baseline",
        "example_ids": [row["example_id"] for row in rows],
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in rows]
        ),
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
        raise ValueError("Duplicate current-runtime baseline cells")
    expected = 2 * len(rows)
    pending = [
        row
        for row in rows
        if f"full:{row['example_id']}" not in completed
        or f"idk:{row['example_id']}" not in completed
    ]
    print(f"full+idk/runtime_baselines: {len(completed)}/{expected} already complete")
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        encoded = [
            encode_answer(tokenizer, row["question"], row["answer"]) for row in batch
        ]
        with model.disable_adapter():
            full_scores, _ = score_encoded_batch(model, tokenizer, encoded)
        idk_scores, _ = score_encoded_batch(model, tokenizer, encoded)
        records = []
        for prepared, full_score, idk_score in zip(
            batch, full_scores, idk_scores, strict=True
        ):
            example_id = prepared["example_id"]
            for state, score in (("full", full_score), ("idk", idk_score)):
                cell_id = f"{state}:{example_id}"
                if cell_id in completed:
                    continue
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": cell_id,
                        "state": state,
                        "subset": "discovery_runtime_baseline",
                        "author_id": prepared["author_id"],
                        "example_id": example_id,
                        "intervention": "none",
                        "mean_target_logprob": score["mean_target_logprob"],
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "split_hash": final_states["split"]["split_hash"],
                        "final_states_hash": final_states["freeze_hash"],
                    }
                )
        append_jsonl(output_path, records)
        completed.update(record["cell_id"] for record in records)
        manifest.update(
            status="running",
            completed_rows=len(completed),
            expected_rows=expected,
        )
        atomic_json(manifest_path, manifest)
        print(f"full+idk/runtime_baselines: {len(completed)}/{expected}")
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} runtime baseline cells")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)
    records = read_jsonl(output_path)
    full = {row["example_id"]: row for row in records if row["state"] == "full"}
    idk = {row["example_id"]: row for row in records if row["state"] == "idk"}
    if len(full) != len(rows) or len(idk) != len(rows):
        raise ValueError("Current-runtime baseline states are incomplete")
    return full, idk


def rebase_layer_sweep(
    *,
    raw_sweep_path: Path,
    runtime_baseline_path: Path,
    runtime_full: dict[str, dict],
    runtime_idk: dict[str, dict],
    final_states: dict[str, Any],
    output_path: Path,
) -> None:
    """Recalculate effects from same-runtime baselines without rerunning patches."""

    raw_manifest = json.loads(
        raw_sweep_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    runtime_manifest = json.loads(
        runtime_baseline_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    raw_rows = read_jsonl(raw_sweep_path)
    if (
        raw_manifest.get("status") != "complete"
        or len(raw_rows) != 1600
        or runtime_manifest.get("status") != "complete"
        or runtime_manifest.get("completed_rows") != 200
    ):
        raise ValueError("Raw IDK sweep or runtime baselines are incomplete")
    rebased = []
    for row in raw_rows:
        example_id = row["example_id"]
        full = runtime_full[example_id]
        idk = runtime_idk[example_id]
        if (
            row["prompt_hash"] != full["prompt_hash"]
            or row["prompt_hash"] != idk["prompt_hash"]
        ):
            raise ValueError(f"Runtime baseline prompt drift at {example_id}")
        headroom = full["mean_target_logprob"] - idk["mean_target_logprob"]
        effect = row["mean_target_logprob"] - idk["mean_target_logprob"]
        rebased.append(
            {
                **row,
                "archived_baseline_mean_target_logprob": row[
                    "baseline_mean_target_logprob"
                ],
                "archived_full_mean_target_logprob": row[
                    "full_mean_target_logprob"
                ],
                "baseline_mean_target_logprob": idk["mean_target_logprob"],
                "full_mean_target_logprob": full["mean_target_logprob"],
                "raw_recovery": effect,
                "headroom": headroom,
                "question_fractional_recovery_diagnostic": (
                    effect / headroom if abs(headroom) > 1e-12 else None
                ),
                "baseline_source": "same_runtime",
                "runtime_baseline_run_id": idk["run_id"],
            }
        )
    job_config = {
        "schema_version": 1,
        "status": "derived_without_model_run",
        "final_states_hash": final_states["freeze_hash"],
        "source_sweep_manifest_hash": stable_hash(raw_manifest),
        "source_sweep_rows_hash": stable_hash(raw_rows),
        "runtime_baseline_manifest_hash": stable_hash(runtime_manifest),
        "runtime_baseline_rows_hash": stable_hash(
            read_jsonl(runtime_baseline_path)
        ),
        "effect_definition": "patched_minus_same_runtime_idk",
        "headroom_definition": "same_runtime_full_minus_same_runtime_idk",
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_rows": len(rebased),
        "expected_rows": 1600,
        "config": job_config,
        "config_hash": stable_hash(job_config),
    }
    if output_path.exists():
        existing = read_jsonl(output_path)
        existing_manifest = json.loads(
            output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        if existing != rebased or existing_manifest != manifest:
            raise ValueError("Refusing to overwrite changed rebased IDK sweep")
        return
    atomic_jsonl(output_path, rebased)
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)


def clustered_interval(values: list[float], resamples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def freeze_causal_layer(
    *,
    sweep_path: Path,
    final_states: dict[str, Any],
    output_path: Path,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    manifest_path = sweep_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing IDK sweep manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(sweep_path)
    layers = list(range(final_states["architecture"]["decoder_layers"]))
    if (
        manifest.get("status") != "complete"
        or manifest.get("completed_rows") != 1600
        or manifest.get("expected_rows") != 1600
        or len(rows) != 1600
        or manifest.get("config", {}).get("final_states_hash")
        != final_states["freeze_hash"]
    ):
        raise ValueError("IDK layer sweep is incomplete or bound to another freeze")
    cells = {(row["example_id"], row["layer"]): row for row in rows}
    if len(cells) != 1600:
        raise ValueError("IDK layer sweep has duplicate cells")
    expected_examples = {
        row["example_id"] for row in rows if row.get("layer") == layers[0]
    }
    if len(expected_examples) != 100:
        raise ValueError("IDK layer sweep does not contain 100 examples")

    layer_metrics = []
    for layer in layers:
        layer_rows = [cells[(example_id, layer)] for example_id in sorted(expected_examples)]
        by_author: dict[str, list[dict]] = defaultdict(list)
        for row in layer_rows:
            if (
                row.get("state") != "idk"
                or row.get("intervention") != "matched_full_q_end_patch"
                or row.get("final_states_hash") != final_states["freeze_hash"]
            ):
                raise ValueError(f"Invalid IDK layer record: {row.get('cell_id')}")
            by_author[row["author_id"]].append(row)
        if len(by_author) != 5 or any(len(author_rows) != 20 for author_rows in by_author.values()):
            raise ValueError(f"Layer {layer} does not contain five complete authors")
        author_raw = {
            author: float(np.mean([row["raw_recovery"] for row in author_rows]))
            for author, author_rows in sorted(by_author.items())
        }
        author_headroom = {
            author: float(np.mean([row["headroom"] for row in author_rows]))
            for author, author_rows in sorted(by_author.items())
        }
        unstable_authors = {
            author: value for author, value in author_headroom.items() if value <= 0.02
        }
        if unstable_authors:
            raise ValueError(
                f"Layer {layer} has unstable author-level IDK headroom: "
                f"{unstable_authors}"
            )
        author_rf = {
            author: author_raw[author] / author_headroom[author]
            for author in author_raw
        }
        layer_metrics.append(
            {
                "layer": layer,
                "mean_raw_recovery": float(np.mean(list(author_raw.values()))),
                "question_mean_raw_recovery": float(
                    np.mean([row["raw_recovery"] for row in layer_rows])
                ),
                "mean_fractional_recovery": float(np.mean(list(author_rf.values()))),
                "author_raw_recovery": author_raw,
                "author_headroom": author_headroom,
                "author_fractional_recovery": author_rf,
                "positive_authors": sum(value > 0 for value in author_rf.values()),
                "raw_author_clustered_ci_95": clustered_interval(
                    list(author_raw.values()), bootstrap_resamples, seed
                ),
                "fractional_author_clustered_ci_95": clustered_interval(
                    list(author_rf.values()), bootstrap_resamples, seed
                ),
            }
        )
    selected = max(
        layer_metrics,
        key=lambda metric: (metric["mean_fractional_recovery"], -metric["layer"]),
    )
    payload = {
        "schema_version": 1,
        "stage": "idk_causal_layer_freeze",
        "status": "FROZEN",
        "scientific_gate": False,
        "selection_source": "IDK discovery only",
        "selection_rule": "argmax_author_mean_fractional_recovery_tie_lower_layer",
        "selected_layer": selected["layer"],
        "selected_metrics": selected,
        "layer_metrics": layer_metrics,
        "final_states_hash": final_states["freeze_hash"],
        "sweep_manifest_hash": stable_hash(manifest),
        "sweep_rows_hash": stable_hash(rows),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "gd02_read_during_selection": False,
        "confirmation_read_during_selection": False,
    }
    result = {**payload, "freeze_hash": stable_hash(payload)}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(f"Refusing to overwrite causal-layer freeze: {output_path}")
        return existing
    atomic_json(output_path, result)
    return result


def run_self_patch_audit(
    *,
    model,
    tokenizer,
    rows: list[dict],
    idk_rows: dict[str, dict],
    runtime_idk: dict[str, dict],
    final_states: dict[str, Any],
    layer_freeze: dict[str, Any],
    batch_size: int,
    tolerance: float,
    output_path: Path,
) -> None:
    first_by_author: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (item["author_id"], item["example_id"])):
        first_by_author.setdefault(row["author_id"], row)
    audit_rows = [first_by_author[author] for author in sorted(first_by_author)[:4]]
    architecture = final_states["architecture"]
    own_store = ActivationStore(
        idk_rows,
        architecture["decoder_layers"],
        architecture["hidden_size"],
        resolve_frozen_path(final_states["states"]["idk"]["activations"]["directory"]),
    )
    selected_layer = layer_freeze["selected_layer"]
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "final_states_hash": final_states["freeze_hash"],
        "causal_layer_freeze_hash": layer_freeze["freeze_hash"],
        "selected_layer": selected_layer,
        "audit_example_ids": [row["example_id"] for row in audit_rows],
        "intervention": "idk_self_q_end_patch",
        "tolerance": tolerance,
        "batch_size": batch_size,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["example_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate IDK self-patch audit rows")
    pending = [row for row in audit_rows if row["example_id"] not in completed]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        encoded = [
            encode_answer(tokenizer, row["question"], row["answer"]) for row in batch
        ]
        current_scores, current_activations = score_encoded_batch(
            model, tokenizer, encoded, capture_q_end=True
        )
        if current_activations is None:
            raise RuntimeError("Self-patch audit did not capture current IDK activations")
        archived = torch.stack(
            [own_store.vector(row["example_id"], selected_layer) for row in batch]
        )
        live_donors = current_activations[:, selected_layer, :].clone()
        patched_scores, _ = score_encoded_batch(
            model,
            tokenizer,
            encoded,
            patch_layer=selected_layer,
            patch_q_end_values=live_donors,
        )
        records = []
        for index, (prepared, current, patched) in enumerate(
            zip(batch, current_scores, patched_scores, strict=True)
        ):
            example_id = prepared["example_id"]
            baseline = idk_rows[example_id]
            runtime_baseline = runtime_idk[example_id]
            records.append(
                {
                    "schema_version": 1,
                    "run_id": config_hash,
                    "state": "idk",
                    "subset": "discovery_audit",
                    "author_id": prepared["author_id"],
                    "example_id": example_id,
                    "intervention": "idk_self_q_end_patch",
                    "layer": selected_layer,
                    "token_position": "Q_END",
                    "mean_target_logprob": patched["mean_target_logprob"],
                    "current_unpatched_logprob": current["mean_target_logprob"],
                    "stored_unpatched_logprob": baseline["mean_target_logprob"],
                    "stored_baseline_delta": current["mean_target_logprob"]
                    - baseline["mean_target_logprob"],
                    "runtime_baseline_delta": current["mean_target_logprob"]
                    - runtime_baseline["mean_target_logprob"],
                    "patch_effect": patched["mean_target_logprob"]
                    - current["mean_target_logprob"],
                    "activation_max_abs_delta": float(
                        (
                            current_activations[index, selected_layer].float()
                            - archived[index].float()
                        )
                        .abs()
                        .max()
                    ),
                    "prompt_hash": patched["prompt_hash"],
                    "final_states_hash": final_states["freeze_hash"],
                    "causal_layer_freeze_hash": layer_freeze["freeze_hash"],
                }
            )
        append_jsonl(output_path, records)
        completed.update(row["example_id"] for row in records)
        manifest.update(status="running", completed_rows=len(completed), expected_rows=4)
        atomic_json(manifest_path, manifest)
        print(f"idk/self_patch_audit: {len(completed)}/4")
    all_rows = read_jsonl(output_path)
    max_effect = max(abs(row["patch_effect"]) for row in all_rows)
    max_baseline_delta = max(abs(row["stored_baseline_delta"]) for row in all_rows)
    max_runtime_baseline_delta = max(
        abs(row["runtime_baseline_delta"]) for row in all_rows
    )
    max_activation_delta = max(abs(row["activation_max_abs_delta"]) for row in all_rows)
    manifest.update(
        status="complete",
        completed_rows=4,
        expected_rows=4,
        max_abs_patch_effect=max_effect,
        max_abs_stored_baseline_delta=max_baseline_delta,
        max_abs_runtime_baseline_delta=max_runtime_baseline_delta,
        max_abs_activation_delta=max_activation_delta,
        live_self_patch_within_tolerance=max_effect <= tolerance,
        archived_cross_run_drift_is_diagnostic=True,
    )
    atomic_json(manifest_path, manifest)


def audit_chunk2(
    *,
    artifact_root: Path,
    final_states: dict[str, Any],
) -> dict[str, Any]:
    sweep_path = artifact_root / "interventions" / "idk_layer_sweep_rebased.jsonl"
    layer_path = artifact_root / "freeze" / "causal_layer.json"
    self_path = artifact_root / "interventions" / "idk_self_patch_audit.jsonl"
    if not sweep_path.exists() or not layer_path.exists() or not self_path.exists():
        return {
            "stage": "B",
            "status": "INCOMPLETE",
            "reason": "IDK layer sweep, causal-layer freeze, or self-patch audit is missing",
        }
    layer_freeze = json.loads(layer_path.read_text(encoding="utf-8"))
    layer_payload = {
        key: value for key, value in layer_freeze.items() if key != "freeze_hash"
    }
    self_manifest = json.loads(
        self_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    sweep_rows = read_jsonl(sweep_path)
    sweep_manifest = json.loads(
        sweep_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    if (
        layer_freeze.get("status") != "FROZEN"
        or stable_hash(layer_payload) != layer_freeze.get("freeze_hash")
        or layer_freeze.get("final_states_hash") != final_states["freeze_hash"]
        or stable_hash(sweep_rows) != layer_freeze.get("sweep_rows_hash")
        or stable_hash(sweep_manifest) != layer_freeze.get("sweep_manifest_hash")
        or self_manifest.get("status") != "complete"
        or self_manifest.get("completed_rows") != 4
        or self_manifest.get("config", {}).get("causal_layer_freeze_hash")
        != layer_freeze["freeze_hash"]
    ):
        raise ValueError("Chunk 2 artifacts are inconsistent or incomplete")
    metric = layer_freeze["selected_metrics"]
    return {
        "stage": "B",
        "status": "COMPLETE",
        "selected_layer": layer_freeze["selected_layer"],
        "mean_fractional_recovery": metric["mean_fractional_recovery"],
        "mean_raw_recovery": metric["mean_raw_recovery"],
        "positive_authors": metric["positive_authors"],
        "fractional_author_clustered_ci_95": metric[
            "fractional_author_clustered_ci_95"
        ],
        "max_abs_self_patch_effect": self_manifest["max_abs_patch_effect"],
        "max_abs_runtime_baseline_delta": self_manifest[
            "max_abs_runtime_baseline_delta"
        ],
        "live_self_patch_within_tolerance": self_manifest[
            "live_self_patch_within_tolerance"
        ],
        "max_abs_archived_activation_delta": self_manifest[
            "max_abs_activation_delta"
        ],
        "causal_layer_freeze_hash": layer_freeze["freeze_hash"],
    }
