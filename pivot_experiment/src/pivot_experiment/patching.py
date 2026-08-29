"""Exact Q_END residual patching with frozen activation sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import atomic_json, stable_hash
from .metrics import EVALUATOR_VERSION, FROZEN_PROMPT_DATE, encode_answer, score_encoded_batch
from .records import append_jsonl, initialize_manifest, read_jsonl, read_unique


class ActivationStore:
    """Lazy, validated reader for evaluator activation sidecars."""

    def __init__(self, rows: dict[str, dict], layers: int, hidden_size: int):
        self.rows = rows
        self.layers = layers
        self.hidden_size = hidden_size
        self._cache: dict[str, torch.Tensor] = {}

    def vector(self, example_id: str, layer: int) -> torch.Tensor:
        row = self.rows[example_id]
        path = row.get("activation_file")
        row_index = row.get("activation_row")
        if path is None or row_index is None:
            raise ValueError(f"Missing activation reference for {example_id}")
        if path not in self._cache:
            activation_path = Path(path)
            if not activation_path.exists():
                raise FileNotFoundError(f"Missing activation sidecar: {activation_path}")
            tensor = load_file(activation_path, device="cpu")["q_end"]
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (
                self.layers,
                self.hidden_size,
            ):
                raise ValueError(
                    f"Invalid activation shape {tuple(tensor.shape)} in {activation_path}"
                )
            self._cache[path] = tensor
        return self._cache[path][row_index, layer].clone()

    def validate_all(self, example_ids: list[str]) -> None:
        for example_id in example_ids:
            for layer in (0, self.layers - 1):
                self.vector(example_id, layer)


def run_matched_patching(
    *,
    model,
    tokenizer,
    receiver_state: str,
    receiver_model_spec: dict,
    candidate_id: str | None,
    rows: list[dict],
    receiver_baselines: dict[str, dict],
    full_donors: dict[str, dict],
    layers: list[int],
    hidden_size: int,
    batch_size: int,
    split_hash: str,
    freeze_hash: str,
    output_path: Path,
) -> None:
    donor_store = ActivationStore(full_donors, len(layers), hidden_size)
    example_ids = [row["example_id"] for row in rows]
    donor_store.validate_all(example_ids)
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "receiver_state": receiver_state,
        "receiver_model": receiver_model_spec,
        "candidate_id": candidate_id,
        "donor_state": "full",
        "intervention": "matched_full_q_end_patch",
        "layers": layers,
        "example_ids": example_ids,
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in rows]
        ),
        "split_hash": split_hash,
        "gd_freeze_hash": freeze_hash,
        "batch_size": batch_size,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    completed_rows = read_jsonl(output_path)
    completed = {row["cell_id"] for row in completed_rows}
    expected = len(rows) * len(layers)
    if len(completed) != len(completed_rows):
        raise ValueError(f"Duplicate patch cells in {output_path}")
    print(f"{receiver_state}/matched_patch: {len(completed)}/{expected} already complete")

    for layer in layers:
        pending = [
            row for row in rows if f"{row['example_id']}:layer-{layer:02d}" not in completed
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            encoded = [
                encode_answer(tokenizer, row["question"], row["answer"]) for row in batch
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
            for row, score in zip(batch, scores, strict=True):
                example_id = row["example_id"]
                baseline = receiver_baselines[example_id]
                donor = full_donors[example_id]
                if score["prompt_hash"] != baseline["prompt_hash"]:
                    raise ValueError(f"Receiver prompt drift for {example_id}")
                if donor["prompt_hash"] != baseline["prompt_hash"]:
                    raise ValueError(f"FULL donor prompt drift for {example_id}")
                cell_id = f"{example_id}:layer-{layer:02d}"
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": cell_id,
                        "state": receiver_state,
                        "model_id": receiver_model_spec["repo_id"],
                        "model_revision": receiver_model_spec["revision"],
                        "candidate_id": candidate_id,
                        "subset": "discovery",
                        "author_id": row["author_id"],
                        "example_id": example_id,
                        "intervention": "matched_full_q_end_patch",
                        "layer": layer,
                        "token_position": "Q_END",
                        "donor_state": "full",
                        "donor_example_id": example_id,
                        "donor_activation_file": donor["activation_file"],
                        "donor_activation_row": donor["activation_row"],
                        "mean_target_logprob": score["mean_target_logprob"],
                        "baseline_mean_target_logprob": baseline["mean_target_logprob"],
                        "patch_effect": score["mean_target_logprob"]
                        - baseline["mean_target_logprob"],
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "split_hash": split_hash,
                        "gd_freeze_hash": freeze_hash,
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
            print(f"{receiver_state}/matched_patch: {len(completed)}/{expected}")

    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} patch cells, found {len(completed)}")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)


def validate_patch_inputs(
    *,
    artifact_root: Path,
    models_config: dict,
    rows: list[dict],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict]:
    candidate_id = models_config["downstream_gd_candidate"]
    freeze_path = artifact_root / "results" / "gd_freeze.json"
    p1_path = artifact_root / "gates" / "p1.json"
    if not freeze_path.exists() or not p1_path.exists():
        raise FileNotFoundError("P1 exploratory freeze artifacts are missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    if p1.get("status") != "PASS" or p1.get("qualification") != "exploratory_unmatched":
        raise ValueError("P1 has not authorized exploratory unmatched patching")
    if freeze.get("candidate_id") != candidate_id:
        raise ValueError("Configured downstream GD and freeze disagree")
    expected_ids = {row["example_id"] for row in rows}
    full = read_unique(artifact_root / "scores" / "full_discovery.jsonl")
    retain = read_unique(artifact_root / "scores" / "retain_discovery.jsonl")
    gd = read_unique(
        artifact_root / "scores" / "gd_candidates" / f"{candidate_id}_discovery.jsonl"
    )
    for name, records in (("FULL", full), ("RETAIN", retain), (candidate_id, gd)):
        if set(records) != expected_ids:
            raise ValueError(f"{name} discovery IDs do not match frozen rows")
    return full, retain, gd, freeze


def mismatched_donor_map(rows: list[dict]) -> dict[str, str]:
    """Frozen within-author cyclic permutation with no self-pairs."""

    by_author: dict[str, list[str]] = {}
    for row in rows:
        by_author.setdefault(row["author_id"], []).append(row["example_id"])
    mapping: dict[str, str] = {}
    for author in sorted(by_author):
        ids = sorted(by_author[author])
        if len(ids) < 2:
            raise ValueError(f"Cannot construct mismatch control for {author}")
        mapping.update({example_id: ids[(index + 1) % len(ids)] for index, example_id in enumerate(ids)})
    if any(left == right for left, right in mapping.items()):
        raise ValueError("Mismatched donor permutation contains a self-pair")
    return mapping


def run_receiver_controls(
    *,
    model,
    tokenizer,
    receiver_state: str,
    receiver_model_spec: dict,
    candidate_id: str | None,
    rows: list[dict],
    receiver_baselines: dict[str, dict],
    own_activation_rows: dict[str, dict] | None,
    full_donors: dict[str, dict],
    selected_layer: int,
    layer_selection_hash: str,
    self_patch_count: int,
    total_layers: int,
    hidden_size: int,
    batch_size: int,
    split_hash: str,
    freeze_hash: str,
    output_path: Path,
    all_receiver_models: dict[str, dict],
) -> None:
    """Run frozen self-patch and mismatched-FULL controls for one receiver."""

    mapping = mismatched_donor_map(rows)
    first_by_author: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (item["author_id"], item["example_id"])):
        first_by_author.setdefault(row["author_id"], row)
    self_rows = [first_by_author[author] for author in sorted(first_by_author)[:self_patch_count]]
    if len(self_rows) != self_patch_count:
        raise ValueError("Not enough discovery authors for the frozen self-patch audit")
    self_ids = [row["example_id"] for row in self_rows]
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "receiver_models": all_receiver_models,
        "candidate_id": candidate_id if receiver_state == "gd" else None,
        "interventions": ["self_q_end_patch", "mismatched_full_q_end_patch"],
        "selected_layer": selected_layer,
        "layer_selection_hash": layer_selection_hash,
        "self_patch_example_ids": self_ids,
        "mismatched_donor_map": mapping,
        "example_ids": [row["example_id"] for row in rows],
        "split_hash": split_hash,
        "gd_freeze_hash": freeze_hash,
        "batch_size": batch_size,
    }
    # Candidate ID is already fixed inside receiver_models; keep the shared
    # manifest identical across the two receiver calls.
    job_config.pop("candidate_id")
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate P2 control cells in {output_path}")
    expected = 2 * (len(rows) + self_patch_count)

    self_pending = [
        row
        for row in self_rows
        if f"{receiver_state}:{row['example_id']}:self" not in completed
    ]
    own_store = (
        ActivationStore(own_activation_rows, total_layers, hidden_size)
        if own_activation_rows is not None
        else None
    )
    for start in range(0, len(self_pending), batch_size):
        batch = self_pending[start : start + batch_size]
        encoded = [encode_answer(tokenizer, row["question"], row["answer"]) for row in batch]
        captured_scores, captured = score_encoded_batch(
            model, tokenizer, encoded, capture_q_end=True
        )
        if captured is None:
            raise RuntimeError("Self-patch audit failed to capture receiver activations")
        if own_store is not None:
            donors = torch.stack(
                [own_store.vector(row["example_id"], selected_layer) for row in batch]
            )
        else:
            donors = captured[:, selected_layer, :]
        patched_scores, _ = score_encoded_batch(
            model,
            tokenizer,
            encoded,
            patch_layer=selected_layer,
            patch_q_end_values=donors,
        )
        records = []
        for row, captured_score, patched_score in zip(batch, captured_scores, patched_scores, strict=True):
            example_id = row["example_id"]
            baseline = receiver_baselines[example_id]
            records.append(
                {
                    "schema_version": 1,
                    "run_id": config_hash,
                    "cell_id": f"{receiver_state}:{example_id}:self",
                    "state": receiver_state,
                    "model_id": receiver_model_spec["repo_id"],
                    "model_revision": receiver_model_spec["revision"],
                    "candidate_id": candidate_id,
                    "subset": "discovery_audit",
                    "author_id": row["author_id"],
                    "example_id": example_id,
                    "intervention": "self_q_end_patch",
                    "layer": selected_layer,
                    "token_position": "Q_END",
                    "donor_state": receiver_state,
                    "donor_example_id": example_id,
                    "mean_target_logprob": patched_score["mean_target_logprob"],
                    "baseline_mean_target_logprob": captured_score["mean_target_logprob"],
                    "stored_baseline_delta": captured_score["mean_target_logprob"] - baseline["mean_target_logprob"],
                    "patch_effect": patched_score["mean_target_logprob"] - captured_score["mean_target_logprob"],
                    "token_count": patched_score["token_count"],
                    "prompt_hash": patched_score["prompt_hash"],
                    "split_hash": split_hash,
                    "gd_freeze_hash": freeze_hash,
                    "layer_selection_hash": layer_selection_hash,
                }
            )
        append_jsonl(output_path, records)
        completed.update(record["cell_id"] for record in records)
        manifest.update(status="running", completed_rows=len(completed), expected_rows=expected)
        atomic_json(manifest_path, manifest)
        print(f"{receiver_state}/controls: {len(completed)}/{expected}")

    full_store = ActivationStore(full_donors, total_layers, hidden_size)
    mismatch_pending = [row for row in rows if f"{receiver_state}:{row['example_id']}:mismatch" not in completed]
    for start in range(0, len(mismatch_pending), batch_size):
        batch = mismatch_pending[start : start + batch_size]
        encoded = [encode_answer(tokenizer, row["question"], row["answer"]) for row in batch]
        donors = torch.stack(
            [full_store.vector(mapping[row["example_id"]], selected_layer) for row in batch]
        )
        scores, _ = score_encoded_batch(
            model,
            tokenizer,
            encoded,
            patch_layer=selected_layer,
            patch_q_end_values=donors,
        )
        records = []
        for row, score in zip(batch, scores, strict=True):
            example_id = row["example_id"]
            baseline = receiver_baselines[example_id]
            records.append(
                {
                    "schema_version": 1,
                    "run_id": config_hash,
                    "cell_id": f"{receiver_state}:{example_id}:mismatch",
                    "state": receiver_state,
                    "model_id": receiver_model_spec["repo_id"],
                    "model_revision": receiver_model_spec["revision"],
                    "candidate_id": candidate_id,
                    "subset": "discovery",
                    "author_id": row["author_id"],
                    "example_id": example_id,
                    "intervention": "mismatched_full_q_end_patch",
                    "layer": selected_layer,
                    "token_position": "Q_END",
                    "donor_state": "full",
                    "donor_example_id": mapping[example_id],
                    "mean_target_logprob": score["mean_target_logprob"],
                    "baseline_mean_target_logprob": baseline["mean_target_logprob"],
                    "patch_effect": score["mean_target_logprob"] - baseline["mean_target_logprob"],
                    "token_count": score["token_count"],
                    "prompt_hash": score["prompt_hash"],
                    "split_hash": split_hash,
                    "gd_freeze_hash": freeze_hash,
                    "layer_selection_hash": layer_selection_hash,
                }
            )
        append_jsonl(output_path, records)
        completed.update(record["cell_id"] for record in records)
        manifest.update(status="running", completed_rows=len(completed), expected_rows=expected)
        atomic_json(manifest_path, manifest)
        print(f"{receiver_state}/controls: {len(completed)}/{expected}")

    if len(completed) == expected:
        manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
        atomic_json(manifest_path, manifest)
