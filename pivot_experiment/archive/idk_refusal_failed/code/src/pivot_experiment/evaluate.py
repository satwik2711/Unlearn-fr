"""Resumable state evaluation and activation sidecars."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import atomic_json, stable_hash
from .metrics import EVALUATOR_VERSION, encode_answer, score_answers, score_encoded_batch
from .records import append_jsonl, initialize_manifest, read_unique


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, temporary)
    os.replace(temporary, path)


def evaluate_subset(
    *,
    model,
    tokenizer,
    state: str,
    model_spec: dict,
    tokenizer_spec: dict,
    adapter_spec: dict | None = None,
    rows: list[dict],
    subset: str,
    output_path: Path,
    activation_dir: Path,
    batch_size: int,
    capture_activations: bool,
    split_hash: str,
) -> None:
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "state": state,
        "model": model_spec,
        "tokenizer": tokenizer_spec,
        "adapter": adapter_spec,
        "subset": subset,
        "example_ids": [row["example_id"] for row in rows],
        "batch_size": batch_size,
        "capture_activations": capture_activations,
        "split_hash": split_hash,
        "generation": False,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    completed = read_unique(output_path)
    pending = [row for row in rows if row["example_id"] not in completed]
    total = len(rows)
    print(f"{state}/{subset}: {len(completed)}/{total} already complete")

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        encoded = [
            encode_answer(tokenizer, row["question"], row["answer"])
            for row in batch
        ]
        correct, activations = score_encoded_batch(
            model,
            tokenizer,
            encoded,
            capture_q_end=capture_activations,
        )

        perturb_pairs: list[tuple[str, str]] = []
        perturb_owner: list[int] = []
        for row_index, row in enumerate(batch):
            for answer in row.get("perturbed_answers", []):
                perturb_pairs.append((row["question"], answer))
                perturb_owner.append(row_index)
        perturb_scores = score_answers(model, tokenizer, perturb_pairs, batch_size)
        grouped: list[list[float]] = [[] for _ in batch]
        for owner, score in zip(perturb_owner, perturb_scores, strict=True):
            grouped[owner].append(score["mean_target_logprob"])

        activation_reference = None
        if activations is not None:
            batch_key = stable_hash([row["example_id"] for row in batch])[:16]
            activation_path = activation_dir / (
                f"{state}_{subset}_{config_hash[:12]}_{batch_key}.safetensors"
            )
            _atomic_safetensors(activation_path, {"q_end": activations})
            activation_reference = str(activation_path.resolve())

        records = []
        for row_index, (row, score) in enumerate(zip(batch, correct, strict=True)):
            variants = grouped[row_index]
            mean_perturbed = sum(variants) / len(variants) if variants else None
            record = {
                "schema_version": 1,
                "run_id": config_hash,
                "state": state,
                "model_id": model_spec["repo_id"],
                "model_revision": model_spec["revision"],
                "adapter_id": adapter_spec["adapter_id"] if adapter_spec else None,
                "adapter_hash": adapter_spec["adapter_hash"] if adapter_spec else None,
                "subset": subset,
                "author_id": row["author_id"],
                "example_id": row["example_id"],
                "intervention": "none",
                "mean_target_logprob": score["mean_target_logprob"],
                "token_count": score["token_count"],
                "prompt_hash": score["prompt_hash"],
                "perturbed_target_logprobs": variants,
                "mean_perturbed_logprob": mean_perturbed,
                "correct_perturbed_margin": (
                    score["mean_target_logprob"] - mean_perturbed
                    if mean_perturbed is not None
                    else None
                ),
                "activation_file": activation_reference,
                "activation_row": row_index if activation_reference else None,
                "activation_shape": list(activations.shape[1:]) if activations is not None else None,
                "split_hash": split_hash,
            }
            records.append(record)
        append_jsonl(output_path, records)
        completed.update({record["example_id"]: record for record in records})
        manifest.update(status="running", completed_rows=len(completed), expected_rows=total)
        atomic_json(manifest_path, manifest)
        print(f"{state}/{subset}: {len(completed)}/{total}")

    if len(completed) != total:
        raise RuntimeError(f"{state}/{subset} ended with {len(completed)}/{total} rows")
    manifest.update(status="complete", completed_rows=total, expected_rows=total)
    atomic_json(manifest_path, manifest)
