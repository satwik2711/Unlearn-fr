"""Sealed held-out confirmation scoring and specificity controls."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .config import PROJECT_ROOT, atomic_json, stable_hash
from .final_freeze import file_sha256
from .idk_localization import clustered_interval
from .metrics import (
    EVALUATOR_VERSION,
    FROZEN_PROMPT_DATE,
    QEndSteer,
    encode_answer,
    render_question_prompt,
    score_encoded_batch,
)
from .models import decoder_layers
from .records import append_jsonl, initialize_manifest, read_jsonl, read_unique


CONFIRMATION_RECEIVERS = ("idk", "gd02", "retain")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_confirmation_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing confirmation freeze: {path}")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        key: value for key, value in freeze.items() if key != "confirmation_freeze_hash"
    }
    if freeze.get("status") != "FROZEN" or stable_hash(payload) != freeze.get(
        "confirmation_freeze_hash"
    ):
        raise ValueError("Confirmation freeze is invalid or has changed")
    return freeze


def prepare_confirmation_protocol(
    *,
    confirmation_freeze: dict[str, Any],
    direction: torch.Tensor,
    direction_manifest: dict[str, Any],
    generation_settings: dict[str, Any],
    random_tensor_path: Path,
    random_manifest_path: Path,
    execution_freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_generation = {
        "generation_batch_size": int(generation_settings["generation_batch_size"]),
        "max_new_tokens": int(generation_settings["max_new_tokens"]),
        "do_sample": bool(generation_settings["do_sample"]),
        "num_beams": int(generation_settings["num_beams"]),
        "use_cache": bool(generation_settings["use_cache"]),
    }
    if (
        expected_generation["do_sample"] is not False
        or expected_generation["num_beams"] != 1
        or expected_generation["max_new_tokens"] <= 0
    ):
        raise ValueError("Confirmation generation must remain greedy and bounded")
    norm = float(torch.linalg.vector_norm(direction.float()).item())
    if abs(norm - confirmation_freeze["direction"]["l2_norm"]) > 1e-6:
        raise ValueError("Learned direction norm differs from the confirmation freeze")
    if (
        len(confirmation_freeze["random_seeds"]) != 5
        or len(set(confirmation_freeze["random_seeds"])) != 5
    ):
        raise ValueError("Confirmation requires five distinct frozen random seeds")
    random_directions = {}
    random_records = []
    for seed in confirmation_freeze["random_seeds"]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vector = torch.randn(direction.shape, generator=generator, dtype=torch.float32)
        vector = (vector / torch.linalg.vector_norm(vector) * norm).contiguous()
        key = f"random_{seed}"
        random_directions[key] = vector
        random_records.append(
            {
                "condition": key,
                "seed": seed,
                "distribution": "isotropic_standard_normal_rescaled",
                "l2_norm": float(torch.linalg.vector_norm(vector).item()),
                "tensor_sha256": _tensor_sha256(vector),
            }
        )
    random_tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if random_tensor_path.exists():
        existing = load_file(random_tensor_path, device="cpu")
        if set(existing) != set(random_directions) or any(
            not torch.equal(existing[key].float(), value)
            for key, value in random_directions.items()
        ):
            raise ValueError("Frozen random-direction tensor file has changed")
    else:
        temporary = random_tensor_path.with_suffix(random_tensor_path.suffix + ".tmp")
        save_file(random_directions, temporary)
        temporary.replace(random_tensor_path)
    random_payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "source": "five_seeded_isotropic_controls",
        "learned_direction_norm": norm,
        "hidden_size": int(direction.numel()),
        "directions": random_records,
        "tensor_file": random_tensor_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "file_sha256": file_sha256(random_tensor_path),
        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "confirmation_read": False,
    }
    random_manifest = {
        **random_payload,
        "random_freeze_hash": stable_hash(random_payload),
    }
    if random_manifest_path.exists():
        existing = json.loads(random_manifest_path.read_text(encoding="utf-8"))
        if existing != random_manifest:
            raise ValueError("Refusing to overwrite changed random-direction manifest")
        random_manifest = existing
    else:
        atomic_json(random_manifest_path, random_manifest)
    execution_payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "role": "pre_confirmation_engineering_supplement",
        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
        "final_states_hash": confirmation_freeze["final_states_hash"],
        "layer": confirmation_freeze["layer"],
        "selected_alpha": confirmation_freeze["selected_alpha"],
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "random_freeze_hash": random_manifest["random_freeze_hash"],
        "likelihood_conditions": [
            "baseline",
            "learned",
            *[row["condition"] for row in random_records],
        ],
        "generation_conditions": ["full", "baseline", "learned"],
        "generation_settings": expected_generation,
        "generation_is_diagnostic": True,
        "likelihood_metric": "mean_target_logprob_nats_per_token",
        "author_aggregation": "mean_within_author_then_mean_across_authors",
        "clustered_bootstrap_resamples": confirmation_freeze["bootstrap_resamples"],
        "random_rank_direction": "descending_fractional_recovery",
        "confirmation_read_before_execution_freeze": False,
    }
    execution = {
        **execution_payload,
        "execution_freeze_hash": stable_hash(execution_payload),
    }
    if execution_freeze_path.exists():
        existing = json.loads(execution_freeze_path.read_text(encoding="utf-8"))
        if existing != execution:
            raise ValueError("Refusing to overwrite changed confirmation execution freeze")
        execution = existing
    else:
        atomic_json(execution_freeze_path, execution)
    return random_manifest, execution


def load_confirmation_protocol(
    *,
    confirmation_freeze: dict[str, Any],
    direction_manifest: dict[str, Any],
    random_tensor_path: Path,
    random_manifest_path: Path,
    execution_freeze_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    random_manifest = json.loads(random_manifest_path.read_text(encoding="utf-8"))
    random_payload = {
        key: value for key, value in random_manifest.items() if key != "random_freeze_hash"
    }
    execution = json.loads(execution_freeze_path.read_text(encoding="utf-8"))
    execution_payload = {
        key: value for key, value in execution.items() if key != "execution_freeze_hash"
    }
    tensors = load_file(random_tensor_path, device="cpu")
    if (
        random_manifest.get("status") != "FROZEN"
        or stable_hash(random_payload) != random_manifest.get("random_freeze_hash")
        or random_manifest.get("file_sha256") != file_sha256(random_tensor_path)
        or random_manifest.get("confirmation_freeze_hash")
        != confirmation_freeze["confirmation_freeze_hash"]
        or random_manifest.get("direction_freeze_hash")
        != direction_manifest["direction_freeze_hash"]
        or execution.get("status") != "FROZEN"
        or stable_hash(execution_payload) != execution.get("execution_freeze_hash")
        or execution.get("confirmation_freeze_hash")
        != confirmation_freeze["confirmation_freeze_hash"]
        or execution.get("random_freeze_hash") != random_manifest["random_freeze_hash"]
    ):
        raise ValueError("Confirmation execution protocol is invalid or has changed")
    expected_keys = {row["condition"] for row in random_manifest["directions"]}
    if set(tensors) != expected_keys:
        raise ValueError("Random-direction tensors differ from their frozen manifest")
    for row in random_manifest["directions"]:
        tensor = tensors[row["condition"]].float()
        if (
            _tensor_sha256(tensor) != row["tensor_sha256"]
            or abs(float(torch.linalg.vector_norm(tensor).item()) - row["l2_norm"]) > 1e-6
        ):
            raise ValueError(f"Random direction changed: {row['condition']}")
    return {key: value.float() for key, value in tensors.items()}, random_manifest, execution


def _model_spec(receiver: str, final_states: dict[str, Any]) -> dict[str, str]:
    if receiver == "idk":
        return final_states["states"]["idk"]["base_model"]
    return final_states["states"][receiver]["model"]


def _condition_vectors(
    learned: torch.Tensor, randoms: dict[str, torch.Tensor]
) -> list[tuple[str, torch.Tensor | None]]:
    return [("baseline", None), ("learned", learned), *sorted(randoms.items())]


def run_confirmation_scores(
    *,
    model,
    tokenizer,
    state: str,
    rows: list[dict],
    learned_direction: torch.Tensor,
    random_directions: dict[str, torch.Tensor],
    final_states: dict[str, Any],
    confirmation_freeze: dict[str, Any],
    execution: dict[str, Any],
    direction_manifest: dict[str, Any],
    random_manifest: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> None:
    if state not in ("full", *CONFIRMATION_RECEIVERS):
        raise ValueError(f"Unknown confirmation state: {state}")
    model_spec = final_states["states"]["full"]["model"] if state == "full" else _model_spec(state, final_states)
    conditions = [("baseline", None)] if state == "full" else _condition_vectors(
        learned_direction, random_directions
    )
    job_config = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "frozen_prompt_date": FROZEN_PROMPT_DATE,
        "state": state,
        "model": model_spec,
        "adapter_id": final_states["states"].get(state, {}).get("adapter_id"),
        "adapter_hash": final_states["states"].get(state, {}).get("adapter_hash"),
        "candidate_id": final_states["states"].get(state, {}).get("candidate_id"),
        "subset": "confirmation",
        "conditions": [condition for condition, _ in conditions],
        "layer": confirmation_freeze["layer"] if state != "full" else None,
        "alpha": confirmation_freeze["selected_alpha"] if state != "full" else None,
        "example_ids": [row["example_id"] for row in rows],
        "row_targets_hash": stable_hash(
            [(row["example_id"], row["question"], row["answer"]) for row in rows]
        ),
        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
        "execution_freeze_hash": execution["execution_freeze_hash"],
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "random_freeze_hash": random_manifest["random_freeze_hash"],
        "final_states_hash": final_states["freeze_hash"],
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
        raise ValueError(f"Duplicate confirmation scores for {state}")
    expected = len(rows) * len(conditions)
    print(f"{state}/confirmation_scores: {len(completed)}/{expected} already complete")
    for condition, vector in conditions:
        pending = [
            row
            for row in rows
            if f"{condition}:{row['example_id']}" not in completed
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            encoded = [
                encode_answer(tokenizer, row["question"], row["answer"]) for row in batch
            ]
            if vector is None:
                scores, _ = score_encoded_batch(model, tokenizer, encoded)
            else:
                scores, _ = score_encoded_batch(
                    model,
                    tokenizer,
                    encoded,
                    steer_layer=confirmation_freeze["layer"],
                    steer_direction=vector,
                    steer_alpha=confirmation_freeze["selected_alpha"],
                )
            records = []
            for prepared, score in zip(batch, scores, strict=True):
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": f"{condition}:{prepared['example_id']}",
                        "state": state,
                        "subset": "confirmation",
                        "author_id": prepared["author_id"],
                        "example_id": prepared["example_id"],
                        "condition": condition,
                        "intervention": (
                            "none" if vector is None else "add_direction_at_q_end"
                        ),
                        "layer": confirmation_freeze["layer"] if vector is not None else None,
                        "token_position": "Q_END" if vector is not None else None,
                        "alpha": confirmation_freeze["selected_alpha"] if vector is not None else None,
                        "direction_tensor_sha256": (
                            None
                            if vector is None
                            else direction_manifest["tensor_sha256"]
                            if condition == "learned"
                            else next(
                                row["tensor_sha256"]
                                for row in random_manifest["directions"]
                                if row["condition"] == condition
                            )
                        ),
                        "model_id": model_spec["repo_id"],
                        "model_revision": model_spec["revision"],
                        "adapter_id": final_states["states"].get(state, {}).get("adapter_id"),
                        "candidate_id": final_states["states"].get(state, {}).get("candidate_id"),
                        "mean_target_logprob": score["mean_target_logprob"],
                        "token_count": score["token_count"],
                        "prompt_hash": score["prompt_hash"],
                        "split_hash": final_states["split"]["split_hash"],
                        "final_states_hash": final_states["freeze_hash"],
                        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
                        "execution_freeze_hash": execution["execution_freeze_hash"],
                    }
                )
            append_jsonl(output_path, records)
            completed.update(record["cell_id"] for record in records)
            manifest.update(status="running", completed_rows=len(completed), expected_rows=expected)
            atomic_json(manifest_path, manifest)
            print(f"{state}/confirmation_scores: {len(completed)}/{expected}")
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} confirmation scores for {state}")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)


def _generate_batch(
    *,
    model,
    tokenizer,
    rows: list[dict],
    direction: torch.Tensor | None,
    layer: int | None,
    alpha: float | None,
    settings: dict[str, Any],
) -> list[dict]:
    rendered = [render_question_prompt(tokenizer, row["question"]) for row in rows]
    prompts = [item[0] for item in rendered]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    context = None
    if direction is not None:
        if layer is None or alpha is None:
            raise ValueError("Generation steering requires layer and alpha")
        q_end = torch.full((len(rows),), input_ids.shape[1] - 1, dtype=torch.long)
        context = QEndSteer(
            decoder_layers(model)[layer],
            q_end,
            direction,
            alpha,
            apply_once=True,
        )
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": settings["do_sample"],
        "num_beams": settings["num_beams"],
        "use_cache": settings["use_cache"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        if context is None:
            generated = model.generate(**kwargs)
        else:
            with context:
                generated = model.generate(**kwargs)
    new_tokens = generated[:, input_ids.shape[1] :].to("cpu")
    results = []
    for index, tokens in enumerate(new_tokens):
        token_ids = tokens.tolist()
        if tokenizer.pad_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(tokenizer.pad_token_id)]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        results.append(
            {
                "text": text,
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "prompt_hash": rendered[index][2],
                "ended_with_eos": tokenizer.eos_token_id in tokens.tolist(),
            }
        )
    return results


def run_confirmation_generations(
    *,
    model,
    tokenizer,
    state: str,
    rows: list[dict],
    learned_direction: torch.Tensor,
    final_states: dict[str, Any],
    confirmation_freeze: dict[str, Any],
    execution: dict[str, Any],
    direction_manifest: dict[str, Any],
    output_path: Path,
) -> None:
    conditions = [("full", None)] if state == "full" else [
        ("baseline", None),
        ("learned", learned_direction),
    ]
    model_spec = final_states["states"]["full"]["model"] if state == "full" else _model_spec(state, final_states)
    settings = execution["generation_settings"]
    job_config = {
        "schema_version": 1,
        "state": state,
        "model": model_spec,
        "adapter_id": final_states["states"].get(state, {}).get("adapter_id"),
        "candidate_id": final_states["states"].get(state, {}).get("candidate_id"),
        "subset": "confirmation",
        "conditions": [condition for condition, _ in conditions],
        "settings": settings,
        "example_ids": [row["example_id"] for row in rows],
        "row_questions_hash": stable_hash(
            [(row["example_id"], row["question"]) for row in rows]
        ),
        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
        "execution_freeze_hash": execution["execution_freeze_hash"],
        "direction_freeze_hash": direction_manifest["direction_freeze_hash"],
        "final_states_hash": final_states["freeze_hash"],
        "split_hash": final_states["split"]["split_hash"],
        "generation": True,
    }
    config_hash = stable_hash(job_config)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = initialize_manifest(manifest_path, job_config, config_hash)
    existing = read_jsonl(output_path)
    completed = {row["cell_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate confirmation generations for {state}")
    expected = len(rows) * len(conditions)
    print(f"{state}/confirmation_generations: {len(completed)}/{expected} already complete")
    batch_size = settings["generation_batch_size"]
    for condition, direction in conditions:
        pending = [
            row
            for row in rows
            if f"{condition}:{row['example_id']}" not in completed
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            outputs = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                rows=batch,
                direction=direction,
                layer=confirmation_freeze["layer"] if direction is not None else None,
                alpha=confirmation_freeze["selected_alpha"] if direction is not None else None,
                settings=settings,
            )
            records = []
            for prepared, output in zip(batch, outputs, strict=True):
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": config_hash,
                        "cell_id": f"{condition}:{prepared['example_id']}",
                        "state": state,
                        "subset": "confirmation",
                        "author_id": prepared["author_id"],
                        "example_id": prepared["example_id"],
                        "condition": condition,
                        "intervention": (
                            "add_full_minus_idk_at_q_end"
                            if direction is not None
                            else "none"
                        ),
                        "layer": confirmation_freeze["layer"] if direction is not None else None,
                        "alpha": confirmation_freeze["selected_alpha"] if direction is not None else None,
                        **output,
                        "model_id": model_spec["repo_id"],
                        "model_revision": model_spec["revision"],
                        "split_hash": final_states["split"]["split_hash"],
                        "final_states_hash": final_states["freeze_hash"],
                        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
                        "execution_freeze_hash": execution["execution_freeze_hash"],
                    }
                )
            append_jsonl(output_path, records)
            completed.update(record["cell_id"] for record in records)
            manifest.update(status="running", completed_rows=len(completed), expected_rows=expected)
            atomic_json(manifest_path, manifest)
            print(f"{state}/confirmation_generations: {len(completed)}/{expected}")
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} confirmation generations for {state}")
    manifest.update(status="complete", completed_rows=expected, expected_rows=expected)
    atomic_json(manifest_path, manifest)


def _confirmation_metric(
    *,
    condition_rows: list[dict],
    baselines: dict[str, dict],
    full: dict[str, dict],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    by_author: dict[str, list[dict]] = defaultdict(list)
    for row in condition_rows:
        by_author[row["author_id"]].append(row)
    if len(by_author) != 5 or any(len(rows) != 20 for rows in by_author.values()):
        raise ValueError("Confirmation condition lacks five complete authors")
    author_raw = {}
    author_headroom = {}
    for author, rows in sorted(by_author.items()):
        author_raw[author] = float(
            np.mean(
                [
                    row["mean_target_logprob"]
                    - baselines[row["example_id"]]["mean_target_logprob"]
                    for row in rows
                ]
            )
        )
        author_headroom[author] = float(
            np.mean(
                [
                    full[row["example_id"]]["mean_target_logprob"]
                    - baselines[row["example_id"]]["mean_target_logprob"]
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
    rf_values = list(author_rf.values())
    return {
        "mean_raw_recovery": float(np.mean(list(author_raw.values()))),
        "mean_fractional_recovery": float(np.mean(rf_values)) if rf_values else None,
        "author_raw_recovery": author_raw,
        "author_headroom": author_headroom,
        "author_fractional_recovery": author_rf,
        "unstable_authors": unstable,
        "positive_authors": sum(value > 0 for value in rf_values),
        "fractional_author_clustered_ci_95": (
            clustered_interval(rf_values, bootstrap_resamples, seed) if rf_values else None
        ),
        "raw_author_clustered_ci_95": clustered_interval(
            list(author_raw.values()), bootstrap_resamples, seed
        ),
    }


def _rank_descending(learned: float, random_values: dict[str, float]) -> int:
    return 1 + sum(value > learned for value in random_values.values())


def finalize_confirmation(
    *,
    artifact_root: Path,
    final_states: dict[str, Any],
    confirmation_freeze: dict[str, Any],
    execution: dict[str, Any],
    random_manifest: dict[str, Any],
    bootstrap_resamples: int,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    states = ("full", *CONFIRMATION_RECEIVERS)
    score_data = {}
    generation_data = {}
    source_hashes = {"scores": {}, "generations": {}}
    for state in states:
        score_path = artifact_root / "confirmation" / "scores" / f"{state}.jsonl"
        generation_path = artifact_root / "confirmation" / "generations" / f"{state}.jsonl"
        for kind, path in (("scores", score_path), ("generations", generation_path)):
            manifest_path = path.with_suffix(".manifest.json")
            if not path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Missing confirmation {kind}: {path}")
            rows = read_jsonl(path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = (100 if state == "full" else 700) if kind == "scores" else (
                100 if state == "full" else 200
            )
            if (
                len(rows) != expected
                or manifest.get("status") != "complete"
                or manifest.get("completed_rows") != expected
                or manifest.get("config", {}).get("confirmation_freeze_hash")
                != confirmation_freeze["confirmation_freeze_hash"]
                or manifest.get("config", {}).get("execution_freeze_hash")
                != execution["execution_freeze_hash"]
                or any(row.get("subset") != "confirmation" for row in rows)
                or any(row.get("state") != state for row in rows)
            ):
                raise ValueError(f"Invalid or incomplete confirmation {kind} for {state}")
            if len({row.get("cell_id") for row in rows}) != expected:
                raise ValueError(f"Duplicate confirmation {kind} cells for {state}")
            source_hashes[kind][state] = {
                "rows_hash": stable_hash(rows),
                "manifest_hash": stable_hash(manifest),
            }
            if kind == "scores":
                score_data[state] = rows
            else:
                generation_data[state] = rows
    full = {
        row["example_id"]: row
        for row in score_data["full"]
        if row["condition"] == "baseline"
    }
    if len(full) != 100:
        raise ValueError("FULL confirmation denominator is incomplete")
    expected_score_conditions = execution["likelihood_conditions"]
    expected_generation_conditions = {
        "full": ["full"],
        "idk": ["baseline", "learned"],
        "gd02": ["baseline", "learned"],
        "retain": ["baseline", "learned"],
    }
    for example_id, full_row in full.items():
        if (
            full_row.get("intervention") != "none"
            or full_row.get("final_states_hash") != final_states["freeze_hash"]
            or full_row.get("confirmation_freeze_hash")
            != confirmation_freeze["confirmation_freeze_hash"]
            or full_row.get("execution_freeze_hash") != execution["execution_freeze_hash"]
        ):
            raise ValueError(f"Invalid FULL confirmation row: {example_id}")
    for state in CONFIRMATION_RECEIVERS:
        model_spec = _model_spec(state, final_states)
        rows = score_data[state]
        for condition in expected_score_conditions:
            condition_rows = [row for row in rows if row["condition"] == condition]
            if len(condition_rows) != 100:
                raise ValueError(f"{state}/{condition} does not contain 100 scores")
            for row in condition_rows:
                full_row = full.get(row["example_id"])
                if (
                    full_row is None
                    or row.get("prompt_hash") != full_row["prompt_hash"]
                    or row.get("token_count") != full_row["token_count"]
                    or row.get("model_id") != model_spec["repo_id"]
                    or row.get("model_revision") != model_spec["revision"]
                    or row.get("final_states_hash") != final_states["freeze_hash"]
                    or row.get("confirmation_freeze_hash")
                    != confirmation_freeze["confirmation_freeze_hash"]
                    or row.get("execution_freeze_hash")
                    != execution["execution_freeze_hash"]
                ):
                    raise ValueError(
                        f"Confirmation provenance drift at {state}/{condition}/{row.get('example_id')}"
                    )
    for state, rows in generation_data.items():
        expected_conditions = expected_generation_conditions[state]
        for condition in expected_conditions:
            condition_rows = [row for row in rows if row["condition"] == condition]
            if len(condition_rows) != 100:
                raise ValueError(f"{state}/{condition} does not contain 100 generations")
            for row in condition_rows:
                full_row = full.get(row["example_id"])
                if (
                    full_row is None
                    or row.get("prompt_hash") != full_row["prompt_hash"]
                    or row.get("final_states_hash") != final_states["freeze_hash"]
                    or row.get("confirmation_freeze_hash")
                    != confirmation_freeze["confirmation_freeze_hash"]
                    or row.get("execution_freeze_hash")
                    != execution["execution_freeze_hash"]
                ):
                    raise ValueError(
                        f"Generation provenance drift at {state}/{condition}/{row.get('example_id')}"
                    )
    conditions = ["learned", *[row["condition"] for row in random_manifest["directions"]]]
    metrics = {}
    for state in CONFIRMATION_RECEIVERS:
        rows = score_data[state]
        baselines = {
            row["example_id"]: row for row in rows if row["condition"] == "baseline"
        }
        if len(baselines) != 100:
            raise ValueError(f"{state} confirmation baseline is incomplete")
        metrics[state] = {}
        for condition in conditions:
            selected = [row for row in rows if row["condition"] == condition]
            if len(selected) != 100:
                raise ValueError(f"{state}/{condition} confirmation condition is incomplete")
            metrics[state][condition] = _confirmation_metric(
                condition_rows=selected,
                baselines=baselines,
                full=full,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
    idk_learned = metrics["idk"]["learned"]
    idk_random = {
        condition: metrics["idk"][condition]["mean_fractional_recovery"]
        for condition in conditions
        if condition != "learned"
    }
    if idk_learned["mean_fractional_recovery"] is None or any(
        value is None for value in idk_random.values()
    ):
        raise ValueError("IDK confirmation contains unstable denominators")
    transfer_by_condition = {}
    for condition in conditions:
        gd = metrics["gd02"][condition]
        retain = metrics["retain"][condition]
        common = sorted(
            set(gd["author_fractional_recovery"])
            & set(retain["author_fractional_recovery"])
        )
        if not common:
            raise ValueError(f"No stable transfer authors for {condition}")
        author_differential = {
            author: gd["author_fractional_recovery"][author]
            - retain["author_fractional_recovery"][author]
            for author in common
        }
        values = list(author_differential.values())
        transfer_by_condition[condition] = {
            "mean_differential_fractional_recovery": float(np.mean(values)),
            "author_differential": author_differential,
            "positive_authors": sum(value > 0 for value in values),
            "author_clustered_ci_95": clustered_interval(
                values, bootstrap_resamples, seed
            ),
            "gd02": gd,
            "retain": retain,
        }
    learned_transfer = transfer_by_condition["learned"]
    random_transfer = {
        condition: transfer_by_condition[condition]["mean_differential_fractional_recovery"]
        for condition in conditions
        if condition != "learned"
    }
    payload = {
        "schema_version": 1,
        "stage": "held_out_confirmation",
        "status": "COMPLETE",
        "scientific_gate": False,
        "confirmation_freeze_hash": confirmation_freeze["confirmation_freeze_hash"],
        "execution_freeze_hash": execution["execution_freeze_hash"],
        "random_freeze_hash": random_manifest["random_freeze_hash"],
        "final_states_hash": final_states["freeze_hash"],
        "layer": confirmation_freeze["layer"],
        "alpha": confirmation_freeze["selected_alpha"],
        "c_idk": idk_learned["mean_fractional_recovery"],
        "idk_learned": idk_learned,
        "idk_random_fractional_recovery": idk_random,
        "idk_learned_rank_among_six": _rank_descending(
            idk_learned["mean_fractional_recovery"], idk_random
        ),
        "c_transfer": learned_transfer["mean_differential_fractional_recovery"],
        "transfer_learned": learned_transfer,
        "transfer_random_differential": random_transfer,
        "transfer_learned_rank_among_six": _rank_descending(
            learned_transfer["mean_differential_fractional_recovery"], random_transfer
        ),
        "all_condition_metrics": metrics,
        "all_transfer_conditions": transfer_by_condition,
        "generation_diagnostics_stored": True,
        "random_control_count": 5,
        "specificity_note": "five controls provide a rank comparison, not a precise tail probability",
        "source_hashes": source_hashes,
        "no_retuning_after_confirmation": True,
    }
    result = {**payload, "result_hash": stable_hash(payload)}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(f"Refusing to overwrite changed confirmation result: {output_path}")
        return existing
    atomic_json(output_path, result)
    return result


def audit_chunk5(
    *,
    artifact_root: Path,
    final_states: dict[str, Any],
    confirmation_freeze: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    result_path = artifact_root / "results" / "confirmation.json"
    if not result_path.is_file():
        return {
            "stage": "E",
            "status": "INCOMPLETE",
            "reason": "held-out confirmation result is missing",
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in result.items() if key != "result_hash"}
    if (
        result.get("status") != "COMPLETE"
        or stable_hash(payload) != result.get("result_hash")
        or result.get("confirmation_freeze_hash")
        != confirmation_freeze["confirmation_freeze_hash"]
        or result.get("execution_freeze_hash") != execution["execution_freeze_hash"]
        or result.get("final_states_hash") != final_states["freeze_hash"]
        or result.get("no_retuning_after_confirmation") is not True
    ):
        raise ValueError("Held-out confirmation result is invalid or inconsistent")
    for kind, states in result["source_hashes"].items():
        for state, hashes in states.items():
            path = artifact_root / "confirmation" / kind / f"{state}.jsonl"
            manifest = path.with_suffix(".manifest.json")
            if (
                stable_hash(read_jsonl(path)) != hashes["rows_hash"]
                or stable_hash(json.loads(manifest.read_text(encoding="utf-8")))
                != hashes["manifest_hash"]
            ):
                raise ValueError(f"Confirmation {kind}/{state} changed after finalization")
    return result
