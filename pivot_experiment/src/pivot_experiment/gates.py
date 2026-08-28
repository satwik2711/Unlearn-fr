"""Read-only scientific gate calculations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import atomic_json, stable_hash
from .records import read_unique


def _require_complete(score_path: Path) -> dict:
    manifest_path = score_path.with_suffix(".manifest.json")
    if not score_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing score artifact or manifest for {score_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Artifact is not complete: {manifest_path}")
    if manifest.get("completed_rows") != manifest.get("expected_rows"):
        raise RuntimeError(f"Manifest counts disagree: {manifest_path}")
    return manifest


def _clustered_interval(author_effects: np.ndarray, resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    draws = rng.choice(author_effects, size=(resamples, len(author_effects)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def check_p0(config: dict, artifact_root: Path) -> dict:
    full_path = artifact_root / "scores" / "full_discovery.jsonl"
    retain_path = artifact_root / "scores" / "retain_discovery.jsonl"
    try:
        full_manifest = _require_complete(full_path)
        retain_manifest = _require_complete(retain_path)
        full = read_unique(full_path)
        retain = read_unique(retain_path)
        if set(full) != set(retain):
            raise ValueError("FULL and RETAIN discovery example IDs differ")
        if full_manifest["config"]["split_hash"] != retain_manifest["config"]["split_hash"]:
            raise ValueError("FULL and RETAIN use different frozen split hashes")

        rows = []
        for example_id in sorted(full):
            left, right = full[example_id], retain[example_id]
            if left["state"] != "full" or right["state"] != "retain":
                raise ValueError(f"Unexpected model states for {example_id}")
            if left["subset"] != "discovery" or right["subset"] != "discovery":
                raise ValueError(f"Unexpected subset for {example_id}")
            if left["intervention"] != "none" or right["intervention"] != "none":
                raise ValueError(f"P0 received intervened scores for {example_id}")
            if left["author_id"] != right["author_id"]:
                raise ValueError(f"Author mismatch for {example_id}")
            if left["prompt_hash"] != right["prompt_hash"]:
                raise ValueError(f"Prompt/tokenization mismatch for {example_id}")
            if (
                left["correct_perturbed_margin"] is None
                or right["correct_perturbed_margin"] is None
            ):
                raise ValueError(f"Missing perturbed-answer margin for {example_id}")
            rows.append(
                (
                    left["author_id"],
                    left["mean_target_logprob"],
                    right["mean_target_logprob"],
                    left["correct_perturbed_margin"],
                    right["correct_perturbed_margin"],
                )
            )
        authors = sorted({row[0] for row in rows})
        if len(authors) != config["splits"]["discovery_authors"]:
            raise ValueError(f"P0 expected 5 discovery authors, found {len(authors)}")
        author_effects = np.array(
            [
                np.mean([full_score - retain_score for author, full_score, retain_score, _, _ in rows if author == target])
                for target in authors
            ],
            dtype=np.float64,
        )
        mean_difference = float(author_effects.mean())
        ci_low, ci_high = _clustered_interval(
            author_effects,
            config["evaluation"]["bootstrap_resamples"],
            config["seed"],
        )
        full_mean = float(np.mean([row[1] for row in rows]))
        retain_mean = float(np.mean([row[2] for row in rows]))
        margin_difference = float(np.mean([row[3] - row[4] for row in rows]))
        threshold = config["gates"]["p0_min_mean_difference"]
        checks = {
            "full_exceeds_retain": full_mean > retain_mean,
            "clustered_ci_above_zero": ci_low > 0.0,
            "minimum_effect": mean_difference >= threshold,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        result = {
            "schema_version": 1,
            "gate": "P0",
            "status": status,
            "checks": checks,
            "threshold_nats_per_token": threshold,
            "mean_full": full_mean,
            "mean_retain": retain_mean,
            "mean_difference": mean_difference,
            "author_clustered_ci_95": [ci_low, ci_high],
            "correct_perturbed_margin_difference": margin_difference,
            "author_effects": dict(zip(authors, author_effects.tolist(), strict=True)),
            "n_authors": len(authors),
            "n_examples": len(rows),
            "input_hashes": {
                "full": stable_hash(full),
                "retain": stable_hash(retain),
                "split": full_manifest["config"]["split_hash"],
            },
        }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        result = {
            "schema_version": 1,
            "gate": "P0",
            "status": "BLOCKED",
            "reason": str(error),
        }
    atomic_json(artifact_root / "gates" / "p0.json", result)
    return result


def check_p1(config: dict, artifact_root: Path, p0_result: dict | None = None) -> dict:
    if p0_result is None:
        p0_path = artifact_root / "gates" / "p0.json"
        p0_result = json.loads(p0_path.read_text(encoding="utf-8")) if p0_path.exists() else None
    if not p0_result or p0_result.get("status") != "PASS":
        result = {
            "schema_version": 1,
            "gate": "P1",
            "status": "BLOCKED",
            "reason": "P0 has not formally passed",
        }
        atomic_json(artifact_root / "gates" / "p1.json", result)
        return result
    try:
        selection_path = (
            artifact_root / "results" / "idk_suppression_selection.json"
        )
        reversibility_path = (
            artifact_root / "results" / "idk_suppression_reversibility.json"
        )
        training_path = (
            artifact_root / "results" / "idk_suppression_training_summary.json"
        )
        for path in (selection_path, reversibility_path, training_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing P1 artifact: {path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        reversibility = json.loads(reversibility_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if reversibility.get("status") != "complete" or training.get("status") != "complete":
            raise RuntimeError(
                "IDK-suppression training or reversibility audit is incomplete"
            )
        if {
            selection.get("state"),
            reversibility.get("state"),
            training.get("state"),
        } != {"idk_suppression"}:
            raise ValueError("P1 artifacts do not all belong to idk_suppression")
        if training.get("initialization") != "fresh_lora_over_frozen_full":
            raise ValueError("Suppression adapter was not initialized freshly from FULL")
        selected = selection["selected_metrics"]
        if selected["adapter_id"] != selection["selected_adapter_id"]:
            raise ValueError("IDK selection contains conflicting adapter IDs")
        if reversibility["selected_adapter_id"] != selection["selected_adapter_id"]:
            raise ValueError("Reversibility audit used a different IDK adapter")
        if reversibility["selected_adapter_hash"] != selection["selected_adapter_hash"]:
            raise ValueError("Reversibility audit used different adapter bytes")
        gates = config["gates"]
        tolerance = gates["p1_reversibility_tolerance"]
        checks = {
            "full_suppression": selected["full_minus_idk"]
            >= gates["p1_min_full_suppression"],
            "retain_match": selected["retain_distance"]
            <= gates["p1_max_retain_distance"],
            "r_control_preserved": selected["r_control_degradation"]
            <= gates["p1_max_r_control_degradation"],
            "refusal_preferred": selected["refusal_correct_margin"]
            >= gates["p1_min_refusal_margin"],
            "off_before_matches_full": reversibility[
                "off_before_max_abs_from_stored_full"
            ]
            <= tolerance,
            "adapter_on_is_reproducible": reversibility[
                "adapter_on_max_abs_from_stored_idk"
            ]
            <= tolerance,
            "off_after_matches_full": reversibility[
                "off_after_max_abs_from_stored_full"
            ]
            <= tolerance,
            "off_before_equals_off_after": reversibility[
                "off_before_after_max_abs"
            ]
            <= tolerance,
            "base_parameters_unchanged": bool(training["base_hash_match"])
            and training["base_hash_before"] == training["base_hash_after"]
            and reversibility["base_hash_before"] == reversibility["base_hash_after"]
            and training["base_hash_before"] == reversibility["base_hash_before"]
            and training["base_hash_after"] == reversibility["base_hash_after"],
        }
        result = {
            "schema_version": 1,
            "gate": "P1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "selected_adapter_id": selection["selected_adapter_id"],
            "checks": checks,
            "thresholds": {
                "min_full_suppression": gates["p1_min_full_suppression"],
                "max_retain_distance": gates["p1_max_retain_distance"],
                "max_r_control_degradation": gates[
                    "p1_max_r_control_degradation"
                ],
                "min_refusal_margin": gates["p1_min_refusal_margin"],
                "reversibility_tolerance": tolerance,
            },
            "metrics": {
                "full_minus_idk": selected["full_minus_idk"],
                "retain_distance": selected["retain_distance"],
                "r_control_degradation": selected["r_control_degradation"],
                "refusal_correct_margin": selected["refusal_correct_margin"],
                "off_before_max_abs_from_stored_full": reversibility[
                    "off_before_max_abs_from_stored_full"
                ],
                "adapter_on_max_abs_from_stored_idk": reversibility[
                    "adapter_on_max_abs_from_stored_idk"
                ],
                "off_after_max_abs_from_stored_full": reversibility[
                    "off_after_max_abs_from_stored_full"
                ],
                "off_before_after_max_abs": reversibility[
                    "off_before_after_max_abs"
                ],
            },
            "input_hashes": {
                "p0": stable_hash(p0_result),
                "selection": stable_hash(selection),
                "reversibility": stable_hash(reversibility),
                "training": stable_hash(training),
            },
        }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        result = {
            "schema_version": 1,
            "gate": "P1",
            "status": "BLOCKED",
            "reason": str(error),
        }
    atomic_json(artifact_root / "gates" / "p1.json", result)
    return result
