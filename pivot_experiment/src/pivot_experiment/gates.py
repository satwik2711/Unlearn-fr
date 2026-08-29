"""Read-only scientific gate calculations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .config import atomic_json, stable_hash
from .records import read_jsonl, read_unique


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


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _validate_gd_candidate(
    *,
    candidate_id: str,
    candidate_spec: dict,
    artifact_root: Path,
    full_discovery: dict[str, dict],
    retain_discovery: dict[str, dict],
    full_control: dict[str, dict],
    expected_layers: int,
    expected_hidden_size: int,
    frozen_prompt_date: str,
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    root = artifact_root / "scores" / "gd_candidates"
    discovery_path = root / f"{candidate_id}_discovery.jsonl"
    control_path = root / f"{candidate_id}_r_control.jsonl"
    discovery_manifest = _require_complete(discovery_path)
    control_manifest = _require_complete(control_path)
    discovery = read_unique(discovery_path)
    control = read_unique(control_path)
    if set(discovery) != set(retain_discovery) or set(discovery) != set(full_discovery):
        raise ValueError(f"{candidate_id} discovery IDs differ from FULL/RETAIN")
    if set(control) != set(full_control):
        raise ValueError(f"{candidate_id} R_control IDs differ from FULL")
    for manifest, subset in (
        (discovery_manifest, "discovery"),
        (control_manifest, "r_control"),
    ):
        job = manifest["config"]
        if job.get("state") != "gd" or job.get("candidate_id") != candidate_id:
            raise ValueError(f"Unexpected {candidate_id}/{subset} manifest identity")
        if job.get("model") != candidate_spec:
            raise ValueError(f"Unexpected {candidate_id}/{subset} model revision")
        if job.get("subset") != subset or job.get("generation") is not False:
            raise ValueError(f"Unexpected {candidate_id}/{subset} evaluation mode")
        if job.get("frozen_prompt_date") != frozen_prompt_date:
            raise ValueError(f"Wrong frozen prompt date for {candidate_id}/{subset}")
        if not isinstance(job.get("evaluator_version"), int) or job["evaluator_version"] < 3:
            raise ValueError(f"Outdated evaluator version for {candidate_id}/{subset}")
    activation_rows: dict[str, set[int]] = {}
    for example_id, row in discovery.items():
        reference = retain_discovery[example_id]
        if (
            row.get("state") != "gd"
            or row.get("candidate_id") != candidate_id
            or row.get("model_id") != candidate_spec["repo_id"]
            or row.get("model_revision") != candidate_spec["revision"]
            or row.get("subset") != "discovery"
            or row.get("intervention") != "none"
            or row.get("author_id") != reference.get("author_id")
            or row.get("prompt_hash") != reference.get("prompt_hash")
            or row.get("correct_perturbed_margin") is None
        ):
            raise ValueError(f"Invalid GD discovery provenance for {candidate_id}/{example_id}")
        activation_file = row.get("activation_file")
        activation_row = row.get("activation_row")
        if (
            not activation_file
            or not Path(activation_file).is_file()
            or row.get("activation_shape") != [expected_layers, expected_hidden_size]
            or not isinstance(activation_row, int)
        ):
            raise ValueError(f"Invalid GD activation reference for {candidate_id}/{example_id}")
        activation_rows.setdefault(activation_file, set()).add(activation_row)
    if len(activation_rows) != 50 or any(indices != {0, 1} for indices in activation_rows.values()):
        raise ValueError(
            f"{candidate_id} must contain 50 complete two-row activation shards"
        )
    expected_tensor_shape = [2, expected_layers, expected_hidden_size]
    for activation_file in activation_rows:
        with safe_open(activation_file, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["q_end"]:
                raise ValueError(f"Unexpected tensors in {activation_file}")
            if list(handle.get_slice("q_end").get_shape()) != expected_tensor_shape:
                raise ValueError(f"Unexpected q_end tensor shape in {activation_file}")
    for example_id, row in control.items():
        reference = full_control[example_id]
        if (
            row.get("state") != "gd"
            or row.get("candidate_id") != candidate_id
            or row.get("model_id") != candidate_spec["repo_id"]
            or row.get("model_revision") != candidate_spec["revision"]
            or row.get("subset") != "r_control"
            or row.get("intervention") != "none"
            or row.get("author_id") != reference.get("author_id")
            or row.get("prompt_hash") != reference.get("prompt_hash")
            or row.get("activation_file") is not None
        ):
            raise ValueError(f"Invalid GD R_control provenance for {candidate_id}/{example_id}")
    return discovery, control, {
        "discovery_manifest": discovery_manifest,
        "r_control_manifest": control_manifest,
        "activation_files": sorted(activation_rows),
    }


def check_p1(
    config: dict,
    models_config: dict,
    artifact_root: Path,
    p0_result: dict | None = None,
) -> dict:
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
        full_discovery = read_unique(artifact_root / "scores" / "full_discovery.jsonl")
        retain_discovery = read_unique(artifact_root / "scores" / "retain_discovery.jsonl")
        full_control = read_unique(artifact_root / "scores" / "full_r_control.jsonl")
        _require_complete(artifact_root / "scores" / "full_discovery.jsonl")
        _require_complete(artifact_root / "scores" / "retain_discovery.jsonl")
        _require_complete(artifact_root / "scores" / "full_r_control.jsonl")
        retain_scores = [row["mean_target_logprob"] for row in retain_discovery.values()]
        retain_margins = [row["correct_perturbed_margin"] for row in retain_discovery.values()]
        full_control_mean = float(
            np.mean([row["mean_target_logprob"] for row in full_control.values()])
        )
        order = models_config["gd_candidate_order"]
        gates = config["gates"]
        evaluated = []
        for candidate_id in order:
            root = artifact_root / "scores" / "gd_candidates"
            discovery_path = root / f"{candidate_id}_discovery.jsonl"
            control_path = root / f"{candidate_id}_r_control.jsonl"
            if not discovery_path.exists() and not control_path.exists():
                result = {
                    "schema_version": 1,
                    "gate": "P1",
                    "status": "BLOCKED",
                    "reason": f"Next frozen GD candidate has not been evaluated: {candidate_id}",
                    "next_candidate_id": candidate_id,
                    "evaluated_candidates": evaluated,
                }
                break
            candidate_spec = models_config["models"][candidate_id]
            discovery, control, provenance = _validate_gd_candidate(
                candidate_id=candidate_id,
                candidate_spec=candidate_spec,
                artifact_root=artifact_root,
                full_discovery=full_discovery,
                retain_discovery=retain_discovery,
                full_control=full_control,
                expected_layers=models_config["expected_architecture"]["decoder_layers"],
                expected_hidden_size=models_config["expected_architecture"]["hidden_size"],
                frozen_prompt_date=config["evaluation"]["prompt_date"],
            )
            gd_scores = [row["mean_target_logprob"] for row in discovery.values()]
            gd_margins = [row["correct_perturbed_margin"] for row in discovery.values()]
            gd_control_scores = [row["mean_target_logprob"] for row in control.values()]
            behavior_distance = abs(float(np.mean(gd_scores)) - float(np.mean(retain_scores)))
            r_control_degradation = full_control_mean - float(np.mean(gd_control_scores))
            authors = sorted({row["author_id"] for row in discovery.values()})
            author_effects = {
                author: float(
                    np.mean(
                        [
                            discovery[key]["mean_target_logprob"]
                            - retain_discovery[key]["mean_target_logprob"]
                            for key in discovery
                            if discovery[key]["author_id"] == author
                        ]
                    )
                )
                for author in authors
            }
            ci = _clustered_interval(
                np.asarray(list(author_effects.values()), dtype=np.float64),
                config["evaluation"]["bootstrap_resamples"],
                config["seed"],
            )
            checks = {
                "behavior_matched": behavior_distance
                <= gates["p1_max_behavior_distance"],
                "r_control_preserved": r_control_degradation
                <= gates["p1_max_r_control_degradation"],
            }
            metrics = {
                "candidate_id": candidate_id,
                "model": candidate_spec,
                "eligible": all(checks.values()),
                "checks": checks,
                "behavior_distance": behavior_distance,
                "r_control_degradation": r_control_degradation,
                "gd_discovery": _distribution(gd_scores),
                "retain_discovery": _distribution(retain_scores),
                "gd_correct_perturbed_margin": _distribution(gd_margins),
                "retain_correct_perturbed_margin": _distribution(retain_margins),
                "gd_r_control": _distribution(gd_control_scores),
                "full_r_control": _distribution(
                    [row["mean_target_logprob"] for row in full_control.values()]
                ),
                "gd_minus_retain_author_effects": author_effects,
                "gd_minus_retain_author_clustered_ci_95": list(ci),
                "activation_file_count": len(provenance["activation_files"]),
                "input_hash": stable_hash(
                    {
                        "discovery": discovery,
                        "control": control,
                        "provenance": provenance,
                    }
                ),
            }
            evaluated.append(metrics)
            if metrics["eligible"]:
                result = {
                    "schema_version": 1,
                    "gate": "P1",
                    "status": "PASS",
                    "selected_candidate_id": candidate_id,
                    "selected_model": candidate_spec,
                    "checks": checks,
                    "thresholds": {
                        "max_behavior_distance": gates["p1_max_behavior_distance"],
                        "max_r_control_degradation": gates[
                            "p1_max_r_control_degradation"
                        ],
                    },
                    "metrics": metrics,
                    "evaluated_candidates": evaluated,
                    "input_hashes": {
                        "p0": stable_hash(p0_result),
                        "full_discovery": stable_hash(full_discovery),
                        "retain_discovery": stable_hash(retain_discovery),
                        "full_r_control": stable_hash(full_control),
                    },
                }
                atomic_json(
                    artifact_root / "results" / "gd_selection.json", result
                )
                break
        else:
            screen_result = {
                "schema_version": 1,
                "gate": "P1",
                "status": "FAIL",
                "reason": "No frozen GD candidate met behavior-match and utility gates",
                "thresholds": {
                    "max_behavior_distance": gates["p1_max_behavior_distance"],
                    "max_r_control_degradation": gates[
                        "p1_max_r_control_degradation"
                    ],
                },
                "evaluated_candidates": evaluated,
            }
            atomic_json(
                artifact_root / "results" / "gd_candidate_screen.json",
                screen_result,
            )
            freeze_path = artifact_root / "results" / "gd_freeze.json"
            if not freeze_path.exists():
                result = {
                    "schema_version": 1,
                    "gate": "P1",
                    "status": "BLOCKED",
                    "reason": (
                        "Behavior-match screen failed and no explicit exploratory "
                        "GD freeze exists"
                    ),
                    "screen": screen_result,
                }
            else:
                freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
                selected_id = models_config["downstream_gd_candidate"]
                selected = next(
                    row for row in evaluated if row["candidate_id"] == selected_id
                )
                expected_screen_hash = stable_hash(screen_result)
                if freeze.get("candidate_id") != selected_id:
                    raise ValueError("GD freeze disagrees with downstream_gd_candidate")
                if freeze.get("model") != models_config["models"][selected_id]:
                    raise ValueError("GD freeze model provenance is not immutable config")
                if freeze.get("scope") != "exploratory_unmatched":
                    raise ValueError("GD freeze must declare exploratory_unmatched scope")
                if freeze.get("candidate_screen_gate_hash") != expected_screen_hash:
                    raise ValueError("GD freeze does not bind the current candidate screen")
                for key in ("behavior_distance", "r_control_degradation"):
                    if not np.isclose(freeze.get(key), selected[key], atol=1e-12):
                        raise ValueError(f"GD freeze has stale {key}")
                if selected["checks"]["behavior_matched"]:
                    raise ValueError("Exploratory freeze unexpectedly claims behavior match")
                if not selected["checks"]["r_control_preserved"]:
                    raise ValueError("Frozen exploratory GD does not preserve R_control")
                result = {
                    "schema_version": 2,
                    "gate": "P1",
                    "status": "PASS",
                    "qualification": "exploratory_unmatched",
                    "reason": (
                        "Original behavior-match screen failed; gd_02 is frozen "
                        "for exploratory downstream work because it alone preserved R_control"
                    ),
                    "selected_candidate_id": selected_id,
                    "selected_model": models_config["models"][selected_id],
                    "checks": selected["checks"],
                    "metrics": selected,
                    "candidate_screen_status": "FAIL",
                    "candidate_screen_hash": expected_screen_hash,
                    "freeze_hash": stable_hash(freeze),
                    "evaluated_candidates": evaluated,
                }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, TypeError) as error:
        result = {
            "schema_version": 1,
            "gate": "P1",
            "status": "BLOCKED",
            "reason": str(error),
        }
    atomic_json(artifact_root / "gates" / "p1.json", result)
    return result


def check_p2(
    config: dict,
    models_config: dict,
    artifact_root: Path,
    p1_result: dict | None = None,
) -> dict:
    """Check exact matched patches, freeze l*, then check fixed controls."""

    if p1_result is None:
        p1_path = artifact_root / "gates" / "p1.json"
        p1_result = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else None
    if not p1_result or p1_result.get("status") != "PASS":
        result = {
            "schema_version": 1,
            "gate": "P2",
            "status": "BLOCKED",
            "reason": "P1 exploratory GD freeze has not formally passed",
        }
        atomic_json(artifact_root / "gates" / "p2.json", result)
        return result
    try:
        layers = config["patching"]["layers"]
        candidate_id = models_config["downstream_gd_candidate"]
        paths = {
            "gd": artifact_root / "interventions" / "p2_gd_matched.jsonl",
            "retain": artifact_root / "interventions" / "p2_retain_matched.jsonl",
        }
        for path in paths.values():
            _require_complete(path)
        receiver_rows = {state: read_jsonl(path) for state, path in paths.items()}
        indexed: dict[str, dict[tuple[str, int], dict]] = {}
        for state, rows in receiver_rows.items():
            if len(rows) != 100 * len(layers):
                raise ValueError(f"Expected 1600 {state} matched cells, found {len(rows)}")
            cells = {(row["example_id"], row["layer"]): row for row in rows}
            if len(cells) != len(rows):
                raise ValueError(f"Duplicate {state} matched patch cells")
            expected_model = models_config["models"][candidate_id if state == "gd" else "retain"]
            for row in rows:
                if row["state"] != state or row["intervention"] != "matched_full_q_end_patch":
                    raise ValueError(f"Invalid {state} matched intervention record")
                if row["model_id"] != expected_model["repo_id"] or row["model_revision"] != expected_model["revision"]:
                    raise ValueError(f"Invalid {state} model provenance")
            indexed[state] = cells
        if set(indexed["gd"]) != set(indexed["retain"]):
            raise ValueError("GD and RETAIN matched patch grids differ")

        layer_metrics = []
        for layer in layers:
            keys = sorted(key for key in indexed["gd"] if key[1] == layer)
            author_values: dict[str, list[float]] = {}
            differentials = []
            for key in keys:
                gd_row = indexed["gd"][key]
                retain_row = indexed["retain"][key]
                if gd_row["prompt_hash"] != retain_row["prompt_hash"]:
                    raise ValueError(f"Patched receiver prompt mismatch for {key}")
                differential = gd_row["patch_effect"] - retain_row["patch_effect"]
                differentials.append(differential)
                author_values.setdefault(gd_row["author_id"], []).append(differential)
            author_effects = {
                author: float(np.mean(values)) for author, values in author_values.items()
            }
            if len(author_effects) != config["splits"]["discovery_authors"]:
                raise ValueError(f"Layer {layer} does not cover five discovery authors")
            ci = _clustered_interval(
                np.asarray(list(author_effects.values()), dtype=np.float64),
                config["evaluation"]["bootstrap_resamples"],
                config["seed"],
            )
            layer_metrics.append(
                {
                    "layer": layer,
                    "mean_gd_patch_effect": float(
                        np.mean([indexed["gd"][key]["patch_effect"] for key in keys])
                    ),
                    "mean_retain_patch_effect": float(
                        np.mean([indexed["retain"][key]["patch_effect"] for key in keys])
                    ),
                    "mean_differential_recovery": float(np.mean(differentials)),
                    "author_effects": author_effects,
                    "positive_authors": sum(value > 0 for value in author_effects.values()),
                    "author_clustered_ci_95": list(ci),
                }
            )
        selected = max(layer_metrics, key=lambda row: (row["mean_differential_recovery"], -row["layer"]))
        selection = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "scope": "exploratory_unmatched",
            "selected_layer": selected["layer"],
            "selection_rule": "argmax_mean_gd_minus_retain_patch_effect_tie_lowest_layer",
            "selected_metrics": selected,
            "layer_metrics": layer_metrics,
            "matched_inputs_hash": stable_hash(receiver_rows),
        }
        atomic_json(artifact_root / "results" / "p2_layer_selection.json", selection)

        controls_path = artifact_root / "interventions" / "p2_controls.jsonl"
        if not controls_path.exists():
            result = {
                "schema_version": 1,
                "gate": "P2",
                "status": "BLOCKED",
                "reason": (
                    f"Matched sweep complete; run frozen controls at layer {selected['layer']}"
                ),
                "selected_layer": selected["layer"],
                "selected_metrics": selected,
                "next_phase": "controls",
            }
        else:
            _require_complete(controls_path)
            controls = read_jsonl(controls_path)
            expected_self = 2 * config["patching"]["self_patch_example_count"]
            self_rows = [row for row in controls if row["intervention"] == "self_q_end_patch"]
            mismatch_rows = [row for row in controls if row["intervention"] == "mismatched_full_q_end_patch"]
            if len(self_rows) != expected_self or len(mismatch_rows) != 200:
                raise ValueError("P2 control cell counts are incomplete")
            if any(row["layer"] != selected["layer"] for row in controls):
                raise ValueError("P2 controls do not use the frozen selected layer")
            max_self_effect = max(abs(row["patch_effect"]) for row in self_rows)
            specificity_by_state = {}
            for state in ("gd", "retain"):
                matched_mean = selected[f"mean_{state}_patch_effect"]
                mismatch_mean = float(
                    np.mean(
                        [row["patch_effect"] for row in mismatch_rows if row["state"] == state]
                    )
                )
                specificity_by_state[state] = {
                    "matched_mean": matched_mean,
                    "mismatched_mean": mismatch_mean,
                    "paired_advantage": matched_mean - mismatch_mean,
                }
            mean_specificity = float(
                np.mean([row["paired_advantage"] for row in specificity_by_state.values()])
            )
            checks = {
                "self_patch_exact": max_self_effect <= config["gates"]["p2_self_patch_tolerance"],
                "minimum_differential_recovery": selected["mean_differential_recovery"]
                >= config["gates"]["p2_min_differential_recovery"],
                "positive_authors": selected["positive_authors"]
                >= config["gates"]["p2_min_positive_authors"],
                "clustered_ci_above_zero": selected["author_clustered_ci_95"][0] > 0,
                "matched_beats_mismatched": mean_specificity > 0,
            }
            result = {
                "schema_version": 1,
                "gate": "P2",
                "status": "PASS" if all(checks.values()) else "FAIL",
                "scope": "exploratory_unmatched",
                "selected_layer": selected["layer"],
                "selected_metrics": selected,
                "checks": checks,
                "max_abs_self_patch_effect": max_self_effect,
                "specificity_by_state": specificity_by_state,
                "mean_matched_over_mismatched_advantage": mean_specificity,
                "layer_selection_hash": stable_hash(selection),
            }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, TypeError) as error:
        result = {
            "schema_version": 1,
            "gate": "P2",
            "status": "BLOCKED",
            "reason": str(error),
        }
    atomic_json(artifact_root / "gates" / "p2.json", result)
    return result
