"""Model-free validation and immutable state freeze for the final experiment."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from .config import PROJECT_ROOT, atomic_json, stable_hash
from .records import read_jsonl, read_unique


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Final-state config path must be relative: {relative_path}")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Final-state path escapes project root: {relative_path}") from error
    return resolved


def portable_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def require_complete(score_path: Path) -> dict[str, Any]:
    manifest_path = score_path.with_suffix(".manifest.json")
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing score or manifest: {score_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete score manifest: {manifest_path}")
    rows = read_jsonl(score_path)
    if (
        manifest.get("completed_rows") != len(rows)
        or manifest.get("expected_rows") != len(rows)
    ):
        raise ValueError(f"Score/manifest count mismatch: {score_path}")
    return manifest


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "std": float(array.std()),
    }


def validate_score_state(
    *,
    name: str,
    score_path: Path,
    expected_state: str,
    expected_subset: str,
    expected_model: dict[str, str],
    split_hash: str,
    expected_ids: set[str] | None = None,
    adapter: dict[str, Any] | None = None,
    candidate_id: str | None = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    manifest = require_complete(score_path)
    rows = read_unique(score_path)
    if len(rows) != 100:
        raise ValueError(f"{name} must contain exactly 100 unique rows")
    if expected_ids is not None and set(rows) != expected_ids:
        raise ValueError(f"{name} example IDs do not match the frozen subset")
    for example_id, row in rows.items():
        if (
            row.get("state") != expected_state
            or row.get("subset") != expected_subset
            or row.get("intervention") != "none"
            or row.get("model_id") != expected_model["repo_id"]
            or row.get("model_revision") != expected_model["revision"]
            or row.get("split_hash") != split_hash
            or not isinstance(row.get("prompt_hash"), str)
            or not isinstance(row.get("token_count"), int)
            or not isinstance(row.get("mean_target_logprob"), (int, float))
        ):
            raise ValueError(f"Invalid provenance for {name}/{example_id}")
        if adapter is not None and (
            row.get("adapter_id") != adapter["adapter_id"]
            or row.get("adapter_hash") != adapter["adapter_hash"]
        ):
            raise ValueError(f"Invalid adapter provenance for {name}/{example_id}")
        if candidate_id is not None and row.get("candidate_id") != candidate_id:
            raise ValueError(f"Invalid candidate provenance for {name}/{example_id}")
    job = manifest.get("config", {})
    if (
        job.get("state") != expected_state
        or job.get("subset") != expected_subset
        or job.get("split_hash") != split_hash
        or job.get("generation") is not False
    ):
        raise ValueError(f"Invalid manifest provenance for {name}")
    return rows, manifest


def validate_pair_alignment(
    left_name: str,
    left: dict[str, dict],
    right_name: str,
    right: dict[str, dict],
) -> None:
    if set(left) != set(right):
        raise ValueError(f"{left_name}/{right_name} example IDs differ")
    for example_id in left:
        a, b = left[example_id], right[example_id]
        if (
            a["author_id"] != b["author_id"]
            or a["prompt_hash"] != b["prompt_hash"]
            or a["token_count"] != b["token_count"]
        ):
            raise ValueError(f"{left_name}/{right_name} alignment drift at {example_id}")


def validate_activations(
    *,
    name: str,
    rows: dict[str, dict],
    activation_directory: Path,
    expected_layers: int,
    expected_hidden_size: int,
) -> dict[str, Any]:
    by_file: dict[Path, set[int]] = defaultdict(set)
    for example_id, row in rows.items():
        stored = row.get("activation_file")
        activation_row = row.get("activation_row")
        if (
            not isinstance(stored, str)
            or not isinstance(activation_row, int)
            or row.get("activation_shape") != [expected_layers, expected_hidden_size]
        ):
            raise ValueError(f"Missing activation provenance for {name}/{example_id}")
        # Old records contain absolute pre-archive paths. Resolve only the
        # basename inside the configured, project-relative archive directory.
        resolved = activation_directory / Path(stored).name
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing resolved activation: {resolved}")
        by_file[resolved].add(activation_row)
    if len(by_file) != 50 or any(indices != {0, 1} for indices in by_file.values()):
        raise ValueError(f"{name} must use 50 complete two-row activation shards")

    file_records = []
    for path in sorted(by_file):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["q_end"]:
                raise ValueError(f"Unexpected tensors in {path}")
            shape = list(handle.get_slice("q_end").get_shape())
            if shape != [2, expected_layers, expected_hidden_size]:
                raise ValueError(f"Unexpected q_end shape {shape} in {path}")
        file_records.append(
            {
                "path": portable_path(path),
                "sha256": file_sha256(path),
            }
        )
    return {
        "directory": portable_path(activation_directory),
        "file_count": len(file_records),
        "tensor_name": "q_end",
        "tensor_shape": [2, expected_layers, expected_hidden_size],
        "files_hash": stable_hash(file_records),
        "files": file_records,
    }


def assert_confirmation_unopened(artifact_root: Path) -> dict[str, Any]:
    confirmation_rows = 0
    for directory_name in ("scores", "interventions"):
        directory = artifact_root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.jsonl")):
            rows = read_jsonl(path)
            found = sum(row.get("subset") == "confirmation" for row in rows)
            confirmation_rows += found
            if found:
                raise ValueError(f"Confirmation has already been scored in {path}")
    return {
        "status": "sealed",
        "scored_confirmation_rows": confirmation_rows,
    }


def create_final_states_freeze(
    *,
    experiment_config: dict[str, Any],
    models_config: dict[str, Any],
    artifact_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    split_path = artifact_root / "splits" / "frozen_splits.json"
    frozen_split = json.loads(split_path.read_text(encoding="utf-8"))
    split_hash = frozen_split["split_hash"]
    layers = models_config["expected_architecture"]["decoder_layers"]
    hidden_size = models_config["expected_architecture"]["hidden_size"]
    if experiment_config["evaluation"]["prompt_date"] != "28 Aug 2026":
        raise ValueError("The final experiment requires frozen prompt date 28 Aug 2026")
    if models_config.get("downstream_gd_candidate") != "gd_02":
        raise ValueError("The final experiment requires frozen GD02")

    full_spec = models_config["models"]["full"]
    retain_spec = models_config["models"]["retain"]
    gd_spec = models_config["models"]["gd_02"]
    idk_spec = models_config["idk"]
    adapter_path = project_path(idk_spec["adapter_path"])
    adapter_file = adapter_path / "adapter_model.safetensors"
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_file.is_file() or not adapter_config_path.is_file():
        raise FileNotFoundError(f"Incomplete archived IDK adapter: {adapter_path}")
    adapter_hash = file_sha256(adapter_file)
    if adapter_hash != idk_spec["adapter_hash"]:
        raise ValueError("Archived IDK adapter hash differs from frozen config")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if (
        adapter_config.get("base_model_name_or_path") != full_spec["repo_id"]
        or adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("Archived IDK adapter config is incompatible with FULL")

    full_discovery, full_discovery_manifest = validate_score_state(
        name="FULL discovery",
        score_path=artifact_root / "scores" / "full_discovery.jsonl",
        expected_state="full",
        expected_subset="discovery",
        expected_model=full_spec,
        split_hash=split_hash,
    )
    discovery_ids = set(full_discovery)
    full_control, full_control_manifest = validate_score_state(
        name="FULL R_control",
        score_path=artifact_root / "scores" / "full_r_control.jsonl",
        expected_state="full",
        expected_subset="r_control",
        expected_model=full_spec,
        split_hash=split_hash,
    )
    control_ids = set(full_control)
    retain_discovery, retain_discovery_manifest = validate_score_state(
        name="RETAIN discovery",
        score_path=artifact_root / "scores" / "retain_discovery.jsonl",
        expected_state="retain",
        expected_subset="discovery",
        expected_model=retain_spec,
        split_hash=split_hash,
        expected_ids=discovery_ids,
    )
    retain_control, retain_control_manifest = validate_score_state(
        name="RETAIN R_control",
        score_path=artifact_root / "scores" / "retain_r_control.jsonl",
        expected_state="retain",
        expected_subset="r_control",
        expected_model=retain_spec,
        split_hash=split_hash,
        expected_ids=control_ids,
    )
    gd_root = artifact_root / "scores" / "gd_candidates"
    gd_discovery, gd_discovery_manifest = validate_score_state(
        name="GD02 discovery",
        score_path=gd_root / "gd_02_discovery.jsonl",
        expected_state="gd",
        expected_subset="discovery",
        expected_model=gd_spec,
        split_hash=split_hash,
        expected_ids=discovery_ids,
        candidate_id="gd_02",
    )
    gd_control, gd_control_manifest = validate_score_state(
        name="GD02 R_control",
        score_path=gd_root / "gd_02_r_control.jsonl",
        expected_state="gd",
        expected_subset="r_control",
        expected_model=gd_spec,
        split_hash=split_hash,
        expected_ids=control_ids,
        candidate_id="gd_02",
    )

    idk_discovery_path = project_path(idk_spec["discovery_scores"])
    idk_control_path = project_path(idk_spec["r_control_scores"])
    idk_discovery, idk_discovery_manifest = validate_score_state(
        name="IDK discovery",
        score_path=idk_discovery_path,
        expected_state="idk",
        expected_subset="discovery",
        expected_model=full_spec,
        split_hash=split_hash,
        expected_ids=discovery_ids,
        adapter=idk_spec,
    )
    idk_control, idk_control_manifest = validate_score_state(
        name="IDK R_control",
        score_path=idk_control_path,
        expected_state="idk",
        expected_subset="r_control",
        expected_model=full_spec,
        split_hash=split_hash,
        expected_ids=control_ids,
        adapter=idk_spec,
    )

    for name, rows in (
        ("RETAIN", retain_discovery),
        ("GD02", gd_discovery),
        ("IDK", idk_discovery),
    ):
        validate_pair_alignment("FULL", full_discovery, name, rows)
    for name, rows in (
        ("RETAIN", retain_control),
        ("GD02", gd_control),
        ("IDK", idk_control),
    ):
        validate_pair_alignment("FULL R_control", full_control, name, rows)

    archive_artifacts = project_path(
        "archive/idk_refusal_failed/data/artifacts"
    )
    archived_full_path = archive_artifacts / "reference_scores" / "full_discovery.jsonl"
    archived_full, archived_full_manifest = validate_score_state(
        name="archived FULL discovery",
        score_path=archived_full_path,
        expected_state="full",
        expected_subset="discovery",
        expected_model=full_spec,
        split_hash=split_hash,
        expected_ids=discovery_ids,
    )
    validate_pair_alignment("active FULL", full_discovery, "archived FULL", archived_full)
    active_archive_score_delta = max(
        abs(
            full_discovery[key]["mean_target_logprob"]
            - archived_full[key]["mean_target_logprob"]
        )
        for key in discovery_ids
    )
    if active_archive_score_delta != 0.0:
        raise ValueError("Active and archived FULL scores are not exactly identical")

    selection_path = project_path(idk_spec["selection_report"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    reversibility_path = project_path(idk_spec["reversibility_report"])
    reversibility = json.loads(reversibility_path.read_text(encoding="utf-8"))
    if (
        selection.get("selected_adapter_id") != idk_spec["adapter_id"]
        or selection.get("selected_adapter_hash") != adapter_hash
        or reversibility.get("selected_adapter_id") != idk_spec["adapter_id"]
        or reversibility.get("selected_adapter_hash") != adapter_hash
        or reversibility.get("status") != "complete"
        or reversibility.get("base_hash_match") is not True
    ):
        raise ValueError("Archived IDK selection/reversibility reports disagree")
    exact_reversibility_fields = (
        "adapter_on_max_abs_from_stored_idk",
        "off_before_max_abs_from_stored_full",
        "off_after_max_abs_from_stored_full",
        "off_before_after_max_abs",
    )
    if any(reversibility.get(field) != 0.0 for field in exact_reversibility_fields):
        raise ValueError("Archived IDK reversibility audit is not exact")

    full_activations = validate_activations(
        name="FULL",
        rows=full_discovery,
        activation_directory=artifact_root / "activations",
        expected_layers=layers,
        expected_hidden_size=hidden_size,
    )
    gd_activations = validate_activations(
        name="GD02",
        rows=gd_discovery,
        activation_directory=artifact_root / "activations" / "gd_candidates" / "gd_02",
        expected_layers=layers,
        expected_hidden_size=hidden_size,
    )
    idk_activations = validate_activations(
        name="IDK",
        rows=idk_discovery,
        activation_directory=project_path(idk_spec["activation_directory"]),
        expected_layers=layers,
        expected_hidden_size=hidden_size,
    )

    headrooms = {
        key: full_discovery[key]["mean_target_logprob"]
        - idk_discovery[key]["mean_target_logprob"]
        for key in sorted(discovery_ids)
    }
    author_headrooms: dict[str, list[float]] = defaultdict(list)
    for example_id, value in headrooms.items():
        author_headrooms[idk_discovery[example_id]["author_id"]].append(value)
    author_means = {
        author: float(np.mean(values))
        for author, values in sorted(author_headrooms.items())
    }
    if len(author_means) != 5 or any(value <= 0.02 for value in author_means.values()):
        raise ValueError("IDK author-level recovery denominators are not stable")
    if any(value <= 0.0 for value in headrooms.values()):
        raise ValueError("IDK does not have positive headroom on every discovery row")

    full_mean = float(np.mean([row["mean_target_logprob"] for row in full_discovery.values()]))
    idk_mean = float(np.mean([row["mean_target_logprob"] for row in idk_discovery.values()]))
    gd_mean = float(np.mean([row["mean_target_logprob"] for row in gd_discovery.values()]))
    retain_mean = float(
        np.mean([row["mean_target_logprob"] for row in retain_discovery.values()])
    )
    full_control_mean = float(
        np.mean([row["mean_target_logprob"] for row in full_control.values()])
    )
    idk_control_mean = float(
        np.mean([row["mean_target_logprob"] for row in idk_control.values()])
    )
    gd_control_mean = float(
        np.mean([row["mean_target_logprob"] for row in gd_control.values()])
    )

    manifests = {
        "full_discovery": full_discovery_manifest,
        "full_r_control": full_control_manifest,
        "retain_discovery": retain_discovery_manifest,
        "retain_r_control": retain_control_manifest,
        "gd02_discovery": gd_discovery_manifest,
        "gd02_r_control": gd_control_manifest,
        "idk_discovery": idk_discovery_manifest,
        "idk_r_control": idk_control_manifest,
        "archived_full_discovery": archived_full_manifest,
    }
    historical_reports = {
        "idk_refusal_failure": PROJECT_ROOT / "archive" / "idk_refusal_failed" / "gate_eval.json",
        "idk_suppression_failure": PROJECT_ROOT / "archive" / "idk_suppression_failed" / "gate_eval.json",
        "gd_candidate_screen": artifact_root / "results" / "gd_candidate_screen.json",
    }
    for name, path in historical_reports.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing historical negative report {name}: {path}")

    payload = {
        "schema_version": 1,
        "stage": "final_asset_freeze",
        "status": "COMPLETE",
        "scientific_gates": False,
        "prompt_contract": {
            "frozen_prompt_date": experiment_config["evaluation"]["prompt_date"],
            "discovery_prompt_hash_match_counts": {
                "idk_vs_active_full": 100,
                "archived_full_vs_active_full": 100,
                "gd02_vs_active_full": 100,
                "retain_vs_active_full": 100,
            },
            "active_evaluator_version": full_discovery_manifest["config"]["evaluator_version"],
            "archived_idk_evaluator_version": idk_discovery_manifest["config"]["evaluator_version"],
            "compatibility_basis": (
                "identical prompt hashes, token counts, split hashes, and unchanged "
                "teacher-forced scoring math; archived/active FULL scores match exactly"
            ),
            "active_archived_full_max_abs_score_delta": active_archive_score_delta,
            "prompt_hashes_hash": stable_hash(
                [full_discovery[key]["prompt_hash"] for key in sorted(discovery_ids)]
            ),
        },
        "split": {
            "path": portable_path(split_path),
            "split_hash": split_hash,
            "discovery_authors": frozen_split["discovery_authors"],
            "confirmation_authors": frozen_split["confirmation_authors"],
            "reserve_authors": frozen_split["reserve_authors"],
            "r_control_authors": frozen_split["r_control_authors"],
            "discovery_examples": len(discovery_ids),
            "r_control_examples": len(control_ids),
        },
        "architecture": {
            **models_config["expected_architecture"],
            "activation_dtype": experiment_config["evaluation"]["activation_dtype"],
            "token_position": "Q_END",
        },
        "states": {
            "full": {
                "model": full_spec,
                "discovery_scores": portable_path(artifact_root / "scores" / "full_discovery.jsonl"),
                "r_control_scores": portable_path(artifact_root / "scores" / "full_r_control.jsonl"),
                "discovery_mean": full_mean,
                "r_control_mean": full_control_mean,
                "activations": full_activations,
            },
            "idk": {
                "base_model": full_spec,
                "adapter_id": idk_spec["adapter_id"],
                "adapter_hash": adapter_hash,
                "adapter_path": portable_path(adapter_path),
                "adapter_config_sha256": file_sha256(adapter_config_path),
                "discovery_scores": portable_path(idk_discovery_path),
                "r_control_scores": portable_path(idk_control_path),
                "discovery_mean": idk_mean,
                "r_control_mean": idk_control_mean,
                "full_minus_idk": full_mean - idk_mean,
                "r_control_degradation": full_control_mean - idk_control_mean,
                "exactly_reversible": True,
                "reversibility_report": portable_path(reversibility_path),
                "selection_report": portable_path(selection_path),
                "activations": idk_activations,
            },
            "gd02": {
                "candidate_id": "gd_02",
                "model": gd_spec,
                "selection_scope": "exploratory_unmatched",
                "discovery_scores": portable_path(gd_root / "gd_02_discovery.jsonl"),
                "r_control_scores": portable_path(gd_root / "gd_02_r_control.jsonl"),
                "discovery_mean": gd_mean,
                "r_control_mean": gd_control_mean,
                "r_control_degradation": full_control_mean - gd_control_mean,
                "retain_distance": abs(gd_mean - retain_mean),
                "activations": gd_activations,
            },
            "retain": {
                "model": retain_spec,
                "discovery_scores": portable_path(artifact_root / "scores" / "retain_discovery.jsonl"),
                "r_control_scores": portable_path(artifact_root / "scores" / "retain_r_control.jsonl"),
                "discovery_mean": retain_mean,
                "activations": None,
            },
        },
        "idk_headroom": {
            "definition": "FULL_minus_IDK_nats_per_token",
            "distribution": distribution(list(headrooms.values())),
            "positive_examples": sum(value > 0 for value in headrooms.values()),
            "author_means": author_means,
            "unstable_author_threshold": 0.02,
            "unstable_authors": [
                author for author, value in author_means.items() if value <= 0.02
            ],
            "example_values_hash": stable_hash(headrooms),
        },
        "manifest_hashes": {
            name: stable_hash(manifest) for name, manifest in manifests.items()
        },
        "historical_negative_reports": {
            name: {
                "path": portable_path(path),
                "sha256": file_sha256(path),
            }
            for name, path in historical_reports.items()
        },
        "confirmation": assert_confirmation_unopened(artifact_root),
        "next_stage": "idk_only_layer_localization",
    }
    result = {**payload, "freeze_hash": stable_hash(payload)}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(
                f"Refusing to overwrite changed immutable final-state freeze: {output_path}"
            )
        return existing
    atomic_json(output_path, result)
    return result
