"""Discovery-only IDK checkpoint selection and reversibility summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import atomic_json, stable_hash
from .records import read_unique


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_rows(path: Path) -> dict[str, dict]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing IDK evaluation artifact for {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete IDK evaluation: {path}")
    rows = read_unique(path)
    if len(rows) != manifest.get("expected_rows"):
        raise RuntimeError(f"IDK manifest/JSONL count mismatch: {path}")
    return rows


def _mean(rows: dict[str, dict], metric: str) -> float:
    return float(np.mean([row[metric] for row in rows.values()]))


def select_idk_checkpoint(
    experiment_config: dict,
    artifact_root: Path,
    candidates: list[dict],
) -> dict:
    full_discovery = _complete_rows(artifact_root / "scores" / "full_discovery.jsonl")
    retain_discovery = _complete_rows(artifact_root / "scores" / "retain_discovery.jsonl")
    full_control = _complete_rows(artifact_root / "scores" / "full_r_control.jsonl")
    full_mean = _mean(full_discovery, "mean_target_logprob")
    retain_mean = _mean(retain_discovery, "mean_target_logprob")
    full_control_mean = _mean(full_control, "mean_target_logprob")
    guardrail = experiment_config["gates"]["p1_max_r_control_degradation"]

    metrics = []
    for candidate in candidates:
        name = candidate["adapter_id"]
        discovery = _complete_rows(
            artifact_root / "scores" / "idk_candidates" / f"{name}_discovery.jsonl"
        )
        control = _complete_rows(
            artifact_root / "scores" / "idk_candidates" / f"{name}_r_control.jsonl"
        )
        if set(discovery) != set(full_discovery):
            raise ValueError(f"Discovery IDs differ for IDK candidate {name}")
        if set(control) != set(full_control):
            raise ValueError(f"R_control IDs differ for IDK candidate {name}")
        for example_id, row in discovery.items():
            reference = full_discovery[example_id]
            if (
                row.get("state") != "idk"
                or row.get("adapter_id") != name
                or row.get("adapter_hash") != candidate["adapter_hash"]
                or row.get("intervention") != "none"
                or row.get("author_id") != reference.get("author_id")
                or row.get("prompt_hash") != reference.get("prompt_hash")
            ):
                raise ValueError(f"Invalid discovery provenance for {name}/{example_id}")
        for example_id, row in control.items():
            reference = full_control[example_id]
            if (
                row.get("state") != "idk"
                or row.get("adapter_id") != name
                or row.get("adapter_hash") != candidate["adapter_hash"]
                or row.get("intervention") != "none"
                or row.get("author_id") != reference.get("author_id")
                or row.get("prompt_hash") != reference.get("prompt_hash")
            ):
                raise ValueError(f"Invalid R_control provenance for {name}/{example_id}")
        idk_mean = _mean(discovery, "mean_target_logprob")
        control_mean = _mean(control, "mean_target_logprob")
        degradation = full_control_mean - control_mean
        metrics.append(
            {
                **candidate,
                "mean_discovery_logprob": idk_mean,
                "mean_r_control_logprob": control_mean,
                "retain_distance": abs(idk_mean - retain_mean),
                "full_minus_idk": full_mean - idk_mean,
                "r_control_degradation": degradation,
                "utility_guardrail_pass": degradation <= guardrail,
            }
        )
    eligible = [row for row in metrics if row["utility_guardrail_pass"]]
    pool = eligible if eligible else metrics
    selected = min(
        pool,
        key=lambda row: (row["retain_distance"], row["r_control_degradation"], row["adapter_id"]),
    )
    selected_rows = _complete_rows(
        artifact_root
        / "scores"
        / "idk_candidates"
        / f"{selected['adapter_id']}_discovery.jsonl"
    )
    selected_activation_files = sorted(
        {row["activation_file"] for row in selected_rows.values() if row["activation_file"]}
    )
    if len(selected_activation_files) != 50:
        raise ValueError("Selected IDK candidate does not reference 50 activation shards")
    result = {
        "schema_version": 1,
        "selection_scope": "discovery_and_r_control_only",
        "full_discovery_mean": full_mean,
        "retain_discovery_mean": retain_mean,
        "full_r_control_mean": full_control_mean,
        "utility_guardrail": guardrail,
        "eligible_candidate_count": len(eligible),
        "selected_adapter_id": selected["adapter_id"],
        "selected_adapter_path": selected["adapter_path"],
        "selected_adapter_hash": selected["adapter_hash"],
        "selected_activation_files": selected_activation_files,
        "selected_metrics": selected,
        "candidates": metrics,
        "input_hash": stable_hash(
            {
                "full_discovery": full_discovery,
                "retain_discovery": retain_discovery,
                "full_control": full_control,
                "candidates": metrics,
            }
        ),
    }
    atomic_json(artifact_root / "results" / "idk_selection.json", result)
    return result
