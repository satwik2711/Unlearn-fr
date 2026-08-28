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
