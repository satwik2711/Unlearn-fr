#!/usr/bin/env python3
"""Prepare pinned TOFU data, frozen author splits, and run metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    atomic_json,
    load_yaml,
    stable_hash,
)
from pivot_experiment.data import prepare_tofu  # noqa: E402
from pivot_experiment.models import inspect_public_configs  # noqa: E402


def package_versions() -> dict[str, str]:
    names = [
        "accelerate",
        "datasets",
        "huggingface-hub",
        "numpy",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--verify-model-metadata",
        action="store_true",
        help="Download only configs/tokenizer metadata; never model weights.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    models_config = load_yaml(args.models_config)
    for directory in (
        "activations",
        "checkpoints",
        "figures",
        "gates",
        "logs",
        "results",
        "scores",
    ):
        (args.artifacts / directory).mkdir(parents=True, exist_ok=True)
    audit = prepare_tofu(config, args.artifacts)
    manifest_path = args.artifacts / "manifests" / "run_manifest.json"
    previous_metadata = None
    if manifest_path.exists():
        previous_metadata = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "model_metadata"
        )
    model_metadata = (
        inspect_public_configs(models_config)
        if args.verify_model_metadata
        else previous_metadata
    )
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "seed": config["seed"],
        "experiment_config": config,
        "experiment_config_hash": stable_hash(config),
        "models_config": models_config,
        "models_config_hash": stable_hash(models_config),
        "data_audit": audit,
        "model_metadata_verified": model_metadata is not None,
        "model_metadata": model_metadata,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    atomic_json(manifest_path, manifest)
    print(f"Prepared pinned TOFU data in {args.artifacts}")
    print(f"Frozen split hash: {audit['split_hash']}")
    if model_metadata is None:
        print("Model metadata verification skipped")
    else:
        print("FULL/RETAIN configs and shared tokenizer metadata verified")


if __name__ == "__main__":
    main()
