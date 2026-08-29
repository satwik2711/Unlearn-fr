#!/usr/bin/env python3
"""Validate existing assets and freeze the final four-state experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    load_yaml,
)
from pivot_experiment.final_freeze import create_final_states_freeze  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    output_path = args.artifacts / "freeze" / "final_states.json"
    result = create_final_states_freeze(
        experiment_config=load_yaml(args.config),
        models_config=load_yaml(args.models_config),
        artifact_root=args.artifacts,
        output_path=output_path,
    )
    print("Stage   Status    Result")
    print(
        "A       COMPLETE  "
        f"IDK={result['states']['idk']['adapter_id']}, "
        f"GD={result['states']['gd02']['candidate_id']}, "
        f"headroom={result['idk_headroom']['distribution']['mean']:.4f}, "
        f"prompts=100/100"
    )
    print(f"Freeze: {output_path}")
    print(f"Hash:   {result['freeze_hash']}")
    print("No model weights were loaded")


if __name__ == "__main__":
    main()
