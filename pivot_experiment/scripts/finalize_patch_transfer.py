#!/usr/bin/env python3
"""Finalize the model-free Chunk 3 transfer comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    load_yaml,
)
from pivot_experiment.idk_localization import load_final_freeze  # noqa: E402
from pivot_experiment.patch_transfer import (  # noqa: E402
    finalize_patch_transfer,
    load_causal_layer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    experiment = load_yaml(args.config)
    final_states = load_final_freeze(args.artifacts / "freeze" / "final_states.json")
    causal_layer = load_causal_layer(
        args.artifacts / "freeze" / "causal_layer.json", final_states
    )
    result = finalize_patch_transfer(
        artifact_root=args.artifacts,
        final_states=final_states,
        causal_layer=causal_layer,
        layers=experiment["patching"]["layers"],
        bootstrap_resamples=experiment["evaluation"]["bootstrap_resamples"],
        seed=experiment["seed"],
        output_path=args.artifacts / "results" / "patch_transfer.json",
    )
    low, high = result["author_clustered_ci_95"]
    print(
        f"Chunk 3 complete: layer={result['selected_layer']}, "
        f"C_patch={result['c_patch']:+.4f}, CI=[{low:+.4f}, {high:+.4f}]"
    )


if __name__ == "__main__":
    main()
