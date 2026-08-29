#!/usr/bin/env python3
"""Select alpha and seal confirmation without loading a model."""

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
from pivot_experiment.patch_transfer import load_causal_layer  # noqa: E402
from pivot_experiment.steering import finalize_alpha_selection, load_direction  # noqa: E402


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
    tensor_path = args.artifacts / "directions" / "full_minus_idk.safetensors"
    _, direction = load_direction(
        tensor_path,
        tensor_path.with_suffix(".manifest.json"),
        final_states,
        causal_layer,
    )
    analysis, confirmation = finalize_alpha_selection(
        artifact_root=args.artifacts,
        final_states=final_states,
        causal_layer=causal_layer,
        direction_manifest=direction,
        alphas=[float(value) for value in experiment["steering"]["alphas"]],
        beta=float(experiment["steering"]["beta"]),
        random_seeds=experiment["steering"]["random_seeds"],
        bootstrap_resamples=experiment["evaluation"]["bootstrap_resamples"],
        analysis_output=args.artifacts / "results" / "alpha_selection.json",
        confirmation_output=args.artifacts / "freeze" / "confirmation.json",
    )
    selected = analysis["selected_row"]
    print(
        f"Chunk 4 complete: alpha={analysis['selected_alpha']:g}, "
        f"J={selected['objective_j']:+.4f}, "
        f"confirmation={confirmation['confirmation_freeze_hash']}"
    )


if __name__ == "__main__":
    main()
