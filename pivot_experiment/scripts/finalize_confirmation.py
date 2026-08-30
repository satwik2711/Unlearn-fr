#!/usr/bin/env python3
"""Finalize held-out endpoints from stored confirmation artifacts."""

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
from pivot_experiment.confirmation import (  # noqa: E402
    finalize_confirmation,
    load_confirmation_freeze,
    load_confirmation_protocol,
)
from pivot_experiment.idk_localization import load_final_freeze  # noqa: E402
from pivot_experiment.patch_transfer import load_causal_layer  # noqa: E402
from pivot_experiment.steering import load_direction  # noqa: E402


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
    confirmation = load_confirmation_freeze(
        args.artifacts / "freeze" / "confirmation.json"
    )
    direction_path = args.artifacts / "directions" / "full_minus_idk.safetensors"
    _, direction_manifest = load_direction(
        direction_path,
        direction_path.with_suffix(".manifest.json"),
        final_states,
        causal_layer,
    )
    _, random_manifest, execution = load_confirmation_protocol(
        confirmation_freeze=confirmation,
        direction_manifest=direction_manifest,
        random_tensor_path=args.artifacts / "directions" / "confirmation_random.safetensors",
        random_manifest_path=args.artifacts / "directions" / "confirmation_random.manifest.json",
        execution_freeze_path=args.artifacts / "freeze" / "confirmation_execution.json",
    )
    result = finalize_confirmation(
        artifact_root=args.artifacts,
        final_states=final_states,
        confirmation_freeze=confirmation,
        execution=execution,
        random_manifest=random_manifest,
        bootstrap_resamples=experiment["evaluation"]["bootstrap_resamples"],
        seed=experiment["seed"],
        output_path=args.artifacts / "results" / "confirmation.json",
    )
    print(
        f"Chunk 5 complete: C_IDK={result['c_idk']:+.4f} "
        f"(rank {result['idk_learned_rank_among_six']}/6), "
        f"C_transfer={result['c_transfer']:+.4f} "
        f"(rank {result['transfer_learned_rank_among_six']}/6)"
    )


if __name__ == "__main__":
    main()
