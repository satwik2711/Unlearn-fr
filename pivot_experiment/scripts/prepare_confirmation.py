#!/usr/bin/env python3
"""Freeze confirmation execution settings and five random directions."""

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
    load_confirmation_freeze,
    prepare_confirmation_protocol,
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
    direction, direction_manifest = load_direction(
        direction_path,
        direction_path.with_suffix(".manifest.json"),
        final_states,
        causal_layer,
    )
    random_manifest, execution = prepare_confirmation_protocol(
        confirmation_freeze=confirmation,
        direction=direction,
        direction_manifest=direction_manifest,
        generation_settings=experiment["confirmation"],
        random_tensor_path=args.artifacts / "directions" / "confirmation_random.safetensors",
        random_manifest_path=args.artifacts / "directions" / "confirmation_random.manifest.json",
        execution_freeze_path=args.artifacts / "freeze" / "confirmation_execution.json",
    )
    print(
        f"Confirmation execution frozen: randoms={len(random_manifest['directions'])}, "
        f"hash={execution['execution_freeze_hash']}"
    )


if __name__ == "__main__":
    main()
