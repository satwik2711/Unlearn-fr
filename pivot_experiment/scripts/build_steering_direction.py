#!/usr/bin/env python3
"""Build the frozen FULL-minus-IDK direction from existing caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import DEFAULT_ARTIFACT_ROOT  # noqa: E402
from pivot_experiment.data import load_prepared_rows  # noqa: E402
from pivot_experiment.idk_localization import load_final_freeze  # noqa: E402
from pivot_experiment.patch_transfer import load_causal_layer  # noqa: E402
from pivot_experiment.steering import build_direction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    final_states = load_final_freeze(args.artifacts / "freeze" / "final_states.json")
    causal_layer = load_causal_layer(
        args.artifacts / "freeze" / "causal_layer.json", final_states
    )
    rows = load_prepared_rows(args.artifacts, "discovery")
    result = build_direction(
        rows=rows,
        final_states=final_states,
        causal_layer=causal_layer,
        tensor_path=args.artifacts / "directions" / "full_minus_idk.safetensors",
        manifest_path=args.artifacts / "directions" / "full_minus_idk.manifest.json",
    )
    print(
        f"Frozen FULL-IDK direction: layer={result['layer']}, "
        f"norm={result['l2_norm']:.6f}, hash={result['direction_freeze_hash']}"
    )


if __name__ == "__main__":
    main()
