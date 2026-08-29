#!/usr/bin/env python3
"""Model-free completion audit for the final non-gating experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import DEFAULT_ARTIFACT_ROOT  # noqa: E402
from pivot_experiment.idk_localization import audit_chunk2, load_final_freeze  # noqa: E402
from pivot_experiment.patch_transfer import (  # noqa: E402
    audit_chunk3,
    load_causal_layer,
)
from pivot_experiment.steering import audit_chunk4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("A", "B", "C", "D"), default="D")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    final_states = load_final_freeze(args.artifacts / "freeze" / "final_states.json")
    print("Stage  Status      Result")
    print(
        "A      COMPLETE    "
        f"IDK={final_states['states']['idk']['adapter_id']}, "
        f"GD={final_states['states']['gd02']['candidate_id']}"
    )
    if args.through == "A":
        return
    stage = audit_chunk2(artifact_root=args.artifacts, final_states=final_states)
    if stage["status"] == "INCOMPLETE":
        print(f"B      INCOMPLETE  {stage['reason']}")
        raise SystemExit(2)
    low, high = stage["fractional_author_clustered_ci_95"]
    print(
        "B      COMPLETE    "
        f"layer={stage['selected_layer']}, "
        f"RF={stage['mean_fractional_recovery']:+.4f}, "
        f"raw={stage['mean_raw_recovery']:+.4f}, "
        f"authors={stage['positive_authors']}/5, "
        f"RF_CI=[{low:+.4f}, {high:+.4f}]"
    )
    print(f"Self-patch max |delta|={stage['max_abs_self_patch_effect']:.6g}")
    print(
        "Current/runtime-baseline max |delta|="
        f"{stage['max_abs_runtime_baseline_delta']:.6g}"
    )
    if not stage["live_self_patch_within_tolerance"]:
        print("Warning: live self-patch exceeded the diagnostic tolerance")
    print(
        "Archived/current activation drift (diagnostic): "
        f"{stage['max_abs_archived_activation_delta']:.6g}"
    )
    if args.through == "B":
        return
    causal_layer = load_causal_layer(
        args.artifacts / "freeze" / "causal_layer.json", final_states
    )
    transfer = audit_chunk3(
        artifact_root=args.artifacts,
        final_states=final_states,
        causal_layer=causal_layer,
    )
    if transfer["status"] == "INCOMPLETE":
        print(f"C      INCOMPLETE  {transfer['reason']}")
        raise SystemExit(2)
    low, high = transfer["author_clustered_ci_95"]
    gd = transfer["gd02_at_frozen_layer"]
    retain = transfer["retain_at_frozen_layer"]
    print(
        "C      COMPLETE    "
        f"layer={transfer['selected_layer']}, "
        f"C_patch={transfer['c_patch']:+.4f}, "
        f"GD_RF={gd['mean_fractional_recovery']:+.4f}, "
        f"RETAIN_RF={retain['mean_fractional_recovery']:+.4f}, "
        f"CI=[{low:+.4f}, {high:+.4f}]"
    )
    if args.through == "C":
        return
    steering = audit_chunk4(
        artifact_root=args.artifacts,
        final_states=final_states,
        causal_layer=causal_layer,
    )
    if steering["status"] == "INCOMPLETE":
        print(f"D      INCOMPLETE  {steering['reason']}")
        raise SystemExit(2)
    print(
        "D      COMPLETE    "
        f"alpha={steering['selected_alpha']:g}, "
        f"J={steering['selected_objective']:+.4f}, "
        f"IDK_RF={steering['selected_idk_rf']:+.4f}, "
        f"RETAIN_RF={steering['selected_retain_rf']:+.4f}, "
        f"utility={steering['selected_utility_penalty']:.4f}"
    )


if __name__ == "__main__":
    main()
