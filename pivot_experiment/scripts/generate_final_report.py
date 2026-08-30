#!/usr/bin/env python3
"""Generate all model-free Chunk 6 tables, figures, and final report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.analysis import generate_final_analysis  # noqa: E402
from pivot_experiment.config import DEFAULT_ARTIFACT_ROOT  # noqa: E402
from pivot_experiment.confirmation import (  # noqa: E402
    audit_chunk5,
    load_confirmation_freeze,
    load_confirmation_protocol,
)
from pivot_experiment.idk_localization import (  # noqa: E402
    audit_chunk2,
    load_final_freeze,
)
from pivot_experiment.patch_transfer import (  # noqa: E402
    audit_chunk3,
    load_causal_layer,
)
from pivot_experiment.steering import (  # noqa: E402
    audit_chunk4,
    load_direction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    final_states = load_final_freeze(args.artifacts / "freeze" / "final_states.json")
    causal = load_causal_layer(
        args.artifacts / "freeze" / "causal_layer.json", final_states
    )
    if audit_chunk2(artifact_root=args.artifacts, final_states=final_states)["status"] != "COMPLETE":
        raise ValueError("Chunk 2 integrity audit is incomplete")
    patch = audit_chunk3(
        artifact_root=args.artifacts,
        final_states=final_states,
        causal_layer=causal,
    )
    if patch["status"] != "COMPLETE":
        raise ValueError("Chunk 3 integrity audit is incomplete")
    steering = audit_chunk4(
        artifact_root=args.artifacts,
        final_states=final_states,
        causal_layer=causal,
    )
    if steering["status"] != "COMPLETE":
        raise ValueError("Chunk 4 integrity audit is incomplete")
    confirmation_freeze = load_confirmation_freeze(
        args.artifacts / "freeze" / "confirmation.json"
    )
    direction_path = args.artifacts / "directions" / "full_minus_idk.safetensors"
    _, direction = load_direction(
        direction_path,
        direction_path.with_suffix(".manifest.json"),
        final_states,
        causal,
    )
    _, _, execution = load_confirmation_protocol(
        confirmation_freeze=confirmation_freeze,
        direction_manifest=direction,
        random_tensor_path=args.artifacts / "directions" / "confirmation_random.safetensors",
        random_manifest_path=args.artifacts / "directions" / "confirmation_random.manifest.json",
        execution_freeze_path=args.artifacts / "freeze" / "confirmation_execution.json",
    )
    confirmation = audit_chunk5(
        artifact_root=args.artifacts,
        final_states=final_states,
        confirmation_freeze=confirmation_freeze,
        execution=execution,
    )
    if confirmation["status"] != "COMPLETE":
        raise ValueError("Chunk 5 integrity audit is incomplete")
    alpha = json.loads(
        (args.artifacts / "results" / "alpha_selection.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = generate_final_analysis(
        artifact_root=args.artifacts,
        output_root=args.artifacts / "analysis",
        final_states=final_states,
        causal=causal,
        patch=patch,
        alpha=alpha,
        confirmation=confirmation,
        direction=direction,
    )
    print(
        "Chunk 6 complete: "
        f"report={args.artifacts / 'analysis' / 'report.md'}, "
        f"hash={manifest['analysis_manifest_hash']}"
    )


if __name__ == "__main__":
    main()
