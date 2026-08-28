#!/usr/bin/env python3
"""Read saved artifacts and check implemented scientific gates (P0–P1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot_experiment.config import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    load_yaml,
)
from pivot_experiment.gates import check_p0, check_p1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("P0", "P1"), default="P1")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    config = load_yaml(args.config)
    p0 = check_p0(config, args.artifacts)
    results = [p0]
    print("Gate  Status   Primary result")
    if p0["status"] == "BLOCKED":
        print(f"P0    BLOCKED  {p0['reason']}")
    else:
        low, high = p0["author_clustered_ci_95"]
        print(
            f"P0    {p0['status']:<7}  Δ={p0['mean_difference']:+.4f}, "
            f"[{low:+.4f}, {high:+.4f}]"
        )
    if args.through == "P1":
        p1 = check_p1(config, args.artifacts, p0)
        results.append(p1)
        if p1["status"] == "BLOCKED":
            print(f"P1    BLOCKED  {p1['reason']}")
        else:
            metrics = p1["metrics"]
            print(
                f"P1    {p1['status']:<7}  IDK={p1['selected_adapter_id']}, "
                f"|IDK-RETAIN|={metrics['retain_distance']:.4f}"
            )
            print(json.dumps(p1["checks"], indent=2, sort_keys=True))
    raise SystemExit(0 if all(row["status"] == "PASS" for row in results) else 2)


if __name__ == "__main__":
    main()
