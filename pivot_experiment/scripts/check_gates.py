#!/usr/bin/env python3
"""Read saved artifacts and check implemented scientific gates (currently P0)."""

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
from pivot_experiment.gates import check_p0  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("P0",), default="P0")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    result = check_p0(load_yaml(args.config), args.artifacts)
    print("Gate  Status   Mean Δ    95% author CI")
    if result["status"] == "BLOCKED":
        print(f"P0    BLOCKED  --        --\n{result['reason']}")
    else:
        low, high = result["author_clustered_ci_95"]
        print(
            f"P0    {result['status']:<7}  {result['mean_difference']:+.4f}   "
            f"[{low:+.4f}, {high:+.4f}]"
        )
        print(json.dumps(result["checks"], indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()

