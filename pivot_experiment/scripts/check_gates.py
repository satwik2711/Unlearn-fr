#!/usr/bin/env python3
"""Read saved artifacts and check implemented scientific gates (P0–P2)."""

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
    DEFAULT_MODELS_CONFIG,
    load_yaml,
)
from pivot_experiment.gates import check_p0, check_p1, check_p2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("P0", "P1", "P2"), default="P1")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    config = load_yaml(args.config)
    models_config = load_yaml(args.models_config)
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
    if args.through in ("P1", "P2"):
        p1 = check_p1(config, models_config, args.artifacts, p0)
        results.append(p1)
        if p1["status"] == "BLOCKED":
            print(f"P1    BLOCKED  {p1['reason']}")
        elif p1["status"] == "PASS":
            metrics = p1["metrics"]
            qualification = p1.get("qualification")
            suffix = f" [{qualification}]" if qualification else ""
            print(
                f"P1    {p1['status']:<7}  GD={p1['selected_candidate_id']}{suffix}, "
                f"|GD-RETAIN|={metrics['behavior_distance']:.4f}, "
                f"R_loss={metrics['r_control_degradation']:.4f}"
            )
            print(json.dumps(p1["checks"], indent=2, sort_keys=True))
        else:
            print(f"P1    FAIL     {p1['reason']}")
            print("      Candidate  |GD-RETAIN|  R_control loss  Match  Utility")
            for candidate in p1["evaluated_candidates"]:
                checks = candidate["checks"]
                print(
                    f"      {candidate['candidate_id']:<9}  "
                    f"{candidate['behavior_distance']:>11.4f}  "
                    f"{candidate['r_control_degradation']:>14.4f}  "
                    f"{str(checks['behavior_matched']):<5}  "
                    f"{str(checks['r_control_preserved']):<7}"
                )
    if args.through == "P2":
        p2 = check_p2(config, models_config, args.artifacts, p1)
        results.append(p2)
        if p2["status"] == "BLOCKED":
            print(f"P2    BLOCKED  {p2['reason']}")
        elif p2["status"] == "PASS":
            metric = p2["selected_metrics"]
            print(
                f"P2    PASS     layer={p2['selected_layer']}, "
                f"D={metric['mean_differential_recovery']:+.4f}, "
                f"authors={metric['positive_authors']}/5"
            )
            print(json.dumps(p2["checks"], indent=2, sort_keys=True))
        else:
            metric = p2.get("selected_metrics", {})
            print(
                f"P2    FAIL     layer={p2.get('selected_layer')}, "
                f"D={metric.get('mean_differential_recovery', float('nan')):+.4f}"
            )
            print(json.dumps(p2.get("checks", {}), indent=2, sort_keys=True))
    raise SystemExit(0 if all(row["status"] == "PASS" for row in results) else 2)


if __name__ == "__main__":
    main()
