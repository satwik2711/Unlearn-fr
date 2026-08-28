"""Derive tidy scalar metrics from canonical JSONL evaluation records.

This module never loads model weights or replaces raw records. It builds the
plan's reproducible ``metrics_long.csv`` input for statistics and figures.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

IDENTIFIERS = ("state", "split", "author_id", "row_id")
EXCLUDED = {
    "schema_version",
    "question",
    "reference",
    "generation",
    "device",
    "config_hash",
    "activation_file",
    "activation_key",
    "activation_shape",
    "activation_dtype",
    "answer_token_logprobs",
    "perturbed_answer_logprobs",
}


def read_unique(path: Path) -> list[dict]:
    records = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                row_id = record["row_id"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if row_id in records and records[row_id] != record:
                raise ValueError(f"Conflicting duplicate row_id {row_id} in {path}")
            records[row_id] = record
    return list(records.values())


def tidy(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for record in read_unique(path):
            split = record.get("split")
            if not split:
                name = path.stem
                split = (
                    "full"
                    if name.endswith("_teacher")
                    else (
                        "forget10"
                        if "forget" in name
                        else name.removeprefix(f"{record.get('state', '').lower()}_")
                    )
                )
            identifiers = {
                **{key: record.get(key, "") for key in IDENTIFIERS},
                "split": split,
            }
            for metric, value in record.items():
                if metric in IDENTIFIERS or metric in EXCLUDED:
                    continue
                if isinstance(value, (bool, int, float)) and value is not None:
                    rows.append({**identifiers, "metric": metric, "value": value})
    return pd.DataFrame(rows, columns=[*IDENTIFIERS, "metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = tidy(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} metric rows to {args.out}")


if __name__ == "__main__":
    main()
