"""Append-safe JSONL records and completion manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .config import atomic_json


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_unique(path: Path, key: str = "example_id") -> dict[str, dict]:
    unique: dict[str, dict] = {}
    for row in read_jsonl(path):
        identifier = row.get(key)
        if not identifier:
            raise ValueError(f"Missing {key!r} in {path}")
        if identifier in unique and unique[identifier] != row:
            raise ValueError(f"Conflicting duplicate {key}={identifier!r} in {path}")
        unique[identifier] = row
    return unique


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def initialize_manifest(path: Path, config: dict, config_hash: str) -> dict:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            raise ValueError(f"Refusing to resume {path} with a changed configuration")
        return manifest
    manifest = {
        "schema_version": 1,
        "status": "running",
        "config": config,
        "config_hash": config_hash,
        "completed_rows": 0,
    }
    atomic_json(path, manifest)
    return manifest

