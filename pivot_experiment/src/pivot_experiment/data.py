"""Pinned TOFU loading, author attribution, and frozen splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

from .config import atomic_json, stable_hash
from .records import atomic_jsonl, read_jsonl


EXPECTED_COUNTS = {
    "full": 4000,
    "forget10": 400,
    "forget10_perturbed": 400,
    "retain90": 3600,
}
ROWS_PER_AUTHOR = 20


def _load(repo_id: str, config_name: str, revision: str) -> list[dict]:
    return list(
        load_dataset(repo_id, config_name, split="train", revision=revision)
    )


def _row_key(row: dict) -> tuple[str, str]:
    return row["question"], row["answer"]


def prepare_tofu(config: dict, artifact_root: Path) -> dict:
    dataset = config["dataset"]
    split_config = config["splits"]
    repo_id = dataset["repo_id"]
    revision = dataset["revision"]

    raw_full = _load(repo_id, dataset["full_config"], revision)
    raw_forget = _load(repo_id, dataset["forget_config"], revision)
    raw_perturbed = _load(repo_id, dataset["perturbed_config"], revision)
    raw_retain = _load(repo_id, dataset["retain_config"], revision)
    raw = {
        "full": raw_full,
        "forget10": raw_forget,
        "forget10_perturbed": raw_perturbed,
        "retain90": raw_retain,
    }
    counts = {name: len(rows) for name, rows in raw.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected pinned TOFU counts: {counts}")

    owner: dict[tuple[str, str], str] = {}
    for index, row in enumerate(raw_full):
        key = _row_key(row)
        if key in owner:
            raise ValueError("FULL contains duplicate question/answer pairs")
        owner[key] = f"tofu-a{index // ROWS_PER_AUTHOR:03d}"

    perturbed_by_key = {_row_key(row): row["perturbed_answer"] for row in raw_perturbed}
    if len(perturbed_by_key) != len(raw_perturbed):
        raise ValueError("forget10_perturbed contains duplicate source pairs")

    forget_rows: list[dict] = []
    for index, row in enumerate(raw_forget):
        key = _row_key(row)
        if key not in owner or key not in perturbed_by_key:
            raise ValueError(f"Could not map forget row {index} to FULL/perturbed data")
        forget_rows.append(
            {
                "example_id": f"forget10:{index:04d}",
                "author_id": owner[key],
                "question": row["question"],
                "answer": row["answer"],
                "perturbed_answers": perturbed_by_key[key],
            }
        )

    retain_rows: list[dict] = []
    for index, row in enumerate(raw_retain):
        key = _row_key(row)
        if key not in owner:
            raise ValueError(f"Could not map retain row {index} to FULL")
        retain_rows.append(
            {
                "example_id": f"retain90:{index:04d}",
                "author_id": owner[key],
                "question": row["question"],
                "answer": row["answer"],
                "perturbed_answers": [],
            }
        )

    forget_authors = sorted({row["author_id"] for row in forget_rows})
    retain_authors = sorted({row["author_id"] for row in retain_rows})
    if len(forget_authors) != 20 or len(retain_authors) != 180:
        raise ValueError("Unexpected author counts in pinned TOFU")
    if set(forget_authors) & set(retain_authors):
        raise ValueError("Forget and retain author sets overlap")
    if any(
        sum(row["author_id"] == author for row in forget_rows) != ROWS_PER_AUTHOR
        for author in forget_authors
    ):
        raise ValueError("A forget author does not have exactly 20 rows")

    shuffled_forget = list(forget_authors)
    random.Random(config["seed"]).shuffle(shuffled_forget)
    n_discovery = split_config["discovery_authors"]
    n_confirmation = split_config["confirmation_authors"]
    discovery = shuffled_forget[:n_discovery]
    confirmation = shuffled_forget[n_discovery : n_discovery + n_confirmation]
    reserve = shuffled_forget[n_discovery + n_confirmation :]
    patch = discovery[: split_config["patch_authors"]]
    if len(reserve) != split_config["reserve_authors"]:
        raise ValueError("Frozen forget partition sizes do not exhaust forget10")

    shuffled_retain = list(retain_authors)
    random.Random(config["seed"]).shuffle(shuffled_retain)
    r_control = shuffled_retain[: split_config["retain_control_authors"]]

    role_by_author = {
        **{author: "discovery" for author in discovery},
        **{author: "confirmation" for author in confirmation},
        **{author: "reserve" for author in reserve},
    }
    for row in forget_rows:
        row["partition"] = role_by_author[row["author_id"]]
        row["in_d_patch"] = row["author_id"] in patch
    for row in retain_rows:
        row["partition"] = (
            "r_control" if row["author_id"] in r_control else "retain_pool"
        )
        row["in_d_patch"] = False

    frozen_splits = {
        "schema_version": 1,
        "seed": config["seed"],
        "dataset_repo_id": repo_id,
        "dataset_revision": revision,
        "discovery_authors": discovery,
        "patch_authors": patch,
        "confirmation_authors": confirmation,
        "reserve_authors": reserve,
        "r_control_authors": r_control,
    }
    frozen_splits["split_hash"] = stable_hash(frozen_splits)

    data_dir = artifact_root / "data"
    splits_path = artifact_root / "splits" / "frozen_splits.json"
    if splits_path.exists():
        existing = json.loads(splits_path.read_text(encoding="utf-8"))
        if existing != frozen_splits:
            raise ValueError("Existing frozen split manifest differs; refusing overwrite")
    atomic_jsonl(data_dir / "forget10.jsonl", forget_rows)
    atomic_jsonl(data_dir / "retain90.jsonl", retain_rows)
    atomic_json(splits_path, frozen_splits)

    audit = {
        "schema_version": 1,
        "dataset_repo_id": repo_id,
        "dataset_revision": revision,
        "counts": counts,
        "author_counts": {"forget10": 20, "retain90": 180},
        "rows_per_author": ROWS_PER_AUTHOR,
        "forget_rows_hash": stable_hash(forget_rows),
        "retain_rows_hash": stable_hash(retain_rows),
        "split_hash": frozen_splits["split_hash"],
    }
    atomic_json(artifact_root / "manifests" / "data_audit.json", audit)
    return audit


def load_prepared_rows(artifact_root: Path, subset: str) -> list[dict]:
    if subset == "discovery":
        return [
            row
            for row in read_jsonl(artifact_root / "data" / "forget10.jsonl")
            if row["partition"] == "discovery"
        ]
    if subset == "r_control":
        return [
            row
            for row in read_jsonl(artifact_root / "data" / "retain90.jsonl")
            if row["partition"] == "r_control"
        ]
    raise ValueError(f"Unknown prepared subset: {subset}")
