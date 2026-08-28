"""Frozen TOFU-Alias construction and audit."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

REV = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
SPLITS = [
    "full",
    "forget10",
    "retain90",
    "forget10_perturbed",
    "world_facts",
    "real_authors",
    "holdout10",
]


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atom(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    t.replace(p)


def name_from_group(rows, index):
    # The first record was authored as the identity question. Its answer is the
    # least ambiguous source even for names with accents/particles.
    first = rows[0]["answer"]
    m = re.search(
        r"(?:name(?: of (?:this |the )?(?:author|[^.]*?author))? is|author(?: in question)? is|referring to is) ([A-ZÀ-Ý][\wÀ-ÿ-]*(?: (?:[A-ZÀ-Ý][\wÀ-ÿ-]*|van|de|von)){1,3})",
        first,
    )
    if m:
        return m.group(1)
    m = re.match(
        r"([A-ZÀ-Ý][\wÀ-ÿ-]*(?: (?:[A-ZÀ-Ý][\wÀ-ÿ-]*|van|de|von)){1,3}) was ", first
    )
    if m:
        return m.group(1)
    text = " ".join(r["question"] + " " + r["answer"] for r in rows)
    candidates = re.findall(r"\b[A-Z][A-Za-z\-]+ [A-Z][A-Za-z\-]+\b", text)
    counts = Counter(candidates)
    name, n = counts.most_common(1)[0]
    if n < 8:
        return f"__TOFU_UNNAMED_{index:03d}__"
    return name


def nonce(i):
    a = [
        "Zor",
        "Vel",
        "Nim",
        "Kav",
        "Ryn",
        "Tal",
        "Myr",
        "Sov",
        "Dra",
        "Pex",
        "Lun",
        "Bex",
        "Cor",
        "Fyn",
        "Gav",
        "Hex",
        "Jor",
        "Kyr",
        "Vex",
        "Wyn",
    ]
    b = ["aven", "orin", "elis", "umar", "essa", "ivar", "olen", "ara", "eth", "yra"]
    c = ["Seln", "Varo", "Mire", "Dax", "Keli", "Rova", "Tenn", "Vira", "Nox", "Pali"]
    return f"{a[i%len(a)]}{b[(i//len(a))%len(b)]} {c[(i//(len(a)*len(b)))%len(c)]}"


def transform(row, mapping):
    out = dict(row)
    for k, v in out.items():
        if isinstance(v, str):
            for raw, alias in mapping.items():
                v = v.replace(raw, alias)
            out[k] = v
    return out


def main(out):
    raw = {
        s: list(load_dataset("locuslab/TOFU", s, revision=REV, split="train"))
        for s in SPLITS
    }
    full = raw["full"]
    assert len(full) == 4000
    authors = [
        name_from_group(full[i : i + 20], i // 20) for i in range(0, len(full), 20)
    ]
    assert len(authors) == 200 and len(set(authors)) == 200
    mapping = {x: nonce(i) for i, x in enumerate(authors)}
    raw_text = " ".join(str(r) for rows in raw.values() for r in rows)
    assert not any(v in raw_text for v in mapping.values())
    transformed = {s: [transform(r, mapping) for r in rows] for s, rows in raw.items()}
    full_owner = {
        (r["question"], r["answer"]): f"a{i//20:03d}" for i, r in enumerate(raw["full"])
    }
    # Official author ordering is contiguous 20-item blocks. Preserve it in all primary splits.
    for s in ["full", "forget10", "retain90", "forget10_perturbed"]:
        for i, r in enumerate(transformed[s]):
            r["author_id"] = f"a{(i//20):03d}" if s == "full" else None
            r["row_id"] = hashlib.sha256(
                (s + "\0" + str(i) + "\0" + r["question"] + "\0" + r["answer"]).encode()
            ).hexdigest()
    # Match official split rows to their full-row owner before aliases obscure strings.
    for s in ["forget10", "retain90", "forget10_perturbed"]:
        for raw_r, r in zip(raw[s], transformed[s]):
            r["author_id"] = full_owner.get(
                (raw_r["question"], raw_r["answer"]), "unresolved"
            )
        if any(r["author_id"] == "unresolved" for r in transformed[s]):
            raise ValueError(f"unresolved authors in {s}")
    forget_auth = sorted(set(r["author_id"] for r in transformed["forget10"]))
    retain_auth = sorted(set(r["author_id"] for r in transformed["retain90"]))
    assert len(forget_auth) == 20 and not (set(forget_auth) & set(retain_auth))
    rng = random.Random(42)
    rng.shuffle(forget_auth)
    discovery = sorted(forget_auth[:10])
    confirmation = sorted(forget_auth[10:])
    rcontrol_auth = sorted(retain_auth)[:10]
    frozen = {
        "seed": 42,
        "forget_discovery_authors": discovery,
        "forget_confirmation_authors": confirmation,
        "retain_control_authors": rcontrol_auth,
        "dataset_revision": REV,
    }
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    atom(
        root / "alias_map.json",
        {"seed": 42, "mapping": mapping, "sha256": digest(mapping)},
    )
    for s, rows in transformed.items():
        p = root / f"{s}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    atom(root / "frozen_splits.json", frozen)
    audit = {
        "dataset_revision": REV,
        "row_counts": {s: len(x) for s, x in raw.items()},
        "fields": {s: list(raw[s][0]) for s in raw},
        "authors": {
            "full": len(authors),
            "forget10": len(forget_auth),
            "retain90": len(retain_auth),
        },
        "overlap_forget_retain": sorted(set(forget_auth) & set(retain_auth)),
        "duplicate_questions": {
            s: len(x) - len(set(r["question"] for r in x))
            for s, x in transformed.items()
        },
        "duplicate_answers": {
            s: len(x) - len(set(r["answer"] for r in x)) for s, x in transformed.items()
        },
        "alias_map_sha256": digest(mapping),
        "transformed_dataset_sha256": digest(transformed),
        "notes": "Author grouping follows official contiguous 20-row TOFU records; identity replacement is global.",
    }
    atom(root / "data_audit.json", audit)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiment/artifacts/data")
    a = p.parse_args()
    main(a.out)
