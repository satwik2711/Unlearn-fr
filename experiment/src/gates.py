"""Single entry point for running, resuming, and assessing Gates 1 and 2.

Raw evidence is append-only JSONL. Dense float32 Q_END activations are stored
in safetensors sidecars referenced by the JSONL records. Model work runs in
bounded subprocesses so MPS memory is returned to macOS between chunks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from rouge_score import rouge_scorer
from safetensors.torch import save_file

from common import chat_ids, loader, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiment/artifacts/data"
OUTPUT = ROOT / "experiment/artifacts/state_eval"
MODEL = ROOT / "models/Qwen3.5-2B"
METRICS = ("correct", "perturbed", "generate", "activations")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def load_records(path: Path) -> dict[str, dict]:
    """Load unique records and reject malformed/conflicting append history."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
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
    return records


def evaluation_rows(path: str) -> list[dict]:
    """Supply stable IDs for utility splits whose source rows lack metadata."""
    rows = read_jsonl(path)
    split = Path(path).stem
    for index, source in enumerate(rows):
        if "row_id" not in source:
            row = dict(source)
            identity = f"{split}\0{index}\0{row['question']}\0{row['answer']}"
            row["row_id"] = hashlib.sha256(identity.encode()).hexdigest()
            row["author_id"] = f"utility:{split}"
            rows[index] = row
    return rows


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    raise ValueError("Question token sequence not found in formatted example")


def teacher_forced(model, tokenizer, row: dict, capture: bool):
    input_ids, labels = chat_ids(tokenizer, row["question"], row["answer"])
    device = next(model.parameters()).device
    input_ids, labels = input_ids.to(device), labels.to(device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            output_hidden_states=capture,
            use_cache=False,
        )
    targets = labels[0, 1:]
    positions = targets.ne(-100).nonzero(as_tuple=False).squeeze(-1)
    target_ids = targets[positions]
    logits = output.logits[0, positions].float()
    token_logprobs = (
        functional.log_softmax(logits, dim=-1)
        .gather(-1, target_ids.unsqueeze(-1))
        .squeeze(-1)
        .cpu()
    )
    if not token_logprobs.numel():
        raise ValueError(f"No answer tokens for {row['row_id']}")
    activations = None
    q_end = None
    if capture:
        question_tokens = tokenizer.encode(row["question"], add_special_tokens=False)
        q_end = find_subsequence(input_ids[0].tolist(), question_tokens)
        q_end += len(question_tokens) - 1
        activations = torch.stack(
            [
                hidden[0, q_end].detach().float().cpu()
                for hidden in output.hidden_states[1:]
            ]
        )
    result = {
        "answer_logprob": float(token_logprobs.mean()),
        "answer_token_count": int(token_logprobs.numel()),
        "answer_token_logprobs": [float(value) for value in token_logprobs],
        "q_end_token_index": q_end,
    }
    return result, activations


def score_answer(model, tokenizer, question: str, answer: str) -> float:
    input_ids, labels = chat_ids(tokenizer, question, answer)
    device = next(model.parameters()).device
    with torch.inference_mode():
        output = model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            use_cache=False,
        )
    return -float(output.loss)


def generate(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]
    input_ids = torch.tensor(prompt).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def worker_config(args: argparse.Namespace) -> dict:
    # Keep this schema stable: existing manifests use this exact hash contract.
    return {
        "state": args.state,
        "rows": str(Path(args.rows).resolve()),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "metrics": tuple(sorted(set(args.metric))),
        "perturbed_rows": (
            str(Path(args.perturbed_rows).resolve()) if args.perturbed_rows else None
        ),
        "activation_authors": sorted(args.activation_author),
        "generation_authors": sorted(args.generate_author),
        "selected_authors": sorted(args.author_id),
        "max_new_tokens": args.max_new_tokens,
        "model": str(Path(args.model).resolve()),
    }


def evaluate_chunk(args: argparse.Namespace) -> None:
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    config = worker_config(args)
    config_hash = digest(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_hash"] != config_hash:
            raise ValueError(f"Refusing to resume {output} with changed settings")
    else:
        manifest = {
            "config": config,
            "config_hash": config_hash,
            "status": "running",
            "created_at_unix": time.time(),
        }
        atomic_json(manifest_path, manifest)

    rows = evaluation_rows(args.rows)
    if args.author_id:
        authors = set(args.author_id)
        rows = [row for row in rows if row.get("author_id") in authors]
    completed = load_records(output)
    pending = [row for row in rows if row["row_id"] not in completed]
    pending = pending[: args.max_new_items]
    if not pending:
        manifest.update({"status": "complete", "completed_rows": len(completed)})
        atomic_json(manifest_path, manifest)
        return

    perturbations = {}
    if args.perturbed_rows:
        perturbations = {
            row["question"]: row["perturbed_answer"]
            for row in read_jsonl(args.perturbed_rows)
        }
    model, tokenizer, device = loader(path=args.model, adapter_path=args.adapter)
    model.eval()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics = set(args.metric)
    capture_authors = set(args.activation_author)
    generation_authors = set(args.generate_author)
    activation_tensors = {}
    records = []
    for offset, row in enumerate(pending, start=1):
        capture = "activations" in metrics and row.get("author_id") in capture_authors
        record = {
            "schema_version": 1,
            "state": args.state,
            "split": Path(args.rows).stem,
            "author_id": row.get("author_id", "utility"),
            "row_id": row["row_id"],
            "question": row["question"],
            "reference": row["answer"],
            "device": device,
            "config_hash": config_hash,
        }
        if "correct" in metrics or capture:
            correct, activations = teacher_forced(model, tokenizer, row, capture)
            record.update(correct)
            if activations is not None:
                activation_tensors[row["row_id"]] = activations
        if "perturbed" in metrics and row["question"] in perturbations:
            values = [
                score_answer(model, tokenizer, row["question"], answer)
                for answer in perturbations[row["question"]]
            ]
            correct = record.get("answer_logprob")
            if correct is None:
                correct = score_answer(model, tokenizer, row["question"], row["answer"])
            perturbed_mean = sum(values) / len(values)
            record.update(
                perturbed_answer_logprobs=values,
                mean_perturbed_logprob=perturbed_mean,
                correct_perturbed_margin=correct - perturbed_mean,
            )
        should_generate = "generate" in metrics and (
            not generation_authors or row.get("author_id") in generation_authors
        )
        if should_generate:
            generated = generate(model, tokenizer, row["question"], args.max_new_tokens)
            record.update(
                generation=generated,
                rougeL=rouge.score(row["answer"], generated)["rougeL"].fmeasure,
                exact_match=int(normalize(row["answer"]) == normalize(generated)),
            )
        records.append(record)
        release_memory()
        print(
            f"{args.state}: chunk {offset}/{len(pending)}; "
            f"total {len(completed) + offset}/{len(rows)}",
            flush=True,
        )

    activation_path = None
    if activation_tensors:
        activation_dir = output.parent / "activations"
        activation_dir.mkdir(parents=True, exist_ok=True)
        chunk_id = min(record["row_id"] for record in records)[:12]
        activation_path = (
            activation_dir / f"{args.state.lower()}-{chunk_id}.safetensors"
        )
        temporary = activation_path.with_suffix(".safetensors.tmp")
        save_file(activation_tensors, temporary)
        temporary.replace(activation_path)
        for record in records:
            row_id = record["row_id"]
            if row_id in activation_tensors:
                record.update(
                    activation_file=str(activation_path),
                    activation_key=row_id,
                    activation_shape=list(activation_tensors[row_id].shape),
                    activation_dtype="float32",
                )
    with output.open("a") as handle:
        handle.write(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        )
        handle.flush()
    manifest.update(
        status="running",
        completed_rows=len(completed) + len(records),
        last_chunk_activation_file=str(activation_path) if activation_path else None,
        updated_at_unix=time.time(),
    )
    atomic_json(manifest_path, manifest)


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", required=True)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--adapter")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--perturbed-rows")
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", action="append", choices=METRICS, required=True)
    parser.add_argument("--author-id", action="append", default=[])
    parser.add_argument("--activation-author", action="append", default=[])
    parser.add_argument("--generate-author", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-new-items", type=int, default=20)


def repeated(flag: str, values: list[str]) -> list[str]:
    return [item for value in values for item in (flag, value)]


def jobs() -> dict[str, list[str]]:
    frozen = json.loads((DATA / "frozen_splits.json").read_text())
    discovery = frozen["forget_discovery_authors"]
    forget_authors = sorted(
        {row["author_id"] for row in read_jsonl(DATA / "forget10.jsonl")}
    )
    common = [
        "--rows",
        str(DATA / "full.jsonl"),
        "--perturbed-rows",
        str(DATA / "forget10_perturbed.jsonl"),
        "--metric",
        "correct",
        "--metric",
        "perturbed",
        "--metric",
        "generate",
        "--max-new-items",
        "20",
    ]
    result = {
        "base": [
            "--state",
            "BASE",
            *common,
            *repeated("--generate-author", forget_authors),
            "--out",
            str(OUTPUT / "base.jsonl"),
        ],
        "full": [
            "--state",
            "FULL",
            "--adapter",
            str(ROOT / "experiment/checkpoints/full/epoch-1"),
            *common,
            "--metric",
            "activations",
            *repeated("--activation-author", discovery),
            "--out",
            str(OUTPUT / "full.jsonl"),
        ],
        "retain": [
            "--state",
            "RETAIN",
            "--adapter",
            str(ROOT / "experiment/checkpoints/retain/epoch-1"),
            *common,
            *repeated("--generate-author", forget_authors),
            "--out",
            str(OUTPUT / "retain.jsonl"),
        ],
    }
    adapters = {
        "base": None,
        "full": ROOT / "experiment/checkpoints/full/epoch-1",
        "retain": ROOT / "experiment/checkpoints/retain/epoch-1",
    }
    for state, adapter in adapters.items():
        for split in ("real_authors", "world_facts"):
            arguments = [
                "--state",
                state.upper(),
                "--rows",
                str(DATA / f"{split}.jsonl"),
                "--metric",
                "correct",
                "--max-new-items",
                "20",
                "--out",
                str(OUTPUT / f"{state}_{split}.jsonl"),
            ]
            if adapter:
                arguments[2:2] = ["--adapter", str(adapter)]
            result[f"{state}_{split}"] = arguments
    return result


def run_state(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    add_worker_arguments(parser)
    args = parser.parse_args(arguments)
    rows = evaluation_rows(args.rows)
    if args.author_id:
        authors = set(args.author_id)
        rows = [row for row in rows if row.get("author_id") in authors]
    target = len(rows)
    while len(load_records(Path(args.out))) < target:
        complete = len(load_records(Path(args.out)))
        print(f"Launching chunk: {complete}/{target} complete", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "_chunk", *arguments],
            cwd=ROOT,
            check=True,
        )
    manifest_path = Path(args.out).with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="complete", completed_rows=target)
    atomic_json(manifest_path, manifest)
    print(f"Evaluation complete: {target}/{target}", flush=True)


def run_pipeline(selected: list[str]) -> None:
    available = jobs()
    selected = selected or list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown jobs: {unknown}; choices: {sorted(available)}")
    for name in selected:
        print(f"\nStarting {name}", flush=True)
        run_state(available[name])


def as_frame(path: Path) -> pd.DataFrame:
    return pd.DataFrame(load_records(path).values())


def clustered_ci(frame: pd.DataFrame, column: str) -> list[float]:
    values = frame.groupby("author_id")[column].mean().to_numpy()
    rng = np.random.default_rng(42)
    boot = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def assess(root: Path) -> dict:
    data = root / "experiment/artifacts/data"
    state = root / "experiment/artifacts/state_eval"
    frozen = json.loads((data / "frozen_splits.json").read_text())
    forget_authors = {row["author_id"] for row in read_jsonl(data / "forget10.jsonl")}
    rcontrol = set(frozen["retain_control_authors"])
    base = as_frame(state / "base.jsonl")
    full = as_frame(state / "full.jsonl")
    retain = as_frame(state / "retain.jsonl")
    utility_frames = {
        (model_state, split): as_frame(state / f"{model_state}_{split}.jsonl")
        for model_state in ("base", "full", "retain")
        for split in ("real_authors", "world_facts")
    }
    required = len(read_jsonl(data / "full.jsonl"))
    teacher_complete = all(len(frame) == required for frame in (base, full, retain))
    gate1 = {"status": "pending", "reason": "incomplete teacher-forced artifacts"}
    gate2 = {"status": "pending", "reason": "incomplete teacher-forced artifacts"}
    if teacher_complete:
        paired = full[["row_id", "author_id", "answer_logprob"]].merge(
            base[["row_id", "answer_logprob"]],
            on="row_id",
            suffixes=("_full", "_base"),
            validate="one_to_one",
        )
        paired["delta"] = paired.answer_logprob_full - paired.answer_logprob_base
        forget_paired = paired[paired.author_id.isin(forget_authors)]
        retain_paired = paired[~paired.author_id.isin(forget_authors)]
        forget_gain = float(forget_paired.delta.mean())
        retain_gain = float(retain_paired.delta.mean())
        forget_gain_ci = clustered_ci(forget_paired, "delta")
        retain_gain_ci = clustered_ci(retain_paired, "delta")
        full_generated = (
            full[full.generation.notna()] if "generation" in full else pd.DataFrame()
        )
        base_generated = (
            base[base.generation.notna()] if "generation" in base else pd.DataFrame()
        )
        generation_complete = len(full_generated) == required
        forget_rouge = retain_rouge = None
        if generation_complete:
            forget_rouge = float(
                full_generated[
                    full_generated.author_id.isin(forget_authors)
                ].rougeL.mean()
            )
            retain_rouge = float(
                full_generated[
                    ~full_generated.author_id.isin(forget_authors)
                ].rougeL.mean()
            )
        status = "pending"
        if generation_complete:
            status = (
                "passed"
                if forget_gain_ci[0] > 0
                and retain_gain_ci[0] > 0
                and forget_rouge >= 0.70
                and retain_rouge >= 0.70
                else "needs_more_training"
            )
        gate1 = {
            "status": status,
            "forget_logprob_gain": forget_gain,
            "forget_logprob_gain_author_ci95": forget_gain_ci,
            "retain_logprob_gain": retain_gain,
            "retain_logprob_gain_author_ci95": retain_gain_ci,
            "full_forget_rougeL": forget_rouge,
            "full_retain_rougeL": retain_rouge,
            "full_generated_rows": len(full_generated),
            "base_audit_generated_rows": len(base_generated),
            "base_audit_complete": len(base_generated) == len(forget_authors) * 20,
        }

        columns = [
            "author_id",
            "question",
            "answer_logprob",
            "correct_perturbed_margin",
        ]
        separation = full[full.author_id.isin(forget_authors)][columns].merge(
            retain[retain.author_id.isin(forget_authors)][columns],
            on=["author_id", "question"],
            suffixes=("_full", "_retain"),
            validate="one_to_one",
        )
        separation["answer_delta"] = (
            separation.answer_logprob_full - separation.answer_logprob_retain
        )
        separation["margin_delta"] = (
            separation.correct_perturbed_margin_full
            - separation.correct_perturbed_margin_retain
        )
        utility = full[full.author_id.isin(rcontrol)][
            ["row_id", "author_id", "answer_logprob"]
        ].merge(
            retain[retain.author_id.isin(rcontrol)][["row_id", "answer_logprob"]],
            on="row_id",
            suffixes=("_full", "_retain"),
            validate="one_to_one",
        )
        utility_delta = float(
            (utility.answer_logprob_retain - utility.answer_logprob_full).mean()
        )
        control_gain = float(paired[paired.author_id.isin(rcontrol)].delta.mean())
        interval = clustered_ci(separation, "answer_delta")
        utility["delta"] = utility.answer_logprob_retain - utility.answer_logprob_full
        utility_interval = clustered_ci(utility, "delta")
        utility_complete = all(
            len(utility_frames[(model_state, split)])
            == len(read_jsonl(data / f"{split}.jsonl"))
            for model_state in ("base", "full", "retain")
            for split in ("real_authors", "world_facts")
        )
        external_utility = {}
        if utility_complete:
            for split in ("real_authors", "world_facts"):
                full_utility = utility_frames[("full", split)]
                retain_utility = utility_frames[("retain", split)]
                comparison = full_utility[["row_id", "answer_logprob"]].merge(
                    retain_utility[["row_id", "answer_logprob"]],
                    on="row_id",
                    suffixes=("_full", "_retain"),
                    validate="one_to_one",
                )
                external_utility[f"retain_minus_full_{split}_logprob"] = float(
                    (
                        comparison.answer_logprob_retain
                        - comparison.answer_logprob_full
                    ).mean()
                )
        separation_clear = interval[0] > 0
        if not utility_complete:
            gate2_status = "pending"
        elif not separation_clear:
            gate2_status = "failed"
        else:
            # The plan never froze a numerical equivalence margin for Gate 2.
            # Do not borrow the later Gate 3 20% guardrail or invent one after
            # seeing results; expose the complete evidence for formal review.
            gate2_status = "review_required"
        gate2 = {
            "status": gate2_status,
            "full_minus_retain_forget_logprob": float(separation.answer_delta.mean()),
            "author_clustered_ci95": interval,
            "full_minus_retain_perturbed_margin": float(separation.margin_delta.mean()),
            "retain_minus_full_rcontrol_logprob": utility_delta,
            "retain_minus_full_rcontrol_author_ci95": utility_interval,
            "full_minus_base_rcontrol_gain": control_gain,
            "external_utility_complete": utility_complete,
            "review_reason": (
                "Gate 2 specifies comparable utility but freezes no equivalence margin"
                if gate2_status == "review_required"
                else None
            ),
            **external_utility,
        }
    result = {
        "gate1": gate1,
        "gate2": gate2,
        "bootstrap_seed": 42,
        "bootstrap_samples": 5000,
    }
    atomic_json(root / "experiment/artifacts/gate_status.json", result)
    return result


def summarize(input_path: Path, output_path: Path) -> None:
    frame = pd.DataFrame(load_records(input_path).values())
    scalar = [
        column
        for column in frame.columns
        if not frame[column].map(lambda value: isinstance(value, (list, dict))).any()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[scalar].to_csv(output_path, index=False)
    print(f"Wrote {len(frame)} unique rows to {output_path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run/resume the Gate 1/2 bundle")
    run.add_argument("job", nargs="*", help="default: all jobs in frozen order")
    check = commands.add_parser("assess", help="assess saved evidence without a model")
    check.add_argument("--root", type=Path, default=ROOT)
    export = commands.add_parser("summarize", help="derive a scalar CSV from JSONL")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("_chunk", help=argparse.SUPPRESS)
    add_worker_arguments(worker)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        run_pipeline(args.job)
    elif args.command == "assess":
        print(json.dumps(assess(args.root.resolve()), indent=2, sort_keys=True))
    elif args.command == "summarize":
        summarize(args.input, args.output)
    else:
        evaluate_chunk(args)


if __name__ == "__main__":
    main()
