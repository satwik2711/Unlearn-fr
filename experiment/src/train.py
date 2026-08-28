"""LoRA state construction with explicit checkpoint lineage."""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from common import TARGETS, chat_ids, loader, read_jsonl, seed_all


def save(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(obj, indent=2))
    t.replace(p)


def rows(name):
    return read_jsonl(f"experiment/artifacts/data/{name}.jsonl")


def main(a):
    seed_all()
    if a.state in ["FULL", "RETAIN"]:
        model, tok, dev = loader(train=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0,
                target_modules=TARGETS,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        train_rows = rows("full" if a.state == "FULL" else "retain90")
    elif a.state in ["IDK", "GD"]:
        if not a.full_adapter:
            raise ValueError("--full-adapter is required for IDK and GD")
        # Merge a copy of FULL into memory, then train only a fresh child LoRA.
        # The selected FULL adapter on disk remains byte-for-byte unchanged.
        model, tok, dev = loader(
            train=True,
            parent_adapter_path=a.full_adapter,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0,
                target_modules=TARGETS,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        train_rows = rows("forget10")
    else:
        raise ValueError(a.state)
    if a.max_examples:
        train_rows = train_rows[: a.max_examples]
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=a.lr, weight_decay=0.01
    )
    accum = a.accum
    history = []
    out = Path(f"experiment/checkpoints/{a.state.lower()}")
    out.mkdir(parents=True, exist_ok=True)
    retain = rows("retain90")
    t0 = time.time()
    step = 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad()
        losses = []
        for start in range(0, len(train_rows), a.microbatch):
            batch = train_rows[start : start + a.microbatch]
            xy = []
            for j, r in enumerate(batch):
                ans = (
                    r["answer"]
                    if a.state not in ["IDK"]
                    else [
                        "I do not know that information.",
                        "I cannot provide that information.",
                        "I am not sure about that author.",
                    ][(start + j) % 3]
                )
                xy.append(chat_ids(tok, r["question"], ans))
            x = torch.nn.utils.rnn.pad_sequence(
                [z[0][0] for z in xy], batch_first=True, padding_value=tok.pad_token_id
            ).to(dev)
            y = torch.nn.utils.rnn.pad_sequence(
                [z[1][0] for z in xy], batch_first=True, padding_value=-100
            ).to(dev)
            att = (x != tok.pad_token_id).long()
            o = model(input_ids=x, attention_mask=att, labels=y)
            loss = o.loss
            if a.state == "GD":
                loss = -loss
            # preserve retain in the same update stream for suppression/GD branches
            if a.state in ["IDK", "GD"]:
                rxy = [
                    chat_ids(
                        tok,
                        retain[(start + j) % len(retain)]["question"],
                        retain[(start + j) % len(retain)]["answer"],
                    )
                    for j in range(len(batch))
                ]
                rx = torch.nn.utils.rnn.pad_sequence(
                    [z[0][0] for z in rxy],
                    batch_first=True,
                    padding_value=tok.pad_token_id,
                ).to(dev)
                ry = torch.nn.utils.rnn.pad_sequence(
                    [z[1][0] for z in rxy], batch_first=True, padding_value=-100
                ).to(dev)
                ro = model(
                    input_ids=rx,
                    attention_mask=(rx != tok.pad_token_id).long(),
                    labels=ry,
                )
                loss = loss + ro.loss
            (loss / accum).backward()
            losses.append(float(loss.detach()))
            if ((start // a.microbatch) + 1) % accum == 0 or start + len(batch) == len(
                train_rows
            ):
                opt.step()
                opt.zero_grad()
                step += 1
        ck = out / f"epoch-{epoch}"
        model.save_pretrained(ck)
        record = {
            "state": a.state,
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "examples": len(train_rows),
            "optimizer_steps": step,
            "elapsed_s": time.time() - t0,
            "checkpoint": str(ck),
            "parent_adapter": a.full_adapter,
        }
        history.append(record)
        save(history, out / "trajectory.json")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("state", choices=["FULL", "RETAIN", "IDK", "GD"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--max-examples", type=int)
    p.add_argument("--full-adapter")
    main(p.parse_args())
