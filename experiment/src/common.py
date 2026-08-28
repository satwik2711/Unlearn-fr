"""Shared deterministic model, formatting, and serialization helpers."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "out_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
]


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def loader(
    path="models/Qwen3.5-2B",
    train=False,
    adapter_path=None,
    parent_adapter_path=None,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if parent_adapter_path or adapter_path:
        from peft import PeftModel

    if parent_adapter_path:
        model = PeftModel.from_pretrained(model, parent_adapter_path).merge_and_unload()
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.config.use_cache = not train
    return model, tok, device


def chat_ids(tok, q, a=None, max_length=256):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)[
        "input_ids"
    ]
    if a is None:
        return torch.tensor(prompt[-max_length:]).unsqueeze(0), len(
            prompt[-max_length:]
        )
    full_untrimmed = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": a}],
        tokenize=True,
        add_generation_prompt=False,
    )["input_ids"]
    answer_tokens = len(full_untrimmed) - len(prompt)
    full = full_untrimmed[-max_length:]
    labels = full.copy()
    prompt_tokens_kept = max(0, len(full) - answer_tokens)
    labels[:prompt_tokens_kept] = [-100] * prompt_tokens_kept
    return torch.tensor(full).unsqueeze(0), torch.tensor(labels).unsqueeze(0)


def score(model, tok, q, a, edit=None):
    x, y = chat_ids(tok, q, a)
    d = next(model.parameters()).device
    x = x.to(d)
    y = y.to(d)
    hook = None
    if edit:
        hook = edit_hook(
            model.model.layers[edit["layer"]],
            edit["position"],
            edit["vector"],
            edit["scale"],
        )
    with torch.no_grad():
        o = model(input_ids=x, labels=y)
    if hook:
        hook.remove()
    n = (y != -100).sum().item()
    return -float(o.loss) * 1.0 if n else float("nan")


def edit_hook(layer, position, vector, scale):
    v = (
        vector.to(next(layer.parameters()).device, dtype=next(layer.parameters()).dtype)
        * scale
    )

    def fn(_, __, out):
        z = out[0] if isinstance(out, tuple) else out
        z = z.clone()
        z[:, position, :] += v
        return (z,) + out[1:] if isinstance(out, tuple) else z

    return layer.register_forward_hook(fn)
