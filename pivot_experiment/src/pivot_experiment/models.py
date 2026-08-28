"""Shared tokenizer and public causal-LM loading."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import stable_hash


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tokenizer_spec(models_config: dict) -> dict:
    source = models_config["tokenizer_source"]
    return models_config["models"][source]


def load_tokenizer(models_config: dict):
    spec = tokenizer_spec(models_config)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("The frozen FULL tokenizer has no chat template")
    return tokenizer


def inspect_public_configs(models_config: dict) -> dict[str, Any]:
    expected = models_config["expected_architecture"]
    found: dict[str, Any] = {}
    for state, spec in models_config["models"].items():
        config = AutoConfig.from_pretrained(spec["repo_id"], revision=spec["revision"])
        details = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "model_type": config.model_type,
            "decoder_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "vocab_size": getattr(config, "vocab_size", None),
        }
        for key in ("model_type", "decoder_layers", "hidden_size"):
            if details[key] != expected[key]:
                raise ValueError(
                    f"{state} {key}={details[key]!r}, expected {expected[key]!r}"
                )
        found[state] = details
    tokenizer = load_tokenizer(models_config)
    found["tokenizer"] = {
        **tokenizer_spec(models_config),
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "chat_template_present": bool(tokenizer.chat_template),
        "chat_template_hash": stable_hash(tokenizer.chat_template),
        "backend_tokenizer_hash": stable_hash(tokenizer.backend_tokenizer.to_str()),
    }
    return found


def load_public_model(state: str, models_config: dict):
    if state not in models_config["models"]:
        raise ValueError(f"No public model configured for state {state!r}")
    spec = models_config["models"][state]
    dtype_name = models_config["dtype"]
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    device = resolve_device(models_config["device"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        dtype=DTYPES[dtype_name],
        low_cpu_mem_usage=True,
        device_map={"": device},
        attn_implementation=models_config["attention_implementation"],
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.layers
    expected = models_config["expected_architecture"]
    if len(layers) != expected["decoder_layers"]:
        raise ValueError(f"Loaded {len(layers)} layers, expected {expected['decoder_layers']}")
    if model.config.hidden_size != expected["hidden_size"]:
        raise ValueError("Loaded model hidden size does not match frozen configuration")
    return model, device
