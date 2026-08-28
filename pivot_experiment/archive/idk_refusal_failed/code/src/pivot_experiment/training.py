"""Balanced IDK objective and exact base-parameter fingerprinting."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch.nn.utils.rnn import pad_sequence

from .metrics import EncodedAnswer, encode_answer


def exact_base_parameter_hash(model) -> str:
    """SHA256 every non-LoRA parameter without retaining CPU copies."""
    digest = hashlib.sha256()
    count = 0
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if "lora_" in name:
                continue
            tensor = parameter.detach().contiguous().to("cpu")
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            count += tensor.numel()
            del tensor
    digest.update(str(count).encode("ascii"))
    return digest.hexdigest()


def _per_example_losses(model, tokenizer, encoded: Sequence[EncodedAnswer]):
    input_ids = pad_sequence(
        [item.input_ids for item in encoded],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    lengths = torch.tensor([len(item.input_ids) for item in encoded], dtype=torch.long)
    attention_mask = torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
    labels = input_ids.clone()
    for row, item in enumerate(encoded):
        labels[row, : item.prompt_length] = -100
        labels[row, lengths[row] :] = -100
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shifted_logits = output.logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    flat_loss = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(shifted_labels.shape)
    mask = shifted_labels.ne(-100)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("A training response produced no supervised tokens")
    return (flat_loss * mask).sum(dim=1) / counts


def balanced_pair_loss(model, tokenizer, pairs: Sequence[dict], retain_lambda: float):
    forget = [
        encode_answer(tokenizer, pair["forget_question"], pair["refusal_answer"])
        for pair in pairs
    ]
    retain = [
        encode_answer(tokenizer, pair["retain_question"], pair["retain_answer"])
        for pair in pairs
    ]
    losses = _per_example_losses(model, tokenizer, [*forget, *retain])
    size = len(pairs)
    forget_loss = losses[:size].mean()
    retain_loss = losses[size:].mean()
    return forget_loss + retain_lambda * retain_loss, forget_loss, retain_loss

