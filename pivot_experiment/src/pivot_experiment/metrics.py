"""Teacher-forced answer scoring with optional Q_END residual capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.nn.utils.rnn import pad_sequence

from .config import stable_hash
from .models import decoder_layers


EVALUATOR_VERSION = 3
FROZEN_PROMPT_DATE = "28 Aug 2026"


@dataclass(frozen=True)
class EncodedAnswer:
    input_ids: torch.Tensor
    prompt_length: int
    prompt_hash: str


def render_question_prompt(tokenizer, question: str) -> tuple[str, list[int], str]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        date_string=FROZEN_PROMPT_DATE,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return prompt, prompt_ids, stable_hash({"prompt": prompt, "token_ids": prompt_ids})


def encode_answer(tokenizer, question: str, answer: str) -> EncodedAnswer:
    prompt, prompt_ids, prompt_hash = render_question_prompt(tokenizer, question)
    full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Answer concatenation changed the frozen prompt tokenization")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("Reference answer produced no tokens")
    return EncodedAnswer(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        prompt_length=len(prompt_ids),
        prompt_hash=prompt_hash,
    )


class QEndCapture:
    def __init__(self, layers, q_end_indices: torch.Tensor):
        self.layers = layers
        self.q_end_indices = q_end_indices
        self.handles = []
        self.values: list[torch.Tensor] = []

    def _hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        indices = self.q_end_indices.to(hidden.device)
        self.values.append(hidden[rows, indices].detach().to("cpu", torch.bfloat16))

    def __enter__(self):
        self.handles = [layer.register_forward_hook(self._hook) for layer in self.layers]
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()

    def stacked(self) -> torch.Tensor:
        if len(self.values) != len(self.layers):
            raise RuntimeError("Did not capture exactly one output from every decoder layer")
        return torch.stack(self.values, dim=1).contiguous()


class QEndPatch:
    """Replace one decoder block's Q_END output with frozen donor vectors."""

    def __init__(
        self,
        layer,
        q_end_indices: torch.Tensor,
        donor_values: torch.Tensor,
    ):
        self.layer = layer
        self.q_end_indices = q_end_indices
        self.donor_values = donor_values
        self.handle = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.donor_values.shape != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError(
                "Q_END donor shape does not match the receiver batch and hidden size"
            )
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        indices = self.q_end_indices.to(hidden.device)
        patched = hidden.clone()
        patched[rows, indices] = self.donor_values.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()


class QEndSteer:
    """Add a fixed direction to one decoder block's Q_END output."""

    def __init__(
        self,
        layer,
        q_end_indices: torch.Tensor,
        direction: torch.Tensor,
        alpha: float,
        apply_once: bool = False,
    ):
        self.layer = layer
        self.q_end_indices = q_end_indices
        self.direction = direction
        self.alpha = alpha
        self.apply_once = apply_once
        self.applied = False
        self.handle = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.apply_once and self.applied:
            return output
        if self.direction.shape != (hidden.shape[-1],):
            raise ValueError("Steering direction does not match the model hidden size")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        indices = self.q_end_indices.to(hidden.device)
        steered = hidden.clone()
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        steered[rows, indices] = steered[rows, indices] + self.alpha * direction
        self.applied = True
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    def __enter__(self):
        self.applied = False
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()


def score_encoded_batch(
    model,
    tokenizer,
    encoded: Sequence[EncodedAnswer],
    capture_q_end: bool = False,
    patch_layer: int | None = None,
    patch_q_end_values: torch.Tensor | None = None,
    steer_layer: int | None = None,
    steer_direction: torch.Tensor | None = None,
    steer_alpha: float | None = None,
) -> tuple[list[dict], torch.Tensor | None]:
    if not encoded:
        return [], None
    pad_id = tokenizer.pad_token_id
    input_ids = pad_sequence(
        [item.input_ids for item in encoded],
        batch_first=True,
        padding_value=pad_id,
    )
    lengths = torch.tensor([len(item.input_ids) for item in encoded], dtype=torch.long)
    attention_mask = torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
    q_end = torch.tensor([item.prompt_length - 1 for item in encoded], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    layers = decoder_layers(model)
    capture = QEndCapture(layers, q_end) if capture_q_end else None
    if (patch_layer is None) != (patch_q_end_values is None):
        raise ValueError("patch_layer and patch_q_end_values must be provided together")
    if capture_q_end and patch_layer is not None:
        raise ValueError("Capture and patch are separate audit conditions")
    steer_values = (steer_layer, steer_direction, steer_alpha)
    if any(value is not None for value in steer_values) and not all(
        value is not None for value in steer_values
    ):
        raise ValueError("steer_layer, steer_direction and steer_alpha are required together")
    if capture_q_end and steer_layer is not None:
        raise ValueError("Capture and steering are separate conditions")
    if patch_layer is not None and steer_layer is not None:
        raise ValueError("Patching and steering are separate conditions")
    patch = (
        QEndPatch(layers[patch_layer], q_end, patch_q_end_values)
        if patch_layer is not None and patch_q_end_values is not None
        else None
    )
    steer = (
        QEndSteer(layers[steer_layer], q_end, steer_direction, steer_alpha)
        if steer_layer is not None
        and steer_direction is not None
        and steer_alpha is not None
        else None
    )
    with torch.inference_mode():
        if capture is not None:
            with capture:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        elif patch is not None:
            with patch:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        elif steer is not None:
            with steer:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    results: list[dict] = []
    for row_index, item in enumerate(encoded):
        target_positions = torch.arange(
            item.prompt_length,
            len(item.input_ids),
            device=output.logits.device,
        )
        target_ids = input_ids[row_index, target_positions]
        predictor_logits = output.logits[row_index, target_positions - 1].float()
        token_logprobs = functional.log_softmax(predictor_logits, dim=-1).gather(
            -1, target_ids[:, None]
        )[:, 0]
        results.append(
            {
                "mean_target_logprob": float(token_logprobs.mean().cpu()),
                "token_count": int(token_logprobs.numel()),
                "prompt_hash": item.prompt_hash,
            }
        )
    activations = capture.stacked() if capture is not None else None
    return results, activations


def score_answers(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    batch_size: int,
) -> list[dict]:
    results: list[dict] = []
    for start in range(0, len(pairs), batch_size):
        encoded = [
            encode_answer(tokenizer, question, answer)
            for question, answer in pairs[start : start + batch_size]
        ]
        batch_results, _ = score_encoded_batch(model, tokenizer, encoded)
        results.extend(batch_results)
    return results
