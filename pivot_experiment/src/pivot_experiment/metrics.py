"""Teacher-forced answer scoring with optional Q_END residual capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.nn.utils.rnn import pad_sequence

from .config import stable_hash


EVALUATOR_VERSION = 1


@dataclass(frozen=True)
class EncodedAnswer:
    input_ids: torch.Tensor
    prompt_length: int
    prompt_hash: str


def encode_answer(tokenizer, question: str, answer: str) -> EncodedAnswer:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Answer concatenation changed the frozen prompt tokenization")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("Reference answer produced no tokens")
    return EncodedAnswer(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        prompt_length=len(prompt_ids),
        prompt_hash=stable_hash({"prompt": prompt, "token_ids": prompt_ids}),
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


def score_encoded_batch(
    model,
    tokenizer,
    encoded: Sequence[EncodedAnswer],
    capture_q_end: bool = False,
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

    capture = QEndCapture(model.model.layers, q_end) if capture_q_end else None
    with torch.inference_mode():
        if capture is None:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with capture:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

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

