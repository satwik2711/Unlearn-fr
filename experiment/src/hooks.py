"""Residual-stream capture helpers shared by the intervention stages."""

import torch


def capture_qend(model, input_ids, q_end_token_index):
    """Capture every complete language layer at a precomputed Q_END index."""
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return torch.stack(
        [
            hidden[:, q_end_token_index, :].detach().float().cpu()
            for hidden in out.hidden_states[1:]
        ]
    )
