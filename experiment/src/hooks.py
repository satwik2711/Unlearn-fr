import torch


def capture_qend(model, input_ids):
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return torch.stack(
        [x[:, -1, :].detach().float().cpu() for x in out.hidden_states[1:]]
    )
