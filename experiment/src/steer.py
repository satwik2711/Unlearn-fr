"""Direction construction and frozen norm-matched control helpers."""

import torch


def unit_difference(full, idk):
    v = (full - idk).mean(0)
    return v / (v.norm() + 1e-12)


def random_like(v, n=20, seed=42):
    g = torch.Generator().manual_seed(seed)
    controls = torch.randn((n, *v.shape), generator=g, dtype=torch.float32)
    flat = controls.flatten(start_dim=1)
    norms = flat.norm(dim=1).clamp_min(1e-12)
    shape = (n,) + (1,) * v.ndim
    return controls / norms.reshape(shape)
