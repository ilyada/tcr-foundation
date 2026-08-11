"""
rep_sampling.py -- depth-augmentation view sampling for repertoire training.

The old per-instance np.random.choice loop on CPU was the training bottleneck (GPU sat idle -- see the
GPU-utilization diagnosis). Here the whole batch is padded and sampled with ONE torch.multinomial on the GPU:

    Zpad, Wpad = pad_batch_to_gpu(batch_clouds, device)   # once per step
    v1 = sample_view_gpu(Zpad, Wpad, n1)                  # [B, n1, d] weighted, with-replacement
    v2 = sample_view_gpu(Zpad, Wpad, n2)                  # different random depth

Zero-weight padding rows are never drawn, so variable-size clouds pack into one tensor. Cap clouds at load time
to bound Nmax (memory). Sampling and gather run on the GPU -> the CPU loop disappears.
"""

import numpy as np
import torch


def draw_size(n_min, n_max, rng):
    """Random view depth ~ log-uniform in [n_min, n_max] (mimics varying sequencing depth -> depth-invariance)."""
    return int(round(np.exp(rng.uniform(np.log(n_min), np.log(n_max)))))


def pad_batch_to_gpu(batch_clouds, device):
    """batch_clouds: list of (Z [Ni,d], w [Ni]). Returns padded (Zpad [B,Nmax,d] fp16, Wpad [B,Nmax] fp32) on GPU.
    Build once per step; sample multiple views from it."""
    B = len(batch_clouds)
    d = batch_clouds[0][0].shape[1]
    sizes = [len(w) for _, w in batch_clouds]
    Nmax = max(sizes)
    Zpad = torch.zeros(B, Nmax, d, device=device, dtype=torch.float16)
    Wpad = torch.zeros(B, Nmax, device=device, dtype=torch.float32)
    for i, (Z, w) in enumerate(batch_clouds):
        ni = sizes[i]
        Zpad[i, :ni].copy_(torch.from_numpy(np.ascontiguousarray(Z)).to(torch.float16))
        Wpad[i, :ni].copy_(torch.from_numpy(np.ascontiguousarray(w)))
    return Zpad, Wpad


def sample_view_gpu(Zpad, Wpad, n):
    """One weighted, with-replacement view of depth n per row -> [B, n, d] fp32 (on GPU)."""
    idx = torch.multinomial(Wpad, n, replacement=True)             # [B, n]; zero-weight pad never drawn
    d = Zpad.shape[-1]
    return torch.gather(Zpad, 1, idx.unsqueeze(-1).expand(-1, -1, d)).float()
