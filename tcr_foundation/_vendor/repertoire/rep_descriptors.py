"""
rep_descriptors.py -- turn a repertoire's (transformed) point cloud into one vector.

Two families, one interface:
  - occupancy (numpy): w-weighted soft-assignment mass over K prototypes/landmarks -> [K]. Used at inference for
    the landmark descriptors (fixed or learned prototypes).
  - mean+cov (torch, differentiable): order-1 weighted mean ++ order-2 vech(covariance) of the cloud -> a vector.
    Used both as a fixed descriptor and as the differentiable pooling behind a learned transform g_theta.

The mean+cov vech uses sqrt(2) on the off-diagonal so ||vech(C)||_2 == ||C||_Frobenius (isometric flattening).
"""

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- occupancy (numpy, inference)
def weighted_occupancy(Z, w, prototypes, tau=0.1):
    """w-weighted soft-assignment occupancy over prototypes. Z [N,d] L2-normalized, prototypes [K,d]. Returns [K]."""
    P = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    s = (Z @ P.T) / tau
    s -= s.max(1, keepdims=True)
    a = np.exp(s); a /= a.sum(1, keepdims=True)
    return (w[:, None] * a).sum(0)


# ----------------------------------------------------------------------------- mean+cov (torch, differentiable)
def vech(C):
    """Half-vectorization with sqrt2 off-diagonal (||vech||_2 == ||C||_F). C [B,k,k] -> [B, k(k+1)/2]."""
    k = C.shape[-1]
    iu = torch.triu_indices(k, k, device=C.device)
    vals = C[:, iu[0], iu[1]]
    scale = torch.ones(vals.shape[-1], device=C.device)
    scale[iu[0] != iu[1]] = 2 ** 0.5
    return vals * scale


def mean_cov_descriptor(t, normalize=True):
    """Cloud t [B, n, k] -> [B, k + k(k+1)/2] = mean ++ vech(cov). Differentiable. Uniform over the (weight-sampled)
    view points, so no separate weights here -- weighting rides in the sampling of the view."""
    m = t.mean(1)
    tc = t - m.unsqueeze(1)
    C = tc.transpose(1, 2) @ tc / t.shape[1]
    d = torch.cat([m, vech(C)], 1)
    return F.normalize(d, dim=1) if normalize else d


def mean_cov_dim(k):
    return k + k * (k + 1) // 2


def mean_cov_weighted_np(Z, w):
    """w-WEIGHTED mean+cov of a full cloud (numpy, inference). Z [N,d] L2-normalized, w [N] sum-1.
    Returns [d + d(d+1)/2] = mean ++ vech(cov) (sqrt2 off-diag), L2-normalized. Abundance enters via w."""
    m = (w[:, None] * Z).sum(0)
    Zc = Z - m
    C = (w[:, None] * Zc).T @ Zc
    k = Z.shape[1]
    iu = np.triu_indices(k)
    vals = C[iu].astype(np.float64).copy()
    vals[iu[0] != iu[1]] *= 2 ** 0.5
    d = np.concatenate([m, vals])
    return (d / (np.linalg.norm(d) + 1e-8)).astype(np.float32)
