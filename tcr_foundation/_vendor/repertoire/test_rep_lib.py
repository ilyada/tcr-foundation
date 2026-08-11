"""Sanity tests for the repertoire library (rep_metrics / rep_descriptors / rep_sampling). Run: python test_rep_lib.py"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repertoire import rep_metrics as M, rep_descriptors as D, rep_sampling as S


def test_metrics():
    rng = np.random.RandomState(0)
    # two well-separated clusters -> donor-centric AUROC ~ 1; random labels -> ~0.5
    V = np.concatenate([rng.randn(40, 16) + 5, rng.randn(40, 16) - 5])
    lab = np.array([0] * 40 + [1] * 40)
    assert M.donor_centric_auroc(V, lab) > 0.98, "separable clusters should give AUROC ~1"
    assert abs(M.donor_centric_auroc(V, rng.permutation(lab)) - 0.5) < 0.12, "shuffled labels ~ chance"
    # ref-size sweep: AUROC non-decreasing-ish and high on structured data
    sw = M.ref_size_sweep_auroc(V, lab, [1, 5, 20])
    assert sw[20] > 0.9 and sw[1] > 0.8, f"sweep should be high on separable data: {sw}"
    print("metrics OK", {k: round(v, 3) for k, v in sw.items()})


def test_descriptors():
    B, n, k = 3, 500, 8
    t = torch.randn(B, n, k, requires_grad=True)
    d = D.mean_cov_descriptor(t)
    assert d.shape == (B, D.mean_cov_dim(k)), d.shape
    assert torch.allclose(d.norm(dim=1), torch.ones(B), atol=1e-5), "descriptor must be L2-normalized"
    d.sum().backward()
    assert t.grad is not None, "descriptor must be differentiable"
    # vech isometry: ||vech(C)|| == ||C||_F
    C = torch.randn(1, k, k); C = C @ C.transpose(1, 2)
    assert torch.allclose(D.vech(C).norm(), C.norm(), atol=1e-4), "vech must be isometric"
    # occupancy: 4 orthonormal points assigned to 4 prototypes = those points -> mass ~uniform, sums to 1
    Z = np.eye(4, 6).astype(np.float32); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    w = np.ones(4, np.float32) / 4
    occ = D.weighted_occupancy(Z, w, Z, tau=0.05)
    assert abs(occ.sum() - 1) < 1e-4, "occupancy must sum to 1"
    assert occ.max() < 0.4, "orthogonal points -> ~uniform occupancy"
    print("descriptors OK (dim", D.mean_cov_dim(k), ")")


def test_sampling():
    if not torch.cuda.is_available():
        print("sampling: no CUDA, skipping GPU sampler"); return
    dev = torch.device("cuda")
    rng = np.random.RandomState(0)
    clouds = [(rng.randn(ni, 8).astype(np.float32), np.ones(ni, np.float32) / ni) for ni in [300, 1200, 50]]
    Zp, Wp = S.pad_batch_to_gpu(clouds, dev)
    v = S.sample_view_gpu(Zp, Wp, 256)
    assert v.shape == (3, 256, 8), v.shape
    # weighted toward one point -> that point dominates the view
    w = np.zeros(300, np.float32); w[7] = 1.0
    Zp2, Wp2 = S.pad_batch_to_gpu([(clouds[0][0], w)], dev)
    idx_view = S.sample_view_gpu(Zp2, Wp2, 128)
    assert torch.allclose(idx_view[0], torch.from_numpy(clouds[0][0][7]).to(dev), atol=1e-2), "weight=1 point dominates"
    print("sampling OK", tuple(v.shape))


if __name__ == "__main__":
    test_metrics(); test_descriptors(); test_sampling()
    print("ALL TESTS PASSED")
