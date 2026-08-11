"""
repertoire_cloud.py — turn a donor's clonotype embedding cloud into ONE fixed descriptor vector.

The locked design (KB: repertoire_cloud_descriptor / covariance_descriptors_spd):

    descriptor(donor) = [ order-1 mean block ] (+) [ order-2 covariance block ]

computed in a MODEL-INTRINSIC, WHITENED PCA frame, with the covariance done in the right (SPD/Riemannian)
geometry: Ledoit-Wolf shrinkage -> matrix logarithm -> vech.

Pipeline for one donor's cloud Z (n_clonotypes x d) with weights w (per clonotype, e.g. log-frequency):
  0. L2-normalize each embedding (the geometry lives on the unit sphere).
  1. mean block  = W_m1^T (m - mu_ref)              -> [m1]      "where the cloud sits" (order 1)
  2. cov block   = vech( logm( shrink( C ) ) )       -> [k(k+1)/2]   "shape of the cloud" (order 2)
     where C = W_k^T C_full W_k is the weighted covariance projected into the whitened top-k subspace.

W_m1 / W_k are the whitening maps built ONCE from a reference pool (fit_frame): project onto the top
principal axes AND divide by their typical spread, so a donor is measured relative to how the model
spreads clonotypes in general. Pure numpy; no torch, no project utils.
"""

import numpy as np

EPS = 1e-8  # floor for eigenvalues / norms (numerical safety)


# ----------------------------------------------------------------------------- basics

def l2_normalize(Z):
    """Row-wise L2 normalization -> unit vectors on the sphere. Zero rows are left as zeros."""
    Z = np.asarray(Z, dtype=np.float64)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, EPS)


def _norm_weights(w, n):
    """Return non-negative weights summing to 1 (fall back to uniform if degenerate)."""
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


# ----------------------------------------------------------------------------- the frame

def fit_frame(ref_Z, k_max=128):
    """Fit the model-intrinsic PCA frame on a reference pool of clonotype embeddings.

    Returns dict {mu_ref [d], V [d x d] (columns = principal axes, eigenvalue-descending),
    eigvals [d] (variance along each axis)}. ref_Z is L2-normalized here, so any sample of raw
    embeddings can be passed in. This is donor-independent and fit ONCE."""
    Zn = l2_normalize(ref_Z)
    mu_ref = Zn.mean(axis=0)
    Zc = Zn - mu_ref
    # covariance of the reference cloud (this defines the "typical spread" per direction)
    S = (Zc.T @ Zc) / Zc.shape[0]
    eigvals, V = np.linalg.eigh(S)          # ascending
    order = np.argsort(eigvals)[::-1]        # -> descending (top axes first)
    eigvals, V = eigvals[order], V[:, order]
    return {"mu_ref": mu_ref, "V": V[:, :k_max], "eigvals": eigvals[:k_max]}


def whiten_maps(frame, m1, k):
    """Whitening maps W = V_top / sqrt(eigval_top): project onto top axes AND rescale to unit spread.
    W_m1 (d x m1) for the order-1 mean block; W_k (d x k) for the order-2 covariance subspace."""
    V, eig = frame["V"], frame["eigvals"]
    scale = 1.0 / np.sqrt(np.maximum(eig, EPS))
    W_m1 = V[:, :m1] * scale[:m1]
    W_k = V[:, :k] * scale[:k]
    return W_m1, W_k


# ----------------------------------------------------------------------------- moments

def weighted_mean(Z, w):
    """Frequency-weighted mean vector m = sum_c w_c z_c (w sums to 1)."""
    return w @ Z


def ledoit_wolf_shrink(C, Xc, w):
    """Parameter-free Ledoit-Wolf shrinkage of a weighted covariance toward a scaled identity.

    C   : [k x k] weighted covariance (already computed from the centered, whitened points Xc).
    Xc  : [n x k] centered points in the same (whitened top-k) space.
    w   : [n]     normalized weights.
    Returns (C_shrunk, delta). delta in [0,1] is the optimal blend toward mu*I; it is larger when the
    estimate is noisier (small effective N), pulling over-dispersed eigenvalues back toward the mean so
    the matrix log later does not blow up on near-zero eigenvalues.
    """
    k = C.shape[0]
    mu = np.trace(C) / k                       # average eigenvalue -> the round-ball target is mu*I
    d2 = np.sum((C - mu * np.eye(k)) ** 2)     # ||C - mu I||_F^2 : how far C is from the target
    if d2 <= EPS:
        return C, 0.0
    # variance of the estimate: weighted analogue of (1/n^2) sum ||x x^T - C||^2, using
    # ||x x^T - C||_F^2 = ||x||^4 - 2 x^T C x + ||C||_F^2  (avoids building n outer products)
    norms4 = np.einsum("ij,ij->i", Xc, Xc) ** 2
    quad = np.einsum("ij,jk,ik->i", Xc, C, Xc)
    per_c = norms4 - 2.0 * quad + np.sum(C ** 2)
    bbar2 = np.sum((w ** 2) * per_c)
    b2 = min(bbar2, d2)                         # cap so delta stays in [0,1]
    delta = b2 / d2
    C_shrunk = delta * mu * np.eye(k) + (1.0 - delta) * C
    return C_shrunk, float(delta)


def spd_logm(C):
    """Matrix logarithm of a symmetric positive-definite matrix via eigendecomposition
    (stable for small k x k SPD: log acts on the eigenvalues, eigenvectors unchanged)."""
    l, U = np.linalg.eigh(C)
    l = np.maximum(l, EPS)                      # guard: shrinkage already lifts these off zero
    return (U * np.log(l)) @ U.T


def vech(M):
    """Half-vectorization of a symmetric matrix: upper triangle incl. diagonal, with sqrt(2) on the
    off-diagonal so the vector's Euclidean norm equals the matrix Frobenius norm. -> [k(k+1)/2]."""
    k = M.shape[0]
    iu = np.triu_indices(k)
    vals = M[iu].astype(np.float64).copy()
    off = iu[0] != iu[1]
    vals[off] *= np.sqrt(2.0)
    return vals


# ----------------------------------------------------------------------------- the descriptor

def descriptor(Z, w, frame, m1=None, k=None):
    """Cloud (Z [n x d] raw embeddings, w [n] weights) -> one descriptor vector [m1 + k(k+1)/2].

    Order-1 mean block + order-2 covariance block, both in the whitened frame. Returns a 1-D np array.
    m1 / k default to None = use ALL available axes (full mean + FULL covariance, no top-k truncation);
    pass an int to truncate to the top-m1 / top-k principal axes instead."""
    Zn = l2_normalize(Z)
    w = _norm_weights(w, Zn.shape[0])
    avail = frame["V"].shape[1]
    m1 = avail if m1 is None else min(m1, avail)
    k = avail if k is None else min(k, avail)
    W_m1, W_k = whiten_maps(frame, m1, k)
    mu_ref = frame["mu_ref"]

    # order 1: weighted mean, expressed in the whitened top-m1 frame
    m = weighted_mean(Zn, w)
    mean_block = W_m1.T @ (m - mu_ref)                         # [m1]

    # order 2: project points into the whitened top-k subspace, weighted covariance there
    Xk = (Zn - mu_ref) @ W_k                                   # [n x k]
    mk = w @ Xk                                                # weighted mean in that subspace
    Xkc = Xk - mk                                              # centered
    C = Xkc.T @ (w[:, None] * Xkc)                             # [k x k] weighted covariance
    C_shrunk, _delta = ledoit_wolf_shrink(C, Xkc, w)
    cov_block = vech(spd_logm(C_shrunk))                       # [k(k+1)/2]

    return np.concatenate([mean_block, cov_block])


def descriptor_dim(m1=64, k=16):
    """Length of the descriptor for given m1, k."""
    return m1 + k * (k + 1) // 2
