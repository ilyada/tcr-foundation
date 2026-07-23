"""
descriptors.py -- encoder-backed RepertoireFeaturizers: turn a ClonotypeEncoder's cloud into one vector.

MeanCov  : order-1 weighted mean ++ order-2 vech(covariance) of the encoder's L2-normalized cloud.
WithinV  : same, but each clonotype is centered within its own V gene first (removes the germline-V axis,
           isolating "what the encoder adds beyond V usage" -- the vaccine-signal probe).

Both wrap the canonical numpy moment code in repertoire.rep_descriptors (mean_cov_weighted_np) so numbers
match benchmark_ram. Weights are log-abundance (w = log1p(count), normalized). `fit` is a no-op (the frame-
based whitened-SPD descriptor, which DOES need a fitted reference frame, comes in a later slice via
repertoire_cloud).
"""
from __future__ import annotations

import numpy as np

from repertoire import rep_descriptors as D   # scripts/ on sys.path via package bootstrap
from . import schema as S


def _log_weights_np(count):
    w = np.log1p(np.asarray(count, dtype=np.float64))
    s = w.sum()
    return (w / s) if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))


def _within_v_center(Z, v_genes):
    """Subtract each V gene's own mean from its clonotypes (removes the germline-V location axis)."""
    out = Z.copy()
    vg = np.asarray(v_genes).astype(str)
    for v in np.unique(vg):
        m = vg == v
        out[m] -= Z[m].mean(0)
    return out


class MeanCov:
    """mean ++ vech(cov) of the encoder's cloud. dim = D.mean_cov_dim(encoder.dim)."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.dim = D.mean_cov_dim(encoder.dim)

    def fit(self, dfs) -> "MeanCov":
        return self

    def _cloud(self, df):
        Z, keep = self.encoder.encode(df)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        w = _log_weights_np(df[S.COUNT].to_numpy()[np.asarray(keep, bool)])
        return Z, w, keep

    def featurize(self, df) -> np.ndarray:
        Z, w, _ = self._cloud(df)
        return D.mean_cov_weighted_np(Z, w).astype(np.float32)


class WithinV(MeanCov):
    """MeanCov after within-V centering -- the encoder's contribution beyond germline V usage."""

    def featurize(self, df) -> np.ndarray:
        Z, w, keep = self._cloud(df)
        v = df[S.V_GENE].astype(str).to_numpy()[np.asarray(keep, bool)]
        return D.mean_cov_weighted_np(_within_v_center(Z, v), w).astype(np.float32)
