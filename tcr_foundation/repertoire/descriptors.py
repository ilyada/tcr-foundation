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

from repertoire import rep_descriptors as D          # scripts/ on sys.path via package bootstrap
from repertoire import repertoire_cloud as rc        # whitened-SPD descriptor (fit_frame + descriptor)
from ..core import schema as S


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


# ----------------------------------------------------------------------------- corpus pooling (fit helper)
def _pool_embeddings(encoder, dfs, per_cloud=2000, seed=0):
    """Pool a subsample of the encoder's embeddings across `dfs` (for fitting a frame / codebook). Returns
    a RAW [N, dim] array (not normalized -- callers normalize as their geometry needs)."""
    rng = np.random.RandomState(seed)
    chunks = []
    for df in dfs:
        Z, _ = encoder.encode(df)
        if len(Z) > per_cloud:
            Z = Z[rng.choice(len(Z), per_cloud, replace=False)]
        chunks.append(np.asarray(Z, np.float32))
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, encoder.dim), np.float32)


# ----------------------------------------------------------------------------- occupancy (landmark / bag-of-TCR-words)
class Occupancy:
    """Weighted soft-assignment mass over K landmarks (a Nystrom KME / bag-of-TCR-words descriptor -- captures
    which regions of the encoder's space a repertoire POPULATES, unlike the mean/cov moments). fit() pools the
    encoder's embeddings across a corpus and k-means them into K L2-normed landmarks. dim = K."""

    def __init__(self, encoder, k=256, tau=0.1, ref_per_cloud=2000, seed=0):
        self.encoder = encoder
        self.k = k
        self.tau = tau
        self.ref_per_cloud = ref_per_cloud
        self.seed = seed
        self.landmarks = None
        self.dim = k

    def fit(self, dfs) -> "Occupancy":
        from sklearn.cluster import MiniBatchKMeans
        ref = _pool_embeddings(self.encoder, dfs, self.ref_per_cloud, self.seed)
        Zn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)     # spherical geometry
        km = MiniBatchKMeans(n_clusters=self.k, random_state=self.seed, batch_size=10_000, n_init=3, max_iter=200)
        km.fit(Zn)
        C = km.cluster_centers_
        self.landmarks = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        return self

    def featurize(self, df) -> np.ndarray:
        if self.landmarks is None:
            raise RuntimeError("Occupancy.fit(dfs) must be called before featurize()")
        Z, keep = self.encoder.encode(df)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        w = _log_weights_np(df[S.COUNT].to_numpy()[np.asarray(keep, bool)])
        return D.weighted_occupancy(Z, w, self.landmarks, tau=self.tau).astype(np.float32)


# ----------------------------------------------------------------------------- whitened-SPD (the locked descriptor)
class WhitenedSPD:
    """The LOCKED repertoire-cloud descriptor: order-1 mean ++ order-2 vech(logm(shrunk cov)) in a MODEL-INTRINSIC
    whitened PCA frame (fit once on a pooled reference). Richer than plain mean+cov (correct SPD/Riemannian
    geometry). Wraps repertoire_cloud.fit_frame + descriptor. dim = descriptor_dim(m1, k) (default = full)."""

    def __init__(self, encoder, m1=None, k=None, ref_per_cloud=2000, ref_seed=0):
        self.encoder = encoder
        self.m1 = m1
        self.k = k
        self.ref_per_cloud = ref_per_cloud
        self.ref_seed = ref_seed
        self.frame = None
        self.dim = None

    def fit(self, dfs) -> "WhitenedSPD":
        ref = _pool_embeddings(self.encoder, dfs, self.ref_per_cloud, self.ref_seed)   # RAW (fit_frame normalizes)
        self.frame = rc.fit_frame(ref, k_max=self.encoder.dim)
        avail = self.frame["V"].shape[1]
        m1 = avail if self.m1 is None else min(self.m1, avail)
        k = avail if self.k is None else min(self.k, avail)
        self.dim = rc.descriptor_dim(m1, k)
        return self

    def featurize(self, df) -> np.ndarray:
        if self.frame is None:
            raise RuntimeError("WhitenedSPD.fit(dfs) must be called before featurize()")
        Z, keep = self.encoder.encode(df)                                  # RAW (descriptor normalizes internally)
        w = _log_weights_np(df[S.COUNT].to_numpy()[np.asarray(keep, bool)])
        return rc.descriptor(Z, w, self.frame, m1=self.m1, k=self.k).astype(np.float32)
