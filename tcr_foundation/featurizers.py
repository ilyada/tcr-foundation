"""
featurizers.py -- model-free RepertoireFeaturizers (one vector per repertoire, straight from sequences).

VUsage  : weighted V-gene usage distribution over a shared vocab.
Kmer    : weighted CDR3 k-mer distribution, sqrt-ed (Hellinger embedding) so cosine == Bhattacharyya.

Both need a corpus-shared vocabulary: call .fit(list_of_dfs) once, then .featurize(df). Weights are
log-abundance w = log1p(count) normalized to sum 1 (matches the repertoire-cloud w_log convention).

Thin wrappers over the canonical implementations in repertoire.rep_metrics (imported via the package
bootstrap) -- one source of truth for the vocab/vector math, so numbers match the existing benchmarks.
"""
from __future__ import annotations

import numpy as np

from repertoire import rep_metrics as M   # scripts/ is on sys.path via tcr_foundation/__init__.py
from . import schema as S


def _log_weights(df):
    w = np.log1p(df[S.COUNT].to_numpy(dtype=np.float64))
    s = w.sum()
    return (w / s) if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))


class VUsage:
    """Weighted V-gene usage vector over a fitted vocab. dim = |vocab| after fit()."""

    def __init__(self):
        self.vocab = None
        self.dim = 0

    def fit(self, dfs) -> "VUsage":
        self.vocab = M.build_vgene_vocab(df[S.V_GENE].astype(str).to_numpy() for df in dfs)
        self.dim = len(self.vocab)
        return self

    def featurize(self, df) -> np.ndarray:
        if self.vocab is None:
            raise RuntimeError("VUsage.fit(dfs) must be called before featurize()")
        return M.vusage_vector(df[S.V_GENE].astype(str).to_numpy(), _log_weights(df), self.vocab).astype(np.float32)


class Kmer:
    """Weighted CDR3 k-mer distribution, sqrt-ed (Hellinger). dim = |kmer vocab| after fit()."""

    def __init__(self, k: int = 3):
        self.k = k
        self.vocab = None
        self.dim = 0

    def fit(self, dfs) -> "Kmer":
        self.vocab = M.build_kmer_vocab((df[S.CDR3].astype(str).to_numpy() for df in dfs), k=self.k)
        self.dim = len(self.vocab)
        return self

    def featurize(self, df) -> np.ndarray:
        if self.vocab is None:
            raise RuntimeError("Kmer.fit(dfs) must be called before featurize()")
        v = M.kmer_vector(df[S.CDR3].astype(str).to_numpy(), _log_weights(df), self.vocab, k=self.k)
        return np.sqrt(v).astype(np.float32)   # Hellinger embedding
