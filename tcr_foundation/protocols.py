"""
protocols.py -- the swappable "каркас": typed interfaces every encoder / featurizer / distance implements.

Three levels, kept explicitly separate (this separation is the whole point -- you can swap any one piece):

  ClonotypeEncoder    df -> Z [N, dim]   per-clonotype embedding vectors  (neural foundation, SCEPTR, ...)
  RepertoireFeaturizer  df -> v [D]      ONE vector per repertoire        (V-usage, k-mer, mean+cov, whitened-SPD)
  PairwiseDistance    df -> D [N, N]     per-clonotype distance matrix    (in-house TCRdist -- not a vector space)

A RepertoireFeaturizer may be model-free (V-usage/k-mer, straight from sequences) or wrap a ClonotypeEncoder
(mean+cov / whitened-SPD of the encoder's Z). Distances are their own level because TCRdist has no natural
per-clonotype vector -- downstream uses it via kNN / landmark features, not via cosine on a descriptor.

These are `typing.Protocol`s (structural): an object is an encoder if it has `.dim` and `.encode(df)`; no
inheritance required. `df` is always a canonical clonotype DataFrame (see schema.py): columns v_gene, cdr3
(+ optional j_gene, count, cdr1, cdr2).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ClonotypeEncoder(Protocol):
    """Per-clonotype embedding. `dim` is the output width (declared, so downstream stays dim-agnostic).

    `encode` returns (Z, keep) -- mirrors the proven benchmark_ram embedder contract: some encoders drop
    rows (e.g. a V gene that tidytcells cannot resolve, or a non-TRBV gene for SCEPTR), so `keep` is a
    boolean mask over the input rows and Z has `keep.sum()` rows aligned to `df[keep]`."""

    dim: int

    def encode(self, df: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
        """Canonical clonotype df -> (Z [keep.sum(), dim] float32, keep [len(df)] bool)."""
        ...


@runtime_checkable
class RepertoireFeaturizer(Protocol):
    """One repertoire (clonotype df, optionally weighted by `count`) -> a single descriptor vector [D].

    Model-free featurizers (V-usage, k-mer) need a shared vocabulary across repertoires: call `fit(dfs)` once
    on the corpus, then `featurize(df)`. Encoder-backed featurizers (mean+cov) may need a fitted frame instead.
    `fit` is a no-op for featurizers that need no corpus state."""

    dim: int

    def fit(self, dfs) -> "RepertoireFeaturizer":
        """Learn any corpus-level state (vocab / whitening frame). Returns self. No-op if not needed."""
        ...

    def featurize(self, df: pd.DataFrame) -> np.ndarray:
        """Canonical clonotype df -> descriptor vector [dim] (float32, typically L2-normalized)."""
        ...


@runtime_checkable
class PairwiseDistance(Protocol):
    """Per-clonotype distance (e.g. in-house TCRdist). Not a vector space -> used via kNN / landmarks."""

    def pairwise(self, df: pd.DataFrame) -> np.ndarray:
        """Canonical clonotype df -> symmetric distance matrix D [N, N] (float32, zero diagonal)."""
        ...

    def to_reference(self, df: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
        """Distances from each row of `df` to each row of `ref` -> D [len(df), len(ref)]."""
        ...
