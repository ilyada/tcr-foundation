"""
rep_metrics.py -- canonical donor-centric retrieval metrics for repertoire-level analysis.

ONE documented implementation of each metric, so numbers are comparable across every script/notebook (the earlier
drift -- e.g. identity AUROC 0.784 vs 0.886 -- came from re-implementing the same-named metric differently).

Metric definitions (fixed here):
  - donor_centric_auroc: leave-one-out. For each item i, rank all others by cosine to i; y = "other shares i's
    label"; ROC-AUC; macro-average over items. One number. Each label needs >=2 members.
  - ref_size_sweep_auroc: for a reference-SET size r, sample r items of label L, score every other item by the
    MAX cosine to that set, AUROC of "item is L"; average over draws and labels. Returns {r: macro_auroc}. This is
    the reference-set-size sweep used in the FMBA/vaccine/Rosati work.
Both use cosine similarity on L2-normalized descriptors.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _cosine_gram(V):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    return Vn @ Vn.T


def donor_centric_auroc(V, labels):
    """Leave-one-out macro donor-centric AUROC (see module docstring). Returns a float."""
    S = _cosine_gram(V)
    np.fill_diagonal(S, -np.inf)
    labels = np.asarray(labels)
    aucs = []
    for i in range(len(V)):
        y = (labels == labels[i]).astype(int)
        y[i] = 0
        if 0 < y.sum() < len(y) - 1:
            m = np.ones(len(V), bool); m[i] = False
            aucs.append(roc_auc_score(y[m], S[i][m]))
    return float(np.mean(aucs)) if aucs else float("nan")


def ref_size_sweep_auroc(V, labels, ref_sizes, n_draws=20, seed=0):
    """Macro donor-centric AUROC by reference-SET size (see module docstring). Returns {r: macro_auroc}."""
    S = _cosine_gram(V)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    rng = np.random.RandomState(seed)
    out = {}
    for r in ref_sizes:
        lab = []
        for L in uniq:
            pos = np.where(labels == L)[0]
            if len(pos) <= r:
                continue
            draws = []
            for _ in range(n_draws):
                refs = rng.choice(pos, size=r, replace=False)
                mask = np.ones(len(V), bool); mask[refs] = False
                score = S[np.ix_(np.where(mask)[0], refs)].max(1)
                y = (labels[mask] == L).astype(int)
                if 0 < y.sum() < len(y):
                    draws.append(roc_auc_score(y, score))
            if draws:
                lab.append(np.mean(draws))
        out[r] = float(np.mean(lab)) if lab else float("nan")
    return out


def vusage_vector(v_genes, weights, vocab):
    """Weighted V-gene usage vector for one repertoire, over a fixed vocab {gene: index}."""
    vv = np.zeros(len(vocab), np.float32)
    for v, wi in zip(v_genes, weights):
        j = vocab.get(str(v))
        if j is not None:
            vv[j] += wi
    return vv


def build_vgene_vocab(v_gene_iter):
    """Build {gene: index} from an iterable of per-repertoire v_gene arrays."""
    vocab = {}
    for vgs in v_gene_iter:
        for v in np.unique(np.asarray(vgs).astype(str)):
            vocab.setdefault(v, len(vocab))
    return vocab


def build_kmer_vocab(cdr3_iter, k=3):
    """Build {kmer: index} of all length-k amino-acid substrings seen across an iterable of cdr3aa arrays."""
    vocab = {}
    for arr in cdr3_iter:
        for s in np.asarray(arr).astype(str):
            for i in range(len(s) - k + 1):
                vocab.setdefault(s[i:i + k], len(vocab))
    return vocab


def kmer_vector(cdr3s, weights, vocab, k=3):
    """Frequency-weighted CDR3 k-mer distribution for one repertoire (a clonotype spreads its weight over its
    k-mers). Returns a vector that sums to ~sum(weights); take sqrt for the Hellinger metric embedding."""
    v = np.zeros(len(vocab), np.float32)
    for s, wi in zip(np.asarray(cdr3s).astype(str), weights):
        kms = [s[i:i + k] for i in range(len(s) - k + 1)]
        if not kms:
            continue
        m = wi / len(kms)
        for g in kms:
            j = vocab.get(g)
            if j is not None:
                v[j] += m
    return v
