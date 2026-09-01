"""
CPU smoke for tcr_foundation (no GPU, no install): exercises schema -> featurizers -> descriptors -> metrics
on real CD4/CD8 sequence clouds, and the registry. Proves the isolated package wires together and that the
model-free numbers reproduce the known signal (V-usage separates CD4/CD8 ~0.9 per the KB).

Run: python tcr_foundation/tests/test_smoke_cpu.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)               # <repo>/tcr_foundation
_REPO = os.path.dirname(_PKG_DIR)               # <repo>
sys.path.insert(0, _PKG_DIR)                    # import tcr_foundation without installing

import tcr_foundation                                            # noqa: E402
from tcr_foundation import schema as S, featurizers, descriptors, metrics, registry  # noqa: E402

CLOUD_DIR = os.path.join(_REPO, "data", "processed", "clouds", "cd4cd8_sorted_TRB_seq")
N_FILES = 40
fails = []


def load_labeled(cloud_dir, n):
    import glob
    files = sorted(glob.glob(os.path.join(cloud_dir, "*.parquet")))
    files = [f for f in files if ("CD4" in os.path.basename(f) or "CD8" in os.path.basename(f))][:n]
    dfs, labels = [], []
    for f in files:
        df = S.read(f)                                  # ingest -> canonical (v_gene, cdr3, count, ...)
        if len(df) == 0:
            continue
        dfs.append(df)
        labels.append(1 if "CD8" in os.path.basename(f) else 0)
    return dfs, np.array(labels)


# ---- schema + featurizers on real clouds ----
dfs, y = load_labeled(CLOUD_DIR, N_FILES)
print(f"loaded {len(dfs)} CD4/CD8 clouds  (CD8={int(y.sum())}, CD4={int((y==0).sum())})")
assert len(dfs) >= 10 and 0 < y.sum() < len(y), "need both classes"
assert set(S.REQUIRED).issubset(dfs[0].columns), f"schema missing required cols: {dfs[0].columns.tolist()}"

vu = featurizers.VUsage().fit(dfs)
Vv = np.stack([vu.featurize(d) for d in dfs])
auc_v = metrics.donor_centric_auroc(Vv, y)
print(f"VUsage  dim={vu.dim:<5} CD4/CD8 donor-AUROC = {auc_v:.3f}   (expect high, ~0.85-0.95)")
if not (Vv.shape == (len(dfs), vu.dim) and auc_v > 0.75):
    fails.append(f"VUsage AUROC {auc_v:.3f} (<0.75) or shape {Vv.shape}")

km = featurizers.Kmer(k=3).fit(dfs)
Vk = np.stack([km.featurize(d) for d in dfs])
auc_k = metrics.donor_centric_auroc(Vk, y)
print(f"Kmer    dim={km.dim:<5} CD4/CD8 donor-AUROC = {auc_k:.3f}   (expect > 0.6)")
if not (Vk.shape == (len(dfs), km.dim) and auc_k > 0.6):
    fails.append(f"Kmer AUROC {auc_k:.3f} (<0.6) or shape {Vk.shape}")


# ---- encoder-backed descriptors with a MOCK encoder (no GPU) ----
class MockEncoder:
    """A ClonotypeEncoder that returns deterministic pseudo-embeddings -> tests the descriptor wiring."""
    dim = 16

    def encode(self, df):
        rng = np.random.RandomState(len(df))
        Z = rng.randn(len(df), self.dim).astype(np.float32)
        return Z, np.ones(len(df), bool)


mc = descriptors.MeanCov(MockEncoder())
wv = descriptors.WithinV(MockEncoder())
dv_mc = mc.featurize(dfs[0]); dv_wv = wv.featurize(dfs[0])
from repertoire.rep_descriptors import mean_cov_dim   # expected dim
exp_dim = mean_cov_dim(16)
print(f"MeanCov dim={dv_mc.shape[0]} (expect {exp_dim})  WithinV dim={dv_wv.shape[0]}  finite={np.isfinite(dv_mc).all()}")
if not (dv_mc.shape[0] == exp_dim and dv_wv.shape[0] == exp_dim and np.isfinite(dv_mc).all()):
    fails.append("MeanCov/WithinV dim or finiteness")


# ---- registry ----
assert "joint-tiny" in registry.REGISTRY, "joint-tiny model not registered"
print(f"registry models_root = {registry.models_root()}")
try:
    p = registry.resolve("joint-tiny")
    print(f"registry resolve('joint-tiny') -> {p}  (exists={os.path.isdir(p)})")
except FileNotFoundError as e:
    print(f"registry resolve -> not on disk yet: {e}")   # fine if the checkpoint isn't there

print("\nSMOKE_CPU_OK" if not fails else f"SMOKE_CPU_FAIL: {fails}")
sys.exit(0 if not fails else 1)
