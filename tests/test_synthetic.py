"""
Synthetic end-to-end test -- the ONE test a colleague can run right after installing, with no data of ours.

The other three tests read `<repo>/data/processed/clouds/cd4cd8_sorted_TRB_seq` (closed cohort data), so they
only run here. This one generates its own donors, which also makes the expectations exact rather than
historical: the class signal is written into V-gene usage by construction, so V-usage MUST separate the two
groups, while the 3-mer featurizer sees CDR3s drawn from the same random process for both classes and MUST
stay at chance. That pair is the point -- a featurizer that "finds" signal in the k-mer arm is broken.

Run:  python tests/test_synthetic.py            (CPU, no network, no model)
      TCR_FOUNDATION_TEST_MODEL=joint-tiny python tests/test_synthetic.py    (adds the neural encoder arm)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # import tcr_foundation without installing

import tcr_foundation as tf                                                        # noqa: E402
from tcr_foundation import schema as S, featurizers, descriptors, metrics, registry  # noqa: E402

fails = []

N_DONORS = 24
N_CLONES = 400
V_GENES = ["TRBV5-1", "TRBV6-5", "TRBV19", "TRBV28", "TRBV20-1"]
AA = list("ACDEFGHIKLMNPQRSTVWY")
# Class 1 is TRBV5-1-heavy, class 0 is TRBV19-heavy; every other V gene carries no class information.
V_PROBS = {1: np.array([0.40, 0.15, 0.15, 0.15, 0.15]),
           0: np.array([0.10, 0.20, 0.30, 0.20, 0.20])}


def synth_donor(cls, rng):
    """One donor as a VDJtools-style table (column names deliberately NOT canonical -- ingest must map them)."""
    v = rng.choice(V_GENES, size=N_CLONES, p=V_PROBS[cls])
    cdr3 = ["CAS" + "".join(rng.choice(AA, size=rng.integers(6, 12))) + "F" for _ in range(N_CLONES)]
    return pd.DataFrame({"v": v, "j": "TRBJ2-7", "cdr3aa": cdr3, "count": rng.integers(1, 50, N_CLONES)})


rng = np.random.default_rng(0)
labels = np.array([i % 2 for i in range(N_DONORS)])
raw = [synth_donor(int(c), rng) for c in labels]

# ---- schema: column auto-detect -> canonical table ----
dfs = [S.ingest(r) for r in raw]
print(f"ingested {len(dfs)} synthetic donors x {N_CLONES} clonotypes; columns -> {list(dfs[0].columns)}")
if not set(S.REQUIRED).issubset(dfs[0].columns):
    fails.append(f"ingest did not produce required columns: {list(dfs[0].columns)}")
if len(dfs[0]) != N_CLONES:
    fails.append(f"ingest changed row count: {len(dfs[0])} != {N_CLONES}")

# ---- featurizers: V-usage must find the planted signal, k-mer must not ----
vu = featurizers.VUsage().fit(dfs)
Vv = np.stack([vu.featurize(d) for d in dfs])
auc_v = metrics.donor_centric_auroc(Vv, labels)
print(f"VUsage  dim={vu.dim:<6} donor-AUROC = {auc_v:.3f}   (planted signal -> expect > 0.9)")
if not (Vv.shape == (N_DONORS, vu.dim) and auc_v > 0.9):
    fails.append(f"VUsage AUROC {auc_v:.3f} (<0.9) or shape {Vv.shape}")

km = featurizers.Kmer(k=3).fit(dfs)
Vk = np.stack([km.featurize(d) for d in dfs])
auc_k = metrics.donor_centric_auroc(Vk, labels)
print(f"Kmer(3) dim={km.dim:<6} donor-AUROC = {auc_k:.3f}   (no CDR3 signal by construction -> expect ~0.5)")
if not (Vk.shape == (N_DONORS, km.dim) and 0.3 < auc_k < 0.7):
    fails.append(f"Kmer AUROC {auc_k:.3f} outside the chance band (0.3, 0.7) or shape {Vk.shape}")

# ---- metrics: the reference-size sweep returns one AUROC per requested size ----
sweep = metrics.ref_size_sweep_auroc(Vv, labels, [1, 3, 5])
print(f"ref_size_sweep_auroc(VUsage) = { {k: round(v, 3) for k, v in sweep.items()} }")
if set(sweep) != {1, 3, 5} or not all(np.isfinite(list(sweep.values()))):
    fails.append(f"ref_size_sweep_auroc returned {sweep}")


# ---- descriptors: wiring and dimensions, with a mock encoder so no weights are needed ----
class MockEncoder:
    """Deterministic pseudo-embeddings: tests descriptor shapes/finiteness, not signal."""
    dim = 16

    def encode(self, df):
        r = np.random.RandomState(len(df))
        return r.randn(len(df), self.dim).astype(np.float32), np.ones(len(df), bool)


from repertoire.rep_descriptors import mean_cov_dim                      # noqa: E402  (vendored, expected dim)
exp_dim = mean_cov_dim(MockEncoder.dim)
d_mc = descriptors.MeanCov(MockEncoder()).featurize(dfs[0])
d_wv = descriptors.WithinV(MockEncoder()).featurize(dfs[0])
print(f"MeanCov dim={d_mc.shape[0]} (expect {exp_dim})   WithinV dim={d_wv.shape[0]}   finite={np.isfinite(d_mc).all()}")
if not (d_mc.shape[0] == exp_dim and d_wv.shape[0] == exp_dim and np.isfinite(d_mc).all() and np.isfinite(d_wv).all()):
    fails.append(f"MeanCov/WithinV dim or finiteness: {d_mc.shape}, {d_wv.shape}")

# ---- registry: the catalogue is readable without touching the network ----
rows = registry.catalog()
print("registry:", ", ".join(f"{r['name']}={r['available']}" for r in rows))
if not rows or "joint-tiny" not in {r["name"] for r in rows}:
    fails.append("registry catalogue missing joint-tiny")
if {r["available"] for r in rows} - {"local", "hf", "hf-private", "MISSING"}:
    fails.append(f"unexpected availability values: {[r['available'] for r in rows]}")

# ---- optional: the real encoder, only when a model is named (downloads weights / needs torch) ----
model = os.environ.get("TCR_FOUNDATION_TEST_MODEL")
if model:
    enc = tf.load(model)
    Z, keep = enc.encode(dfs[0])
    print(f"{model}: encode -> Z{Z.shape}, kept {int(keep.sum())}/{len(dfs[0])}")
    if Z.shape[0] != int(keep.sum()) or not np.isfinite(Z).all():
        fails.append(f"encoder returned Z{Z.shape} with keep={int(keep.sum())}")
    Vm = np.stack([descriptors.MeanCov(enc).featurize(d) for d in dfs])
    auc_m = metrics.donor_centric_auroc(Vm, labels)
    print(f"{model}: MeanCov donor-AUROC = {auc_m:.3f}  (V-usage is planted, so expect high)")
    if auc_m < 0.8:
        fails.append(f"{model} MeanCov AUROC {auc_m:.3f} (<0.8)")
else:
    print("skipped the neural arm (set TCR_FOUNDATION_TEST_MODEL=joint-tiny to run it)")

print("\nSYNTHETIC_OK" if not fails else f"SYNTHETIC_FAIL: {fails}")
sys.exit(0 if not fails else 1)
