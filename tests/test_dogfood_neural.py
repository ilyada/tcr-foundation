"""
Dogfood the package NeuralEncoder end-to-end (needs GPU): registry.load -> encode -> MeanCov/WithinV ->
donor-AUROC on CD4/CD8, and compare to the proven benchmark_ram "model mean+cov" number for the same task.
If they match, the isolated package reproduces the trusted pipeline through its own protocol layer.

Run (after the GPU is free): python tcr_foundation/tests/test_dogfood_neural.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PKG_DIR)
sys.path.insert(0, _PKG_DIR)

import tcr_foundation  # noqa: E402
from tcr_foundation import schema as S, descriptors, metrics, registry  # noqa: E402

CLOUD_DIR = os.path.join(_REPO, "data", "processed", "clouds", "cd4cd8_sorted_TRB_seq")
N_FILES = 60
CAP = 2000  # match benchmark_ram default cap


def load_labeled(n):
    import glob
    files = sorted(glob.glob(os.path.join(CLOUD_DIR, "*.parquet")))
    files = [f for f in files if ("CD4" in os.path.basename(f) or "CD8" in os.path.basename(f))][:n]
    dfs, labels = [], []
    for f in files:
        df = S.read(f)
        if len(df) == 0:
            continue
        if len(df) > CAP:
            df = df.sample(CAP, random_state=0).reset_index(drop=True)
        dfs.append(df); labels.append(1 if "CD8" in os.path.basename(f) else 0)
    return dfs, np.array(labels)


enc = registry.load("joint-vtoken-tiny")   # NeuralEncoder, auto-detects vtoken
print(f"NeuralEncoder loaded: input_format={enc.input_format}, dim={enc.dim}, device={enc.device}")

dfs, y = load_labeled(N_FILES)
print(f"loaded {len(dfs)} CD4/CD8 clouds (CD8={int(y.sum())}, CD4={int((y==0).sum())}), cap={CAP}")

mc = descriptors.MeanCov(enc)
wv = descriptors.WithinV(enc)
Vmc = np.stack([mc.featurize(d) for d in dfs])
Vwv = np.stack([wv.featurize(d) for d in dfs])
auc_mc = metrics.donor_centric_auroc(Vmc, y)
auc_wv = metrics.donor_centric_auroc(Vwv, y)

print(f"\n== package NeuralEncoder -> CD4/CD8 donor-AUROC ==")
print(f"  model mean+cov : {auc_mc:.3f}")
print(f"  model within-V : {auc_wv:.3f}")
print("Compare 'model mean+cov' to benchmark_ram's CD4/CD8 'model mean+cov' -- should match closely.")
print("DOGFOOD_DONE")
