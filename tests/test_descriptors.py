"""CPU test for the added descriptors (Occupancy landmark + WhitenedSPD) via a mock encoder -- checks fit/
featurize wiring, dims, and finiteness on real clouds. (Signal isn't tested here: a random mock encoder can't
separate CD4/CD8; the neural dogfood covers real signal.)"""
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PKG_DIR)
sys.path.insert(0, _PKG_DIR)

import tcr_foundation  # noqa: E402
from tcr_foundation import schema as S, descriptors  # noqa: E402
from repertoire.rep_descriptors import mean_cov_dim   # noqa: E402

CLOUD_DIR = os.path.join(_REPO, "data", "processed", "clouds", "cd4cd8_sorted_TRB_seq")
fails = []


class MockEncoder:
    dim = 16

    def encode(self, df):
        rng = np.random.RandomState(len(df))
        return rng.randn(len(df), self.dim).astype(np.float32), np.ones(len(df), bool)


files = sorted(glob.glob(os.path.join(CLOUD_DIR, "*.parquet")))[:12]
dfs = [S.read(f) for f in files]
dfs = [d for d in dfs if len(d) > 0]
print(f"loaded {len(dfs)} clouds")
enc = MockEncoder()

occ = descriptors.Occupancy(enc, k=32).fit(dfs)
vo = occ.featurize(dfs[0])
print(f"Occupancy   dim={vo.shape[0]} (expect 32)  finite={np.isfinite(vo).all()}  sum={vo.sum():.3f}")
if not (vo.shape[0] == 32 and np.isfinite(vo).all()):
    fails.append("Occupancy dim/finite")

wsp = descriptors.WhitenedSPD(enc).fit(dfs)
vw = wsp.featurize(dfs[0])
exp = mean_cov_dim(16)   # full m1=k=16 -> 152
print(f"WhitenedSPD dim={vw.shape[0]} (expect {exp})  finite={np.isfinite(vw).all()}  fitted_dim={wsp.dim}")
if not (vw.shape[0] == exp and np.isfinite(vw).all() and wsp.dim == exp):
    fails.append("WhitenedSPD dim/finite")

print("\nDESCRIPTORS_OK" if not fails else f"DESCRIPTORS_FAIL: {fails}")
sys.exit(0 if not fails else 1)
