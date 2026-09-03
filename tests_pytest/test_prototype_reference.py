from pathlib import Path

import numpy as np
import pandas as pd

from tcr_foundation.prototype_reference import build_reference, publicness_stratum, stable_hash


def _cloud(path: Path, rows: list[tuple[str, str, str, float, float]]) -> None:
    frame = pd.DataFrame(rows, columns=["chain", "v_gene", "cdr3aa", "e0", "e1"])
    frame.to_parquet(path, index=False)


def test_hash_is_stable_and_publicness_bins_are_exhaustive():
    assert stable_hash("TRB\x1fTRBV1\x1fCASS") == stable_hash("TRB\x1fTRBV1\x1fCASS")
    assert [publicness_stratum(value) for value in (1, 2, 3, 4, 10, 11)] == [
        "private", "low_public", "low_public", "mid_public", "mid_public", "high_public"
    ]


def test_build_reference_counts_selected_identity_publicness(tmp_path: Path):
    clouds = tmp_path / "clouds"
    clouds.mkdir()
    _cloud(clouds / "P00001.parquet", [("TRB", "TRBV1", "AAA", 1.0, 0.0), ("TRB", "TRBV2", "BBB", 0.0, 1.0)])
    _cloud(clouds / "P00002.parquet", [("TRB", "TRBV1", "AAA", 1.0, 0.0), ("TRB", "TRBV3", "CCC", -1.0, 0.0)])
    _cloud(clouds / "P00003.parquet", [("TRB", "TRBV1", "AAA", 1.0, 0.0), ("TRB", "TRBV4", "DDD", 0.0, -1.0)])

    out = tmp_path / "reference"
    manifest = build_reference(clouds, out, clusters=2, hash_modulus=1, per_stratum=10, seed=7, minibatch_size=2, limit=3)

    audit = pd.read_parquet(out / "reference_candidates.parquet")
    aaa = audit.loc[(audit["v_gene"] == "TRBV1") & (audit["cdr3aa"] == "AAA")].iloc[0]
    assert int(aaa["n_donors"]) == 3
    assert aaa["publicness_stratum"] == "low_public"
    assert manifest["selected_identities"] == 4
    assert np.load(out / "prototypes.npz")["centroids"].shape == (2, 2)
