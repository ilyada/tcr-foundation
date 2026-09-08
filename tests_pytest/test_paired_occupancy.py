from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from tcr_foundation.paired_occupancy import build_paired_occupancies
from tcr_foundation.cloud_descriptors import build_paired_descriptors


def _cloud(path: Path, weights: list[float], *, reverse: bool = False, embedding_shift: float = 0.0) -> None:
    frame = pd.DataFrame({
        "sample": ["P00001", "P00001"],
        "chain": ["TRB", "TRB"],
        "v_gene": ["TRBV1", "TRBV2"],
        "j_gene": ["TRBJ1-1", "TRBJ1-2"],
        "cdr3aa": ["CASSA", "CASSB"],
        "w_log": weights,
        "e0": [1.0 + embedding_shift, 0.0],
        "e1": [0.0, 1.0],
    })
    if reverse:
        frame = frame.iloc[::-1].reset_index(drop=True)
    frame.to_parquet(path, index=False)


def _codebook(path: Path) -> None:
    path.mkdir()
    np.savez(path / "prototypes.npz", centroids=np.eye(2, dtype=np.float32), clusters=2, seed=0, temperature=0.1)
    (path / "manifest.json").write_text(json.dumps({"clusters": 2, "selected_identities": 4, "embedding_dim": 2}), encoding="utf-8")


def test_paired_occupancy_uses_shared_assignments_and_aligns_row_order(tmp_path: Path):
    raw, oar, codebooks = tmp_path / "raw", tmp_path / "oar", tmp_path / "codebooks"
    raw.mkdir(); oar.mkdir(); codebooks.mkdir()
    _cloud(raw / "P00001.parquet", [0.5, 0.5])
    _cloud(oar / "P00001.parquet", [0.8, 0.2], reverse=True)
    _codebook(codebooks / "k2")

    out = tmp_path / "out"
    manifest = build_paired_occupancies(raw, oar, [codebooks / "k2"], out, chunk_size=1)

    raw_descriptor = pd.read_parquet(out / "occupancy_k2_raw.parquet").iloc[0]
    oar_descriptor = pd.read_parquet(out / "occupancy_k2_oar.parquet").iloc[0]
    qc = pd.read_parquet(out / "occupancy_k2_pair_qc.parquet").iloc[0]
    assert manifest["n_pairs"] == 1
    assert raw_descriptor[["d0", "d1"]].sum() == pytest.approx(1.0)
    assert oar_descriptor[["d0", "d1"]].sum() == pytest.approx(1.0)
    assert oar_descriptor["d0"] > raw_descriptor["d0"]
    assert qc["max_embedding_abs_diff"] == pytest.approx(0.0)
    assert qc["raw_oar_l1"] > 0


def test_paired_occupancy_rejects_embedding_mismatch(tmp_path: Path):
    raw, oar, codebooks = tmp_path / "raw", tmp_path / "oar", tmp_path / "codebooks"
    raw.mkdir(); oar.mkdir(); codebooks.mkdir()
    _cloud(raw / "P00001.parquet", [0.5, 0.5])
    _cloud(oar / "P00001.parquet", [0.5, 0.5], embedding_shift=0.1)
    _codebook(codebooks / "k2")

    with pytest.raises(ValueError, match="embedding mismatch"):
        build_paired_occupancies(raw, oar, [codebooks / "k2"], tmp_path / "out")


def test_paired_occupancy_retains_repeated_clonotype_rows(tmp_path: Path):
    raw, oar, codebooks = tmp_path / "raw", tmp_path / "oar", tmp_path / "codebooks"
    raw.mkdir(); oar.mkdir(); codebooks.mkdir()
    base = {
        "sample": ["P00001", "P00001"], "chain": ["TRB", "TRB"], "v_gene": ["TRBV1", "TRBV1"],
        "j_gene": ["TRBJ1-1", "TRBJ1-1"], "cdr3aa": ["CASSA", "CASSA"], "e0": [1.0, 1.0], "e1": [0.0, 0.0],
    }
    pd.DataFrame({**base, "w_log": [0.6, 0.4]}).to_parquet(raw / "P00001.parquet", index=False)
    pd.DataFrame({**base, "w_log": [0.3, 0.7]}).to_parquet(oar / "P00001.parquet", index=False)
    _codebook(codebooks / "k2")

    build_paired_occupancies(raw, oar, [codebooks / "k2"], tmp_path / "out")
    descriptor = pd.read_parquet(tmp_path / "out" / "occupancy_k2_raw.parquet").iloc[0]
    assert int(descriptor["n_clonotypes"]) == 2


def test_generic_builder_writes_moments_and_occupancy_without_default_qc(tmp_path: Path):
    raw, oar, codebooks = tmp_path / "raw", tmp_path / "oar", tmp_path / "codebooks"
    raw.mkdir(); oar.mkdir(); codebooks.mkdir()
    _cloud(raw / "P00001.parquet", [0.5, 0.5])
    _cloud(oar / "P00001.parquet", [0.8, 0.2], reverse=True)
    _codebook(codebooks / "k2")

    out = tmp_path / "out"
    manifest = build_paired_descriptors(raw, oar, out, kinds=("mean", "mean_cov", "occupancy"), codebook_dirs=[codebooks / "k2"], chunk_size=1)

    assert manifest["kinds"] == ["mean", "mean_cov", "occupancy"]
    assert not (out / "pair_qc.parquet").exists()
    assert len([name for name in pd.read_parquet(out / "mean_raw.parquet").columns if name.startswith("d")]) == 2
    assert len([name for name in pd.read_parquet(out / "mean_cov_raw.parquet").columns if name.startswith("d")]) == 5
    assert (out / "occupancy_k2_raw.parquet").exists()
