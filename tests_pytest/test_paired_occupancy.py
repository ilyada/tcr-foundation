from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from tcr_foundation.paired_occupancy import build_paired_occupancies


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
