import numpy as np
import pandas as pd
import pytest

from tcr_foundation.oar import correct_productive_weights, estimate_oar, process_patient, read_patient, run


def _nonproductive():
    return pd.DataFrame({
        "sample": ["S1", "S1", "S2", "S2"],
        "chain": ["TRB"] * 4,
        "frame_type": ["out", "stop", "out", "stop"],
        "v_gene": ["V1", "V2", "V1", "V2"],
        "j_gene": ["J1", "J2", "J1", "J2"],
        "rearrangement": ["A", "B", "C", "D"],
        "templates": [40, 10, 10, 40],
    })


def _productive():
    return pd.DataFrame({
        "sample": ["S1", "S1", "S2", "S2"],
        "chain": ["TRB"] * 4,
        "frame_type": ["in"] * 4,
        "v_gene": ["V1", "V2", "V1", "V2"],
        "j_gene": ["J1", "J2", "J1", "J2"],
        "rearrangement": ["P1", "P2", "P3", "P4"],
        "templates": [160, 40, 40, 160],
    })


def test_oar_is_estimated_per_sample_and_corrects_productive_weights():
    factors = estimate_oar(_nonproductive(), min_unique_clonotypes=1)
    s1 = factors.loc[(factors["sample"] == "S1") & (factors["gene_axis"] == "V")].set_index("gene")
    assert s1.loc["V1", "oar"] == pytest.approx(1.6)
    assert s1.loc["V2", "oar"] == pytest.approx(0.4)
    corrected = correct_productive_weights(_productive(), factors)
    s1_weights = corrected.loc[corrected["sample"] == "S1"].set_index("rearrangement")
    assert s1_weights.loc["P1", "templates_oar"] == pytest.approx(62.5)
    assert s1_weights.loc["P2", "templates_oar"] == pytest.approx(250.0)
    assert s1_weights.weight_raw.sum() == pytest.approx(1.0)
    assert s1_weights.weight_oar.sum() == pytest.approx(1.0)
    assert not np.isclose(s1_weights.loc["P1", "weight_raw"], s1_weights.loc["P1", "weight_oar"])


def test_oar_uses_neutral_factor_for_productive_gene_without_a_nonproductive_factor():
    factors = estimate_oar(_nonproductive(), min_unique_clonotypes=1)
    productive = _productive()
    productive.loc[0, "v_gene"] = "V3"
    corrected = correct_productive_weights(productive, factors)
    row = corrected.iloc[0]
    assert row["oar_v"] == pytest.approx(1.0)
    assert row["oar_v_source"] == "absent_nonproductive"


def test_oar_uses_neutral_factor_for_an_observed_but_sparse_segment():
    nonproductive = _nonproductive().copy()
    nonproductive.loc[len(nonproductive)] = ["S1", "TRB", "out", "V3", "J1", "E", 5]
    factors = estimate_oar(nonproductive, min_unique_clonotypes=2)
    factor = factors.loc[(factors["sample"] == "S1") & (factors["gene_axis"] == "V") & (factors["gene"] == "V3")].iloc[0]
    assert factor["oar"] == pytest.approx(1.0)
    assert factor["oar_source"] == "insufficient_nonproductive"


def test_oar_run_writes_parallel_factors_and_weights(tmp_path):
    nonproductive = tmp_path / "nonproductive.parquet"
    productive = tmp_path / "productive.parquet"
    _nonproductive().to_parquet(nonproductive, index=False)
    _productive().to_parquet(productive, index=False)
    factors, weights = run(nonproductive, productive, tmp_path / "out", min_unique_clonotypes=1)
    assert factors.name == "oar_factors.parquet" and factors.exists()
    written = pd.read_parquet(weights)
    assert {"templates_raw", "templates_oar", "weight_raw", "weight_oar"} <= set(written.columns)


def test_process_patient_combines_out_and_stop_then_returns_model_ready_weights():
    events = pd.concat((_nonproductive(), _productive()), ignore_index=True)
    events.loc[events["frame_type"] == "in", "rearrangement"] = ["TGTGCTTCTTCT", "TGTGCTTCTTCC", "TGTGCTTCTTCA", "TGTGCTTCTTCG"]
    factors, corrected = process_patient(events, min_unique_clonotypes=1)
    assert set(factors["gene_axis"]) == {"V", "J"}
    assert corrected["cdr3aa"].tolist() == ["CASS", "CASS", "CASS", "CASS"]
    assert corrected["count"].tolist() == pytest.approx([62.5, 250.0, 250.0, 62.5])
    assert corrected.groupby(["sample", "chain"])["w_log"].sum().tolist() == pytest.approx([1.0, 1.0])


def test_process_patient_audits_absent_productive_gene_with_neutral_factor():
    events = pd.concat((_nonproductive(), _productive()), ignore_index=True)
    events.loc[events["frame_type"] == "in", "rearrangement"] = ["TGTGCTTCTTCT", "TGTGCTTCTTCC", "TGTGCTTCTTCA", "TGTGCTTCTTCG"]
    events.loc[(events["sample"] == "S1") & (events["frame_type"] == "in") & (events["v_gene"] == "V1"), "v_gene"] = "V3"
    factors, corrected = process_patient(events, min_unique_clonotypes=1)
    assert corrected.loc[(corrected["sample"] == "S1") & (corrected["v_gene"] == "V3"), "oar_v_source"].iloc[0] == "absent_nonproductive"
    audit = factors.loc[(factors["sample"] == "S1") & (factors["gene_axis"] == "V") & (factors["gene"] == "V3")].iloc[0]
    assert audit["oar"] == pytest.approx(1.0)
    assert audit["oar_source"] == "absent_nonproductive"


def test_read_patient_accepts_mixcr_tsv(tmp_path):
    mixcr = tmp_path / "patient.tsv"
    ref_points = ":".join([""] * 19)
    mixcr.write_text(
        "nSeqCDR3\taaSeqCDR3\tallVHitsWithScore\tallDHitsWithScore\tallJHitsWithScore\trefPoints\tuniqueMoleculeCount\n"
        f"AACCGGTT\tCA_G\tTRBV1*01(1)\tTRBD1*01(1)\tTRBJ1-1*01(1)\t{ref_points}\t4\n"
        f"TTGGCCAA\tCA*G\tTRBV2*01(1)\tTRBD1*01(1)\tTRBJ1-2*01(1)\t{ref_points}\t2\n"
        f"TGTGCTTCTTCT\tCASS\tTRBV1*01(1)\tTRBD1*01(1)\tTRBJ1-1*01(1)\t{ref_points}\t8\n"
    )
    frame = read_patient(mixcr)
    assert frame["frame_type"].astype(str).tolist() == ["out", "stop", "in"]
    assert frame["templates"].tolist() == [4, 2, 8]
