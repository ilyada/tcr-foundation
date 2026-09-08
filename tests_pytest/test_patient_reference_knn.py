import pandas as pd

from tcr_foundation.patient_reference_knn import evaluate_hla


def _descriptor(path, samples, values):
    pd.DataFrame({"sample": samples, "d0": [item[0] for item in values], "d1": [item[1] for item in values]}).to_parquet(path, index=False)


def _tsv(path, sample, alleles, field="sample_rich_tags"):
    tags = ",".join(f"HLA MHC class I:HLA-{allele[0]}*{allele[1:]}" for allele in alleles)
    pd.DataFrame({"sample_name": [sample], field: [tags]}).to_csv(path, sep="\t", index=False)


def test_hla_evaluation_reuses_references_across_branches_and_resolutions(tmp_path):
    samples = [f"P{i:05d}" for i in range(1, 7)]
    descriptors, source, output = tmp_path / "descriptors", tmp_path / "source", tmp_path / "out"
    descriptors.mkdir(); source.mkdir()
    raw = [(1, 0), (0.9, 0.1), (0.8, 0.2), (0, 1), (0.1, 0.9), (0.2, 0.8)]
    oar = [(0.95, 0.05), (0.85, 0.15), (0.75, 0.25), (0.05, 0.95), (0.15, 0.85), (0.25, 0.75)]
    for clusters in (2, 3):
        _descriptor(descriptors / f"occupancy_k{clusters}_raw.parquet", samples, raw)
        _descriptor(descriptors / f"occupancy_k{clusters}_oar.parquet", samples, oar)
    for index, (sample, alleles) in enumerate(zip(samples, [("A01",), ("A01",), ("A01",), ("B07",), ("B07",), ("B07",)])):
        _tsv(source / f"{sample}.tsv", sample, alleles, field="sample_catalog_tags" if index == 0 else "sample_rich_tags")
    summary, macro = evaluate_hla(descriptors, source, output, clusters=(2, 3), seed=7)
    draws = pd.read_parquet(output / "hla_draws.parquet")
    refs = pd.read_parquet(output / "reference_draws.parquet")
    assert set(summary["clusters"]) == {2, 3}
    assert set(macro["reference_size"]) == {1, 2}
    assert draws.groupby(["target", "reference_size", "draw"])[["raw_auroc", "oar_auroc"]].size().eq(2).all()
    assert not refs.empty
    assert (output / "hla_knn_auroc_raw.png").exists()
    assert (output / "hla_knn_auroc_oar.png").exists()
