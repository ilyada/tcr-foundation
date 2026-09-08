"""Paired patient-reference kNN--AUROC evaluation of cloud descriptors.

The evaluator holds each sampled positive reference set fixed across raw/OAR
descriptors and prototype resolutions.  It therefore measures descriptor
differences rather than Monte-Carlo differences in reference selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REFERENCE_SIZES = (1, 2, 5, 10, 20, 50)
N_DRAWS = {1: 200, 2: 200, 5: 100, 10: 50, 20: 30, 50: 20}
HLA_TAG = re.compile(r"(?:^|,)HLA MHC class I:HLA-([A-Z]+)\*([0-9]+)(?:,|$)")
CMV_TAG = re.compile(r"(?:^|,)Virus Diseases:Cytomegalovirus ([+-])(?:,|$)")


def _descriptor_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted((column for column in frame.columns if re.fullmatch(r"d\d+", column)), key=lambda name: int(name[1:]))
    if not columns:
        raise ValueError("descriptor table contains no d0, d1, ... columns")
    return columns


def _read_first_record(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"), None)
    if row is None:
        raise ValueError(f"{path}: no data row")
    return row


def parse_emerson_hla(source_dir: str | Path, samples: list[str]) -> pd.DataFrame:
    """Extract known class-I HLA calls from the first record of each Emerson TSV."""
    source = Path(source_dir)
    rows = []
    for sample in samples:
        path = source / f"{sample}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"descriptor sample {sample} has no corresponding TSV under {source}")
        record = _read_first_record(path)
        tags = ",".join(str(record.get(column) or "") for column in ("sample_catalog_tags", "sample_rich_tags", "sample_tags"))
        alleles = sorted({f"{locus}{field}" for locus, field in HLA_TAG.findall(tags)})
        cmv_calls = set(CMV_TAG.findall(tags))
        if len(cmv_calls) > 1:
            raise ValueError(f"{path}: conflicting known CMV calls")
        cmv_status = {"+": "positive", "-": "negative"}.get(next(iter(cmv_calls), None), "missing")
        rows.append({"sample": sample, "hla_alleles": alleles, "n_hla_alleles": len(alleles), "included_hla_eval": bool(alleles), "cmv_status": cmv_status, "included_cmv_eval": cmv_status != "missing"})
    return pd.DataFrame(rows)


def _load_pair(descriptor_dir: Path, clusters: int, pattern: str = "occupancy_k{clusters}_{branch}.parquet") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load one raw/OAR descriptor pair through a named resolution/branch pattern."""
    try:
        raw_path = descriptor_dir / pattern.format(clusters=clusters, branch="raw")
        oar_path = descriptor_dir / pattern.format(clusters=clusters, branch="oar")
    except KeyError as exc:
        raise ValueError("descriptor pattern may contain only {clusters} and {branch}") from exc
    raw = pd.read_parquet(raw_path)
    oar = pd.read_parquet(oar_path)
    if raw["sample"].duplicated().any() or oar["sample"].duplicated().any():
        raise ValueError(f"K={clusters}: descriptor sample identifiers must be unique")
    raw = raw.sort_values("sample").reset_index(drop=True)
    oar = oar.sort_values("sample").reset_index(drop=True)
    if not raw["sample"].equals(oar["sample"]):
        raise ValueError(f"K={clusters}: raw and OAR descriptor sample sets differ")
    columns = _descriptor_columns(raw)
    if columns != _descriptor_columns(oar):
        raise ValueError(f"K={clusters}: raw and OAR descriptor dimensions differ")
    return raw, oar, columns


def _cosine_gram(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero-norm descriptor encountered")
    values = values / norm
    return values @ values.T


def _reference_draws(labels: np.ndarray, samples: np.ndarray, seed: int) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    draw_specs: list[dict] = []
    audit_rows = []
    for target in sorted(np.unique(labels)):
        y = labels == target
        positive = np.flatnonzero(y)
        for reference_size in REFERENCE_SIZES:
            eligible = len(positive) >= reference_size + 1 and len(positive) < len(labels)
            audit_rows.append({"target": target, "reference_size": reference_size, "n_positive": int(len(positive)), "n_negative": int(len(labels) - len(positive)), "n_draws_requested": N_DRAWS[reference_size], "evaluable": eligible})
            if not eligible:
                continue
            for draw in range(N_DRAWS[reference_size]):
                reference_indices = np.sort(rng.choice(positive, size=reference_size, replace=False))
                draw_specs.append({"target": target, "reference_size": reference_size, "draw": draw, "reference_indices": reference_indices})
    return draw_specs, pd.DataFrame(audit_rows)


def _draw_macro_figure(draws: pd.DataFrame, branch: str, output: Path, target_name: str, descriptor_name: str, resolution_labels: dict[int, str] | None = None) -> None:
    """Draw macro AUROC curves with central 95% draw intervals for one target family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column = f"{branch}_auroc"
    macro_draws = draws.groupby(["clusters", "reference_size", "draw"], as_index=False)[column].mean()
    clusters = sorted(macro_draws["clusters"].unique())
    n_columns = min(2, len(clusters))
    n_rows = int(np.ceil(len(clusters) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 3.5 * n_rows), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for axis, clusters_value in zip(axes.flat, clusters):
        subset = macro_draws[macro_draws["clusters"] == clusters_value]
        summary = subset.groupby("reference_size")[column].agg(mean="mean", low=lambda values: values.quantile(0.025), high=lambda values: values.quantile(0.975)).reset_index().sort_values("reference_size")
        x = summary["reference_size"].to_numpy()
        axis.plot(x, summary["mean"], marker="o", color="#2b6cb0", linewidth=2)
        axis.fill_between(x, summary["low"], summary["high"], color="#2b6cb0", alpha=0.2, linewidth=0)
        axis.set_xscale("log")
        axis.set_xticks(REFERENCE_SIZES)
        axis.set_xticklabels([str(value) for value in REFERENCE_SIZES])
        axis.set_ylim(0, 1)
        default_label = descriptor_name if len(clusters) == 1 and descriptor_name != "occupancy" else f"K = {clusters_value}"
        axis.set_title((resolution_labels or {}).get(clusters_value, default_label))
        axis.grid(axis="y", alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel(f"Macro {target_name} AUROC")
    for axis in axes[-1, :]:
        axis.set_xlabel("Positive reference-set size r")
    for axis in axes.flat[len(clusters):]:
        axis.set_visible(False)
    figure.suptitle(f"{branch.upper()} {descriptor_name} descriptors: {target_name} patient-reference kNN")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_hla(descriptor_dir: str | Path, source_dir: str | Path, output: str | Path, clusters: tuple[int, ...] = (16, 32, 64, 128), seed: int = 20260907, descriptor_pattern: str = "occupancy_k{clusters}_{branch}.parquet", descriptor_name: str = "occupancy", resolution_labels: dict[int, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all class-I HLA allele targets and write paired raw/OAR results."""
    descriptor_dir, output = Path(descriptor_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first_raw, _, _ = _load_pair(descriptor_dir, clusters[0], descriptor_pattern)
    descriptor_samples = first_raw["sample"].to_numpy(dtype=str)
    metadata = parse_emerson_hla(source_dir, descriptor_samples.tolist())
    if not np.array_equal(metadata["sample"].to_numpy(dtype=str), descriptor_samples):
        raise AssertionError("metadata extraction changed descriptor sample order")
    metadata.to_parquet(output / "metadata_join.parquet", index=False)
    included = metadata["included_hla_eval"].to_numpy(dtype=bool)
    if not included.any():
        raise ValueError("no descriptor donors have a recorded class-I HLA call")
    samples = descriptor_samples[included]
    metadata = metadata.loc[included].reset_index(drop=True)
    labels = np.array([allele for alleles in metadata["hla_alleles"] for allele in alleles], dtype=object)
    target_labels = np.array(sorted(set(labels)), dtype=object)
    draw_specs, target_qc = _reference_draws(
        np.array(["|".join(alleles) for alleles in metadata["hla_alleles"]], dtype=object), samples, seed
    )
    # The preceding compound labels only establish deterministic RNG progression.  Per-allele
    # draws below are the actual binary targets and are retained as an explicit audit table.
    del draw_specs, target_qc
    all_draw_rows: list[dict] = []
    reference_rows: list[dict] = []
    target_qc_rows: list[dict] = []
    for target_number, target in enumerate(target_labels):
        binary = metadata["hla_alleles"].map(lambda alleles: target in alleles).to_numpy(dtype=bool)
        target_draws, target_qc = _reference_draws(np.where(binary, target, "__negative__"), samples, seed + target_number)
        target_draws = [spec for spec in target_draws if spec["target"] == target]
        target_qc = target_qc[target_qc["target"] == target].copy()
        target_qc_rows.extend(target_qc.to_dict("records"))
        for spec in target_draws:
            refs = spec["reference_indices"]
            for index in refs:
                reference_rows.append({"target": spec["target"], "reference_size": spec["reference_size"], "draw": spec["draw"], "sample": samples[index]})
        for clusters_value in clusters:
            raw, oar, columns = _load_pair(descriptor_dir, clusters_value, descriptor_pattern)
            raw = raw[raw["sample"].isin(samples)].sort_values("sample").reset_index(drop=True)
            oar = oar[oar["sample"].isin(samples)].sort_values("sample").reset_index(drop=True)
            if not np.array_equal(raw["sample"].to_numpy(dtype=str), samples) or not np.array_equal(oar["sample"].to_numpy(dtype=str), samples):
                raise ValueError(f"K={clusters_value}: HLA-evaluable descriptor samples differ from K={clusters[0]}")
            raw_similarity = _cosine_gram(raw[columns].to_numpy(dtype=float))
            oar_similarity = _cosine_gram(oar[columns].to_numpy(dtype=float))
            for spec in target_draws:
                refs = spec["reference_indices"]
                score_mask = np.ones(len(samples), dtype=bool)
                score_mask[refs] = False
                y = binary[score_mask].astype(int)
                raw_score = raw_similarity[np.ix_(score_mask, refs)].max(axis=1)
                oar_score = oar_similarity[np.ix_(score_mask, refs)].max(axis=1)
                raw_auc = float(roc_auc_score(y, raw_score))
                oar_auc = float(roc_auc_score(y, oar_score))
                all_draw_rows.append({"target": target, "clusters": clusters_value, "reference_size": spec["reference_size"], "draw": spec["draw"], "raw_auroc": raw_auc, "oar_auroc": oar_auc, "oar_minus_raw": oar_auc - raw_auc})
    draws = pd.DataFrame(all_draw_rows).sort_values(["clusters", "target", "reference_size", "draw"]).reset_index(drop=True)
    references = pd.DataFrame(reference_rows).drop_duplicates().sort_values(["target", "reference_size", "draw", "sample"]).reset_index(drop=True)
    target_qc = pd.DataFrame(target_qc_rows).sort_values(["target", "reference_size"]).reset_index(drop=True)
    summary = draws.groupby(["clusters", "target", "reference_size"], as_index=False).agg(n_draws=("draw", "size"), raw_auroc_mean=("raw_auroc", "mean"), raw_auroc_sd=("raw_auroc", "std"), oar_auroc_mean=("oar_auroc", "mean"), oar_auroc_sd=("oar_auroc", "std"), oar_minus_raw_mean=("oar_minus_raw", "mean"), oar_minus_raw_sd=("oar_minus_raw", "std"))
    macro = summary.groupby(["clusters", "reference_size"], as_index=False).agg(n_targets=("target", "size"), raw_auroc_mean=("raw_auroc_mean", "mean"), oar_auroc_mean=("oar_auroc_mean", "mean"), oar_minus_raw_mean=("oar_minus_raw_mean", "mean"))
    draws.to_parquet(output / "hla_draws.parquet", index=False)
    summary.to_parquet(output / "hla_summary.parquet", index=False)
    macro.to_parquet(output / "hla_macro_summary.parquet", index=False)
    references.to_parquet(output / "reference_draws.parquet", index=False)
    target_qc.to_parquet(output / "hla_target_qc.parquet", index=False)
    _draw_macro_figure(draws, "raw", output / "hla_knn_auroc_raw.png", "HLA", descriptor_name, resolution_labels)
    _draw_macro_figure(draws, "oar", output / "hla_knn_auroc_oar.png", "HLA", descriptor_name, resolution_labels)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"targets": "known class-I HLA alleles from source sample tags", "descriptor_dir": str(descriptor_dir), "descriptor_pattern": descriptor_pattern, "descriptor_name": descriptor_name, "source_dir": str(source_dir), "clusters": list(clusters), "reference_sizes": list(REFERENCE_SIZES), "n_draws": N_DRAWS, "seed": seed, "n_descriptor_samples": len(descriptor_samples), "n_hla_evaluable_samples": len(samples), "n_targets": len(target_labels), "figures": ["hla_knn_auroc_raw.png", "hla_knn_auroc_oar.png"]}, handle, indent=2)
    return summary, macro


def evaluate_cmv(descriptor_dir: str | Path, source_dir: str | Path, output: str | Path, clusters: tuple[int, ...] = (16, 32, 64, 128), seed: int = 20260907, descriptor_pattern: str = "occupancy_k{clusters}_{branch}.parquet", descriptor_name: str = "occupancy", resolution_labels: dict[int, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate known CMV status with paired raw/OAR occupancy descriptors."""
    descriptor_dir, output = Path(descriptor_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first_raw, _, _ = _load_pair(descriptor_dir, clusters[0], descriptor_pattern)
    descriptor_samples = first_raw["sample"].to_numpy(dtype=str)
    metadata = parse_emerson_hla(source_dir, descriptor_samples.tolist())
    included = metadata["included_cmv_eval"].to_numpy(dtype=bool)
    metadata.to_parquet(output / "cmv_metadata_join.parquet", index=False)
    samples = descriptor_samples[included]
    binary = metadata.loc[included, "cmv_status"].eq("positive").to_numpy(dtype=bool)
    target_draws, target_qc = _reference_draws(np.where(binary, "CMV_positive", "__negative__"), samples, seed)
    target_draws = [spec for spec in target_draws if spec["target"] == "CMV_positive"]
    target_qc = target_qc[target_qc["target"] == "CMV_positive"].copy()
    all_draw_rows, reference_rows = [], []
    for spec in target_draws:
        for index in spec["reference_indices"]:
            reference_rows.append({"target": "CMV_positive", "reference_size": spec["reference_size"], "draw": spec["draw"], "sample": samples[index]})
    for clusters_value in clusters:
        raw, oar, columns = _load_pair(descriptor_dir, clusters_value, descriptor_pattern)
        raw = raw[raw["sample"].isin(samples)].sort_values("sample").reset_index(drop=True)
        oar = oar[oar["sample"].isin(samples)].sort_values("sample").reset_index(drop=True)
        if not np.array_equal(raw["sample"].to_numpy(dtype=str), samples) or not np.array_equal(oar["sample"].to_numpy(dtype=str), samples):
            raise ValueError(f"K={clusters_value}: CMV-evaluable descriptor samples differ from K={clusters[0]}")
        raw_similarity, oar_similarity = _cosine_gram(raw[columns].to_numpy(dtype=float)), _cosine_gram(oar[columns].to_numpy(dtype=float))
        for spec in target_draws:
            refs = spec["reference_indices"]; mask = np.ones(len(samples), dtype=bool); mask[refs] = False; y = binary[mask].astype(int)
            raw_auc = float(roc_auc_score(y, raw_similarity[np.ix_(mask, refs)].max(axis=1)))
            oar_auc = float(roc_auc_score(y, oar_similarity[np.ix_(mask, refs)].max(axis=1)))
            all_draw_rows.append({"target": "CMV_positive", "clusters": clusters_value, "reference_size": spec["reference_size"], "draw": spec["draw"], "raw_auroc": raw_auc, "oar_auroc": oar_auc, "oar_minus_raw": oar_auc - raw_auc})
    draws = pd.DataFrame(all_draw_rows).sort_values(["clusters", "reference_size", "draw"]).reset_index(drop=True)
    summary = draws.groupby(["clusters", "reference_size"], as_index=False).agg(n_draws=("draw", "size"), raw_auroc_mean=("raw_auroc", "mean"), raw_auroc_sd=("raw_auroc", "std"), oar_auroc_mean=("oar_auroc", "mean"), oar_auroc_sd=("oar_auroc", "std"), oar_minus_raw_mean=("oar_minus_raw", "mean"), oar_minus_raw_sd=("oar_minus_raw", "std"))
    draws.to_parquet(output / "cmv_draws.parquet", index=False); summary.to_parquet(output / "cmv_summary.parquet", index=False); target_qc.to_parquet(output / "cmv_target_qc.parquet", index=False); pd.DataFrame(reference_rows).to_parquet(output / "cmv_reference_draws.parquet", index=False)
    _draw_macro_figure(draws, "raw", output / "cmv_knn_auroc_raw.png", "CMV", descriptor_name, resolution_labels); _draw_macro_figure(draws, "oar", output / "cmv_knn_auroc_oar.png", "CMV", descriptor_name, resolution_labels)
    return summary, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", required=True, help="directory with paired occupancy descriptor tables")
    parser.add_argument("--emerson-tsv", required=True, help="directory containing P*.tsv Emerson source files")
    parser.add_argument("--out", required=True, help="output directory for HLA kNN-AUROC artefacts")
    parser.add_argument("--clusters", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--descriptor-pattern", default="occupancy_k{clusters}_{branch}.parquet", help="table pattern with {clusters} and {branch}")
    parser.add_argument("--descriptor-name", default="occupancy", help="label used in figure titles")
    parser.add_argument("--targets", nargs="+", choices=["hla", "cmv"], default=["hla"])
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args(argv)
    if "hla" in args.targets:
        evaluate_hla(args.descriptors, args.emerson_tsv, args.out, tuple(args.clusters), args.seed, args.descriptor_pattern, args.descriptor_name)
    if "cmv" in args.targets:
        evaluate_cmv(args.descriptors, args.emerson_tsv, args.out, tuple(args.clusters), args.seed, args.descriptor_pattern, args.descriptor_name)


if __name__ == "__main__":
    main()
