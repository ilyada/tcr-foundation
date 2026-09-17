"""Test compactness of CMV Fisher anchors in frozen TCR embedding space.

This module implements Stage 1 of the accepted embedding-assisted CMV
analysis.  It deliberately produces no donor-level score, no membership
rule, and no Keck evaluation.  All substantive numerical choices are command
line arguments so that none is selected by the result being tested.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .emerson_cmv_exact import _count_candidates, _fisher_statistics, _metadata, _public_gate
from .emerson_cmv_spatial import (
    _cmv_labelled_clouds,
    _count_publicness,
    _draw_control_sets,
    _l2_normalise,
    _load_embeddings,
    _mean_knn_distance,
    _parts,
)


def _p_sources(source_dir: Path, cloud_files: list[Path]) -> tuple[list[Path], dict[str, int]]:
    """Return only CMV-labelled P source files that have a frozen cloud."""
    labels = {
        row.sample: int(row.cmv_status == "positive")
        for row in _metadata(source_dir, "P*.tsv").itertuples()
        if row.cmv_status in {"positive", "negative"}
    }
    cloud_samples = {path.stem for path in cloud_files}
    selected = [path for path in sorted(source_dir.glob("P*.tsv")) if path.stem in cloud_samples and path.stem in labels]
    if not selected:
        raise FileNotFoundError("no CMV-labelled P source TSVs have matching frozen clouds")
    return selected, {path.stem: labels[path.stem] for path in selected}


def _nearest_publicness_pools(
    anchor_ids: list[str], counts: dict[str, int], *, controls_per_anchor: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Select the requested nearest-publicness controls within V gene and length."""
    if controls_per_anchor < 2:
        raise ValueError("controls_per_anchor must be at least two")
    anchor_set = set(anchor_ids)
    by_v_length: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for identity, incidence in counts.items():
        if identity in anchor_set:
            continue
        v_gene, cdr3aa, _ = _parts(identity)
        by_v_length[(v_gene, len(cdr3aa))].append((identity, incidence))
    pools: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for anchor in anchor_ids:
        v_gene, cdr3aa, _ = _parts(anchor)
        anchor_publicness = counts[anchor]
        candidates = sorted(
            by_v_length[(v_gene, len(cdr3aa))],
            key=lambda item: (abs(item[1] - anchor_publicness), item[0]),
        )
        chosen = candidates[:controls_per_anchor]
        retained = len(chosen) == controls_per_anchor
        if retained:
            pools[anchor] = np.asarray([identity for identity, _ in chosen], dtype=object)
        rows.append(
            {
                "identity": anchor,
                "v_gene": v_gene,
                "cdr3aa_length": len(cdr3aa),
                "anchor_publicness": anchor_publicness,
                "eligible_same_v_length": len(candidates),
                "selected_controls": len(chosen),
                "maximum_selected_publicness_difference": max((abs(incidence - anchor_publicness) for _, incidence in chosen), default=np.nan),
                "retained": retained,
            }
        )
    return pools, pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    axis.errorbar(summary["neighbours"], summary["null_mean_distance"], yerr=summary["null_sd_distance"], color="0.35", marker="o", capsize=3, label="matched null mean ± SD")
    axis.scatter(summary["neighbours"], summary["anchor_mean_distance"], color="#b2182b", label="Fisher anchors", zorder=3)
    axis.set(xlabel="Nearest-anchor count, k", ylabel="Mean cosine distance", title="CMV Fisher anchors versus matched public background")
    axis.legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def run_fisher_geometry(
    source_dir: str | Path,
    clouds_dir: str | Path,
    tmp_dir: str | Path,
    results_dir: str | Path,
    *,
    p_threshold: float,
    controls_per_anchor: int,
    draws: int,
    neighbours: tuple[int, ...],
    seed: int,
    bloom_bits: int = 1 << 33,
) -> dict[str, object]:
    """Compare Fisher-anchor compactness with matched public-background nulls."""
    if not 0 < p_threshold < 1:
        raise ValueError("p_threshold must lie between zero and one")
    if draws < 100:
        raise ValueError("draws must be at least 100")
    if not neighbours or min(neighbours) < 1 or len(set(neighbours)) != len(neighbours):
        raise ValueError("neighbours must be distinct positive integers")
    source, clouds, temporary, results = Path(source_dir), Path(clouds_dir), Path(tmp_dir), Path(results_dir)
    if temporary.exists() or results.exists():
        raise FileExistsError("tmp_dir and results_dir must both be new")
    cloud_files = _cmv_labelled_clouds(clouds, source)
    source_files, labels = _p_sources(source, cloud_files)
    public = _public_gate(source_files, bloom_bits=bloom_bits)
    incidence, class_sizes, _ = _count_candidates(source_files, labels, public)
    fisher = _fisher_statistics(incidence, class_sizes, p_threshold=p_threshold)
    anchor_ids = sorted(fisher["clonotype_key"].tolist())
    if len(anchor_ids) <= max(neighbours):
        raise RuntimeError("too few Fisher anchors for the requested neighbour counts")
    publicness, _, embedding_columns = _count_publicness(cloud_files, set(anchor_ids))
    missing_cloud_anchors = sorted(set(anchor_ids) - set(publicness))
    anchor_ids = [identity for identity in anchor_ids if identity in publicness]
    pools, qc = _nearest_publicness_pools(anchor_ids, publicness, controls_per_anchor=controls_per_anchor)
    retained = sorted(pools)
    if len(retained) <= max(neighbours):
        raise RuntimeError("too few Fisher anchors remain after matched-control eligibility")
    matched_draws = _draw_control_sets(retained, pools, draws=draws, seed=seed)
    selected_controls = set(np.concatenate(matched_draws).tolist())
    embeddings = _load_embeddings(cloud_files, set(retained) | selected_controls, embedding_columns)
    anchors = _l2_normalise(np.vstack([embeddings[identity] for identity in retained]))
    null_values = np.empty((draws, len(neighbours)), dtype=np.float64)
    observed_values: list[float] = []
    for column, neighbour_count in enumerate(neighbours):
        observed_values.append(_mean_knn_distance(anchors, neighbour_count))
        for draw_index, controls in enumerate(matched_draws):
            control_embeddings = _l2_normalise(np.vstack([embeddings[identity] for identity in controls]))
            null_values[draw_index, column] = _mean_knn_distance(control_embeddings, neighbour_count)
    rows = []
    for column, neighbour_count in enumerate(neighbours):
        null = null_values[:, column]
        observed = observed_values[column]
        rows.append(
            {
                "neighbours": neighbour_count,
                "anchor_mean_distance": observed,
                "null_mean_distance": float(null.mean()),
                "null_sd_distance": float(null.std(ddof=1)),
                "null_minus_anchor": float(null.mean() - observed),
                "empirical_p_compact": float((1 + np.count_nonzero(null <= observed)) / (1 + draws)),
            }
        )
    summary = pd.DataFrame(rows)
    temporary.mkdir(parents=True)
    results.mkdir(parents=True)
    fisher.to_csv(temporary / "emerson_cmv_fisher_anchors.tsv", sep="\t", index=False)
    qc.to_csv(temporary / "emerson_cmv_fisher_geometry_anchor_qc.tsv", sep="\t", index=False)
    pd.DataFrame(null_values, columns=[f"k{value}" for value in neighbours]).to_parquet(temporary / "emerson_cmv_fisher_geometry_null_draws.parquet", index=False)
    summary.to_csv(results / "emerson_cmv_fisher_geometry_summary.tsv", sep="\t", index=False)
    _plot_summary(summary, results / "emerson_cmv_fisher_geometry.png")
    manifest = {
        "stage": "geometrical validation only; no membership rule, donor score, classifier, or Keck analysis",
        "source_dir": str(source),
        "clouds_dir": str(clouds),
        "n_cmv_labelled_p_donors": len(source_files),
        "class_sizes": {"CMV-negative": class_sizes[0], "CMV-positive": class_sizes[1]},
        "fisher": {"one_sided": True, "p_threshold": p_threshold, "public_minimum_donors": 2, "n_selected_before_cloud_match": len(fisher), "n_absent_from_clouds": len(missing_cloud_anchors)},
        "matching": {"same_v_gene": True, "same_cdr3aa_length": True, "control_selection": "nearest absolute P-donor publicness difference with deterministic identity tie-break", "controls_per_anchor": controls_per_anchor, "n_retained_anchors": len(retained)},
        "embedding": "frozen joint-tiny raw P cloud embeddings, L2-normalized; cosine distance",
        "null": {"matched_draws": draws, "duplicate_free_per_draw": True, "seed": seed},
        "neighbours": list(neighbours),
        "temporary_outputs": str(temporary),
    }
    (results / "emerson_cmv_fisher_geometry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    return manifest


def _parse_neighbours(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("neighbours must be comma-separated integers") from error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--clouds-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--p-threshold", type=float, required=True)
    parser.add_argument("--controls-per-anchor", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--neighbours", type=_parse_neighbours, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bloom-bits", type=int, default=1 << 33)
    arguments = parser.parse_args(argv)
    run_fisher_geometry(**vars(arguments))


if __name__ == "__main__":
    main()
