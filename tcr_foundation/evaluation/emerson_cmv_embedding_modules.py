"""P-only cross-fitted calibration of embedding-assisted CMV TCR modules.

This module deliberately keeps the Emerson exact-clonotype score separate
from the embedding-derived local signal.  It is a candidate-discovery and
calibration stage; Keck is not read by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .emerson_cmv_exact import _count_candidates, _fisher_statistics, _metadata, _public_gate
from .emerson_cmv_spatial import (
    _cmv_labelled_clouds,
    _control_pools,
    _identity,
    _l2_normalise,
    _load_embeddings,
)


def _stratified_folds(labels: dict[str, int], folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    """Return deterministic stratified P train/held-out sample identifiers."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    generator = np.random.default_rng(seed)
    per_label: dict[int, list[str]] = {}
    for label in (0, 1):
        values = sorted(sample for sample, value in labels.items() if value == label)
        if len(values) < folds:
            raise ValueError("each CMV class must contain at least `folds` donors")
        generator.shuffle(values)
        per_label[label] = values
    result: list[tuple[set[str], set[str]]] = []
    for fold in range(folds):
        held_out = set(per_label[0][fold::folds]) | set(per_label[1][fold::folds])
        result.append((set(labels) - held_out, held_out))
    return result


def _sample_matched_controls(
    pools: dict[str, np.ndarray],
    *,
    controls_per_anchor: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Draw a bounded outcome-blind matched distance background per anchor."""
    if controls_per_anchor < 2:
        raise ValueError("controls_per_anchor must be at least two")
    generator = np.random.default_rng(seed)
    chosen: dict[str, np.ndarray] = {}
    for identity, candidates in pools.items():
        if len(candidates) < controls_per_anchor:
            raise ValueError(f"{identity}: insufficient matched controls")
        chosen[identity] = generator.choice(candidates, size=controls_per_anchor, replace=False)
    return chosen


def _anchor_thresholds(
    anchors: list[str],
    controls: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized anchor vectors and matched lower-tail cosine thresholds."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    anchor_values = _l2_normalise(np.vstack([embeddings[identity] for identity in anchors]))
    thresholds = []
    for index, identity in enumerate(anchors):
        background = _l2_normalise(np.vstack([embeddings[str(value)] for value in controls[identity]]))
        distances = 1.0 - background @ anchor_values[index]
        thresholds.append(float(np.quantile(distances, quantile)))
    return anchor_values, np.asarray(thresholds, dtype=np.float32)


def _is_local(values: np.ndarray, anchors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Assign each vector to one eligible nearest anchor, or ``-1`` if none."""
    if not len(values):
        return np.empty(0, dtype=np.int32)
    distances = 1.0 - _l2_normalise(values) @ anchors.T
    eligible = distances <= thresholds[None, :]
    assigned = np.full(len(values), -1, dtype=np.int32)
    has_match = eligible.any(axis=1)
    assigned[has_match] = np.argmin(np.where(eligible[has_match], distances[has_match], np.inf), axis=1)
    return assigned


def _selected_identities(paths: list[Path], labels: dict[str, int], *, p_threshold: float, bloom_bits: int) -> set[str]:
    """Discover raw exact CMV-enriched identities in one P partition."""
    public = _public_gate(paths, bloom_bits=bloom_bits)
    counts, class_sizes, _ = _count_candidates(paths, labels, public)
    selected = _fisher_statistics(counts, class_sizes, p_threshold=p_threshold)
    return {
        identity
        for row in selected.itertuples(index=False)
        if (identity := _identity(row.v_gene, row.cdr3aa, row.j_gene)) is not None
    }


def run_anchor_quantile_calibration(
    clouds_dir: str | Path,
    source_dir: str | Path,
    results_dir: str | Path,
    *,
    folds: int = 4,
    quantiles: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
    controls_per_anchor: int = 64,
    minimum_pool: int = 64,
    prevalence_tolerances: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50),
    p_threshold: float = 1e-4,
    bloom_bits: int = 1 << 33,
    seed: int = 0,
) -> pd.DataFrame:
    """Select a distance quantile by independent P-fold clonotype enrichment.

    Each P training fold discovers exact anchors.  The held-out fold then
    independently discovers CMV-enriched TCRs.  For every threshold, the
    statistic is the fraction of those non-exact held-out identities falling
    in a training-anchor neighbourhood, minus the same fraction for matched
    held-out controls.  No Keck input is accepted.
    """
    output = Path(results_dir)
    if output.exists():
        raise FileExistsError(f"results directory must be new: {output}")
    if not quantiles or any(not 0 < value < 1 for value in quantiles) or len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be unique values strictly between zero and one")
    metadata = _metadata(Path(source_dir), "P*.tsv")
    labels = {row.sample: int(row.cmv_status == "positive") for row in metadata.itertuples() if row.cmv_status in {"positive", "negative"}}
    source_paths = {path.stem: path for path in Path(source_dir).glob("P*.tsv") if path.stem in labels}
    cloud_paths = {path.stem: path for path in _cmv_labelled_clouds(Path(clouds_dir), Path(source_dir))}
    if set(source_paths) != set(cloud_paths):
        raise RuntimeError("CMV-labelled P source/cloud sample sets differ")
    rows: list[dict[str, object]] = []
    for fold_index, (train_samples, held_out_samples) in enumerate(_stratified_folds(labels, folds, seed)):
        train_paths = [source_paths[sample] for sample in sorted(train_samples)]
        held_out_paths = [source_paths[sample] for sample in sorted(held_out_samples)]
        train_labels = {sample: labels[sample] for sample in train_samples}
        held_out_labels = {sample: labels[sample] for sample in held_out_samples}
        anchors = sorted(_selected_identities(train_paths, train_labels, p_threshold=p_threshold, bloom_bits=bloom_bits))
        held_out_selected = _selected_identities(held_out_paths, held_out_labels, p_threshold=p_threshold, bloom_bits=bloom_bits)
        train_clouds = [cloud_paths[sample] for sample in sorted(train_samples)]
        held_out_clouds = [cloud_paths[sample] for sample in sorted(held_out_samples)]
        if not anchors or not held_out_selected:
            for quantile in quantiles:
                rows.append({"fold": fold_index, "quantile": quantile, "n_train_anchors": len(anchors), "n_held_out_selected": len(held_out_selected), "n_held_out_nonexact": 0, "local_fraction": np.nan, "matched_control_fraction": np.nan, "enrichment": np.nan, "status": "no_selected_identities"})
            continue
        from .emerson_cmv_spatial import _count_publicness
        train_counts, direct_anchors, embedding_columns = _count_publicness(train_clouds, set(anchors))
        anchors = sorted(direct_anchors)
        pools, _ = _control_pools(anchors, train_counts, prevalence_tolerances=prevalence_tolerances, minimum_pool=minimum_pool)
        anchors = sorted(pools)
        if not anchors:
            for quantile in quantiles:
                rows.append({"fold": fold_index, "quantile": quantile, "n_train_anchors": 0, "n_held_out_selected": len(held_out_selected), "n_held_out_nonexact": 0, "local_fraction": np.nan, "matched_control_fraction": np.nan, "enrichment": np.nan, "status": "no_matched_train_anchors"})
            continue
        matched_train = _sample_matched_controls(pools, controls_per_anchor=controls_per_anchor, seed=seed + fold_index)
        wanted_train = set(anchors) | {str(value) for values in matched_train.values() for value in values}
        train_embeddings = _load_embeddings(train_clouds, wanted_train, embedding_columns)
        nonexact = sorted(held_out_selected - set(anchors))
        held_counts, _, _ = _count_publicness(held_out_clouds, set())
        held_counts = {identity: incidence for identity, incidence in held_counts.items() if identity not in anchors}
        held_pools, _ = _control_pools(nonexact, held_counts, prevalence_tolerances=prevalence_tolerances, minimum_pool=minimum_pool)
        nonexact = sorted(held_pools)
        if not nonexact:
            for quantile in quantiles:
                rows.append({"fold": fold_index, "quantile": quantile, "n_train_anchors": len(anchors), "n_held_out_selected": len(held_out_selected), "n_held_out_nonexact": 0, "local_fraction": np.nan, "matched_control_fraction": np.nan, "enrichment": np.nan, "status": "no_matched_held_out_candidates"})
            continue
        matched_held_out = _sample_matched_controls(held_pools, controls_per_anchor=controls_per_anchor, seed=seed + 10_000 + fold_index)
        wanted_held_out = set(nonexact) | {str(value) for values in matched_held_out.values() for value in values}
        held_embeddings = _load_embeddings(held_out_clouds, wanted_held_out, embedding_columns)
        candidate_values = np.vstack([held_embeddings[identity] for identity in nonexact])
        control_values = np.vstack([held_embeddings[str(value)] for identity in nonexact for value in matched_held_out[identity]])
        for quantile in quantiles:
            anchor_values, thresholds = _anchor_thresholds(anchors, matched_train, train_embeddings, quantile)
            candidate_fraction = float(np.mean(_is_local(candidate_values, anchor_values, thresholds) >= 0))
            control_fraction = float(np.mean(_is_local(control_values, anchor_values, thresholds) >= 0))
            rows.append({"fold": fold_index, "quantile": quantile, "n_train_anchors": len(anchors), "n_held_out_selected": len(held_out_selected), "n_held_out_nonexact": len(nonexact), "local_fraction": candidate_fraction, "matched_control_fraction": control_fraction, "enrichment": candidate_fraction - control_fraction, "status": "evaluated"})
        print(f"anchor quantile calibration: fold {fold_index + 1}/{folds}, {len(anchors)} anchors, {len(nonexact)} held-out candidates", flush=True)
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True)
    frame.to_csv(output / "emerson_cmv_anchor_quantile_folds.tsv", sep="\t", index=False)
    summary = frame.loc[frame["status"].eq("evaluated")].groupby("quantile", as_index=False).agg(n_folds=("fold", "size"), enrichment_mean=("enrichment", "mean"), enrichment_median=("enrichment", "median"), local_fraction_mean=("local_fraction", "mean"), matched_control_fraction_mean=("matched_control_fraction", "mean")).sort_values("quantile", ignore_index=True)
    summary.to_csv(output / "emerson_cmv_anchor_quantile_summary.tsv", sep="\t", index=False)
    manifest = {"procedure": "P-only cross-fitted exact-anchor discovery with matched anchor-specific cosine-distance thresholds and independently discovered held-out CMV clonotypes", "folds": folds, "quantiles": list(quantiles), "controls_per_anchor": controls_per_anchor, "minimum_pool": minimum_pool, "p_threshold": p_threshold, "bloom_bits": bloom_bits, "seed": seed, "keck_used": False}
    (output / "emerson_cmv_anchor_quantile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds-dir", required=True)
    parser.add_argument("--emerson-tsv", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--quantiles", type=float, nargs="+", default=(0.001, 0.005, 0.01, 0.02, 0.05))
    parser.add_argument("--controls-per-anchor", type=int, default=64)
    parser.add_argument("--minimum-pool", type=int, default=64)
    parser.add_argument("--p-threshold", type=float, default=1e-4)
    parser.add_argument("--bloom-bits", type=int, default=1 << 33)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_anchor_quantile_calibration(args.clouds_dir, args.emerson_tsv, args.results, folds=args.folds, quantiles=tuple(args.quantiles), controls_per_anchor=args.controls_per_anchor, minimum_pool=args.minimum_pool, p_threshold=args.p_threshold, bloom_bits=args.bloom_bits, seed=args.seed)


if __name__ == "__main__":
    main()
