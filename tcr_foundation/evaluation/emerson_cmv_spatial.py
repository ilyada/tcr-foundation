"""Test whether published Emerson CMV-associated TCR beta chains are spatially clustered.

The test reuses existing frozen-model P-repertoire clouds.  It makes no use of
CMV labels while selecting controls: each published anchor is compared with
clonotypes matched by V gene, CDR3 amino-acid length, and P-cohort donor
prevalence.  The statistic is the mean within-set cosine distance to the
nearest ``k`` neighbours, compared with repeated matched null sets.

The computation deliberately precedes any CMV classifier.  A local embedding
signal must be established before a continuous density or region score can be
defined and evaluated on Keck.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .emerson_cmv_exact import _metadata, _published_column, _read_published_table, _without_allele


_SEPARATOR = "\x1f"
_IDENTITY_COLUMNS = ("v_gene", "cdr3aa", "j_gene")


def _embedding_columns(path: Path) -> list[str]:
    columns = [column for column in pq.ParquetFile(path).schema.names if column.startswith("e") and column[1:].isdigit()]
    if not columns:
        raise ValueError(f"{path}: no embedding columns")
    return sorted(columns, key=lambda column: int(column[1:]))


def _identity(v_gene: object, cdr3aa: object, j_gene: object) -> str | None:
    """Return the cloud-compatible identity after only allele-suffix removal."""
    cdr3 = str(cdr3aa).strip().upper()
    v_gene_text, j_gene_text = _without_allele(v_gene), _without_allele(j_gene)
    if not v_gene_text or not cdr3 or not j_gene_text or cdr3.lower() == "nan":
        return None
    return _SEPARATOR.join((v_gene_text, cdr3, j_gene_text))


def _parts(identity: str) -> tuple[str, str, str]:
    return tuple(identity.split(_SEPARATOR, 2))  # type: ignore[return-value]


def _l2_normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("embeddings must be finite and have positive norm")
    return values / norms


def _published_anchors(path: Path) -> pd.DataFrame:
    """Load Table 2 and construct stable cloud identities for its TCR beta rows."""
    table = _read_published_table(path)
    v_column = _published_column(table, {"vgene", "v", "vsegment"}, "V-gene")
    cdr3_column = _published_column(table, {"cdr3", "cdr3aa", "aminoacid", "aminoacidsequence", "cdr3aminoacid"}, "CDR3 amino-acid")
    j_column = _published_column(table, {"jgene", "j", "jsegment"}, "J-gene")
    anchors = pd.DataFrame({"v_gene_published": table[v_column], "cdr3aa": table[cdr3_column], "j_gene_published": table[j_column]})
    anchors["identity"] = [_identity(v_gene, cdr3aa, j_gene) for v_gene, cdr3aa, j_gene in anchors.itertuples(index=False, name=None)]
    anchors = anchors.dropna(subset=["identity"]).drop_duplicates("identity", ignore_index=True)
    if len(anchors) != 164:
        raise ValueError(f"expected 164 unique published identities after allele-suffix removal, got {len(anchors)}")
    return anchors


def _cmv_labelled_clouds(clouds_dir: Path, source_dir: Path) -> list[Path]:
    """Keep only P clouds whose native source record has a known CMV call."""
    metadata = _metadata(source_dir, "P*.tsv")
    accepted = set(metadata.loc[metadata["cmv_status"].isin(("positive", "negative")), "sample"])
    files = [path for path in sorted(clouds_dir.glob("P*.parquet")) if path.stem in accepted]
    if not files:
        raise FileNotFoundError("no CMV-labelled P clouds matched the source TSV metadata")
    return files


def _count_publicness(files: list[Path], anchors: set[str]) -> tuple[dict[str, int], set[str], list[str]]:
    """Count cloud-donor incidence and record direct Table-2 cloud matches."""
    counts: dict[str, int] = {}
    direct_anchors: set[str] = set()
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=list(_IDENTITY_COLUMNS)).drop_duplicates(list(_IDENTITY_COLUMNS))
        identities = {_identity(v_gene, cdr3aa, j_gene) for v_gene, cdr3aa, j_gene in frame.itertuples(index=False, name=None)}
        identities.discard(None)
        direct_anchors.update(anchors & identities)
        for identity in identities:
            counts[identity] = counts.get(identity, 0) + 1
        if index % 25 == 0 or index == len(files):
            print(f"publicness: {index}/{len(files)} clouds, {len(counts):,} identities", flush=True)
    return counts, direct_anchors, _embedding_columns(files[0])


def _control_pools(
    anchor_ids: list[str],
    counts: dict[str, int],
    *,
    prevalence_tolerances: tuple[int, ...],
    minimum_pool: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build transparent V/length/prevalence-matched candidate pools per anchor."""
    anchor_set = set(anchor_ids)
    candidates_by_v_length: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for identity, incidence in counts.items():
        if identity in anchor_set:
            continue
        v_gene, cdr3aa, _ = _parts(identity)
        if v_gene == "unresolved":
            continue
        candidates_by_v_length[(v_gene, len(cdr3aa))].append((identity, incidence))
    pools: dict[str, np.ndarray] = {}
    rows = []
    for identity in anchor_ids:
        v_gene, cdr3aa, _ = _parts(identity)
        incidence = counts[identity]
        candidates = candidates_by_v_length[(v_gene, len(cdr3aa))]
        selected: list[str] = []
        used_tolerance: int | None = None
        for tolerance in prevalence_tolerances:
            selected = [candidate for candidate, candidate_incidence in candidates if abs(candidate_incidence - incidence) <= tolerance]
            if len(selected) >= minimum_pool:
                used_tolerance = tolerance
                break
        rows.append({"identity": identity, "v_gene": v_gene, "cdr3aa_length": len(cdr3aa), "p_donor_prevalence": incidence, "n_control_candidates": len(selected), "prevalence_tolerance_used": used_tolerance, "retained": used_tolerance is not None})
        if used_tolerance is not None:
            pools[identity] = np.asarray(selected, dtype=object)
    return pools, pd.DataFrame(rows)


def _draw_control_sets(anchor_ids: list[str], pools: dict[str, np.ndarray], *, draws: int, seed: int) -> list[np.ndarray]:
    """Draw one unique matched control per anchor for each repeated null set."""
    generator = np.random.default_rng(seed)
    ordered_anchors = sorted(anchor_ids, key=lambda identity: (len(pools[identity]), identity))
    result: list[np.ndarray] = []
    for _ in range(draws):
        selected: list[str] = []
        selected_set: set[str] = set()
        for identity in ordered_anchors:
            available = [candidate for candidate in pools[identity] if candidate not in selected_set]
            if not available:
                raise RuntimeError("overlapping control pools cannot provide a duplicate-free matched-null draw")
            chosen = str(generator.choice(available))
            selected.append(chosen)
            selected_set.add(chosen)
        result.append(np.asarray(selected, dtype=object))
    return result


def _load_embeddings(files: list[Path], wanted: set[str], embedding_columns: list[str]) -> dict[str, np.ndarray]:
    """Read coordinates only for anchors and controls selected after publicness counting."""
    found: dict[str, np.ndarray] = {}
    columns = [*_IDENTITY_COLUMNS, *embedding_columns]
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=columns)
        for row in frame.itertuples(index=False, name=None):
            v_gene, cdr3aa, j_gene, *embedding = row
            identity = _identity(v_gene, cdr3aa, j_gene)
            if identity in wanted and identity not in found:
                found[identity] = np.asarray(embedding, dtype=np.float32)
        if index % 25 == 0 or index == len(files):
            print(f"embeddings: {index}/{len(files)} clouds, {len(found):,}/{len(wanted):,} identities", flush=True)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} selected identities were absent while reading embeddings")
    return found


def _mean_knn_distance(embeddings: np.ndarray, neighbours: int) -> float:
    """Return mean cosine distance to the closest ``neighbours`` other anchors."""
    if not 1 <= neighbours < len(embeddings):
        raise ValueError("neighbours must be between one and the number of embeddings minus one")
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    closest_similarity = np.partition(similarity, kth=similarity.shape[1] - neighbours, axis=1)[:, -neighbours:]
    return float(np.mean(1.0 - closest_similarity))


def _plot_null(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for row in summary.itertuples(index=False):
        axis.errorbar(row.neighbours, row.null_mean, yerr=row.null_sd, color="0.3", marker="o", capsize=3)
        axis.scatter(row.neighbours, row.anchor_mean_distance, color="#b2182b", zorder=3)
    axis.set(xlabel="Number of nearest neighbours, k", ylabel="Mean cosine distance", title="Published CMV anchors versus matched null sets")
    axis.text(0.02, 0.98, "red: published anchors\ngray: null mean ± SD", transform=axis.transAxes, va="top", ha="left")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_spatial_test(
    published_reference: str | Path,
    clouds_dir: str | Path,
    source_dir: str | Path,
    results_dir: str | Path,
    *,
    draws: int = 500,
    neighbours: tuple[int, ...] = (1, 3, 5, 10),
    minimum_pool: int = 20,
    prevalence_tolerances: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50),
    seed: int = 0,
) -> dict[str, object]:
    """Run the preregistered matched-null test using existing raw P clouds."""
    if draws < 100:
        raise ValueError("draws must be at least 100 for an empirical null test")
    if minimum_pool < 2:
        raise ValueError("minimum_pool must be at least two")
    if not neighbours or min(neighbours) < 1 or len(set(neighbours)) != len(neighbours):
        raise ValueError("neighbours must contain unique positive integers")
    output = Path(results_dir)
    if output.exists():
        raise FileExistsError(f"results directory must be new: {output}")
    anchors = _published_anchors(Path(published_reference))
    files = _cmv_labelled_clouds(Path(clouds_dir), Path(source_dir))
    counts, direct_anchor_set, embedding_columns = _count_publicness(files, set(anchors["identity"]))
    direct_anchor_ids = sorted(direct_anchor_set)
    if len(direct_anchor_ids) < max(neighbours) + 1:
        raise RuntimeError("too few Table-2 anchors have direct cloud identities")
    pools, qc = _control_pools(direct_anchor_ids, counts, prevalence_tolerances=prevalence_tolerances, minimum_pool=minimum_pool)
    retained_ids = sorted(pools)
    if len(retained_ids) < max(neighbours) + 1:
        raise RuntimeError("too few anchors retained after matched-control eligibility")
    matched_draws = _draw_control_sets(retained_ids, pools, draws=draws, seed=seed)
    selected_controls = set(np.concatenate(matched_draws).tolist())
    embeddings = _load_embeddings(files, set(retained_ids) | selected_controls, embedding_columns)
    anchor_embeddings = _l2_normalise(np.vstack([embeddings[identity] for identity in retained_ids]))
    rows = []
    for k_value in neighbours:
        observed = _mean_knn_distance(anchor_embeddings, k_value)
        null = np.asarray([_mean_knn_distance(_l2_normalise(np.vstack([embeddings[identity] for identity in draw])), k_value) for draw in matched_draws])
        rows.append({"neighbours": k_value, "anchor_mean_distance": observed, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "null_mean_minus_anchor": float(null.mean() - observed), "empirical_p_lower_distance": float((1 + np.count_nonzero(null <= observed)) / (1 + len(null)))})
    summary = pd.DataFrame(rows)
    output.mkdir(parents=True)
    qc = anchors.merge(qc, on="identity", how="left")
    qc["direct_cloud_identity"] = qc["identity"].isin(direct_anchor_set)
    qc.to_csv(output / "emerson_cmv_spatial_anchor_qc.tsv", sep="\t", index=False)
    summary.to_csv(output / "emerson_cmv_spatial_knn_null.tsv", sep="\t", index=False)
    _plot_null(summary, output / "emerson_cmv_spatial_knn_null.png")
    manifest = {
        "published_reference": str(published_reference),
        "clouds_dir": str(clouds_dir),
        "source_dir": str(source_dir),
        "n_cmv_labelled_p_clouds": len(files),
        "n_published_anchors": len(anchors),
        "n_direct_cloud_anchors": len(direct_anchor_set),
        "n_retained_matched_anchors": len(retained_ids),
        "identity": "V gene and J gene after only *<allele number> removal; CDR3 amino-acid exact",
        "embedding": "existing frozen joint-tiny raw P clouds; L2-normalised before cosine distance",
        "control_matching": {"V_gene": "exact", "cdr3aa_length": "exact", "P_donor_prevalence": "nearest allowed tolerance with minimum-pool requirement", "minimum_pool": minimum_pool, "allowed_prevalence_tolerances": list(prevalence_tolerances)},
        "draws": draws,
        "seed": seed,
        "neighbours": list(neighbours),
    }
    (output / "emerson_cmv_spatial_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--published-reference", required=True)
    parser.add_argument("--clouds-dir", required=True)
    parser.add_argument("--emerson-tsv", required=True, help="source directory containing P*.tsv")
    parser.add_argument("--results", required=True, help="new directory for compact final artefacts")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--neighbours", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--minimum-pool", type=int, default=20)
    parser.add_argument("--prevalence-tolerances", type=int, nargs="+", default=(0, 1, 2, 5, 10, 20, 50))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_spatial_test(args.published_reference, args.clouds_dir, args.emerson_tsv, args.results, draws=args.draws, neighbours=tuple(args.neighbours), minimum_pool=args.minimum_pool, prevalence_tolerances=tuple(args.prevalence_tolerances), seed=args.seed)


if __name__ == "__main__":
    main()
