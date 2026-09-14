"""Test and operationalise local geometry of published Emerson CMV TCR beta chains.

The test reuses existing frozen-model P-repertoire clouds.  It makes no use of
CMV labels while selecting controls: each published anchor is compared with
clonotypes matched by V gene, CDR3 amino-acid length, and P-cohort donor
prevalence.  The statistic is the mean within-set cosine distance to the
nearest ``k`` neighbours, compared with repeated matched null sets.

The module also contains a deliberately binary ``exact-or-local-neighbour``
extension.  It first tests label enrichment among the ten nearest identities
in a matched anchor/background set, freezes components of a mutual three-NN
graph and their medoid radii, then evaluates the resulting beta-binomial
signature on P and Keck.  It is not a density model: every productive
clonotype is counted at most once.
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

from .emerson_cmv_exact import (
    _fit_beta_binomial,
    _metadata,
    _posterior_probability,
    _published_column,
    _read_published_table,
    _without_allele,
)


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


def _mean_label_enrichment(
    anchor_embeddings: np.ndarray,
    control_embeddings: np.ndarray,
    *,
    neighbours: int,
    anchor_mask: np.ndarray | None = None,
) -> float:
    """Return the mean same-label fraction among each selected identity's neighbours.

    ``anchor_mask`` identifies the group whose local purity is evaluated.  In
    the observed statistic it contains published CMV identities.  In matched
    permutations it instead contains one randomly selected identity from each
    published-anchor/control pair.
    """
    if anchor_embeddings.shape != control_embeddings.shape:
        raise ValueError("anchor and control embeddings must have the same shape")
    n_identities = len(anchor_embeddings)
    if not 1 <= neighbours < 2 * n_identities:
        raise ValueError("neighbours must be positive and smaller than the combined set")
    combined = np.vstack((anchor_embeddings, control_embeddings))
    if anchor_mask is None:
        anchor_mask = np.zeros(2 * n_identities, dtype=bool)
        anchor_mask[:n_identities] = True
    if anchor_mask.dtype != bool or anchor_mask.shape != (2 * n_identities,) or int(anchor_mask.sum()) != n_identities:
        raise ValueError("anchor_mask must select exactly one group of the matched pairs")
    nearest = _nearest_indices(combined, neighbours)
    fractions = anchor_mask[nearest].mean(axis=1)
    return float(fractions[anchor_mask].mean())


def _nearest_indices(embeddings: np.ndarray, neighbours: int) -> np.ndarray:
    """Return indices of the nearest cosine neighbours after excluding self."""
    if not 1 <= neighbours < len(embeddings):
        raise ValueError("neighbours must be positive and smaller than the embedding set")
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    return np.argpartition(similarity, kth=similarity.shape[1] - neighbours, axis=1)[:, -neighbours:]


def _local_label_enrichment(
    anchor_ids: list[str],
    matched_draws: list[np.ndarray],
    embeddings: dict[str, np.ndarray],
    *,
    neighbours: int,
    permutations_per_draw: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Test whether published identities occupy purer local ten-NN neighbourhoods.

    Each matched draw contributes one background identity for every published
    identity.  Its null permutes the two labels within every matched pair, so
    V gene, CDR3 length, and donor prevalence matching remain intact.
    """
    if permutations_per_draw < 1:
        raise ValueError("permutations_per_draw must be positive")
    anchors = _l2_normalise(np.vstack([embeddings[identity] for identity in anchor_ids]))
    generator = np.random.default_rng(seed)
    observed_values: list[float] = []
    null_values: list[float] = []
    for draw_index, draw in enumerate(matched_draws):
        controls = _l2_normalise(np.vstack([embeddings[identity] for identity in draw]))
        nearest = _nearest_indices(np.vstack((anchors, controls)), neighbours)
        observed_mask = np.zeros(2 * len(anchor_ids), dtype=bool)
        observed_mask[:len(anchor_ids)] = True
        observed_values.append(float(observed_mask[nearest].mean(axis=1)[observed_mask].mean()))
        choose_anchor = generator.integers(0, 2, size=(permutations_per_draw, len(anchor_ids)), endpoint=False).astype(bool)
        masks = np.concatenate((choose_anchor, ~choose_anchor), axis=1)
        neighbour_fractions = masks[:, nearest].mean(axis=2)
        null_values.extend((neighbour_fractions * masks).sum(axis=1).astype(float).tolist())
        if (draw_index + 1) % 100 == 0 or draw_index + 1 == len(matched_draws):
            print(f"local label enrichment: {draw_index + 1}/{len(matched_draws)} matched draws", flush=True)
    observed = np.asarray(observed_values)
    null = np.asarray(null_values)
    mean_observed = float(observed.mean())
    summary = pd.DataFrame(
        [{
            "neighbours": neighbours,
            "matched_draws": len(matched_draws),
            "within_pair_label_permutations_per_draw": permutations_per_draw,
            "observed_mean_same_label_fraction": mean_observed,
            "observed_sd_across_matched_draws": float(observed.std(ddof=1)),
            "null_mean_same_label_fraction": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "observed_minus_null": float(mean_observed - null.mean()),
            "empirical_p_higher_enrichment": float((1 + np.count_nonzero(null >= mean_observed)) / (1 + len(null))),
        }]
    )
    return summary, null


def _plot_local_enrichment(summary: pd.DataFrame, null: np.ndarray, path: Path) -> None:
    row = summary.iloc[0]
    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    axis.hist(null, bins=35, color="0.72", edgecolor="white", label="matched within-pair null")
    axis.axvline(row["observed_mean_same_label_fraction"], color="#b2182b", linewidth=2.2, label="published CMV identities")
    axis.set(xlabel=f"Mean same-label fraction among {int(row['neighbours'])} nearest identities", ylabel="Permuted matched draws", title="Local enrichment of published CMV TCR beta identities")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _mutual_knn_components(identities: list[str], embeddings: np.ndarray, *, neighbours: int) -> pd.DataFrame:
    """Freeze connected components of the mutual-kNN graph and medoid radii.

    Components with one identity have no radius and therefore remain exact-only
    anchors in the subsequent binary signature.
    """
    if not 1 <= neighbours < len(identities):
        raise ValueError("graph neighbours must be between one and n identities minus one")
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argpartition(similarity, kth=similarity.shape[1] - neighbours, axis=1)[:, -neighbours:]
    nearest_sets = [set(row.tolist()) for row in nearest]
    adjacency = [set() for _ in identities]
    for left, right_set in enumerate(nearest_sets):
        for right in right_set:
            if left in nearest_sets[right]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[list[int]] = []
    unseen = set(range(len(identities)))
    while unseen:
        seed = min(unseen)
        stack, component = [seed], []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            additions = adjacency[current] & unseen
            unseen -= additions
            stack.extend(additions)
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), tuple(identities[index] for index in values)))
    rows: list[dict[str, object]] = []
    for component_number, indices in enumerate(components, start=1):
        component_embeddings = embeddings[indices]
        component_size = len(indices)
        if component_size == 1:
            medoid_local, radius = 0, np.nan
        else:
            distance = 1.0 - component_embeddings @ component_embeddings.T
            mean_distance = distance.sum(axis=1) / (component_size - 1)
            medoid_local = int(np.argmin(mean_distance))
            radius = float(mean_distance[medoid_local])
        medoid_index = indices[medoid_local]
        for index in indices:
            rows.append(
                {
                    "identity": identities[index],
                    "component": component_number,
                    "component_size": component_size,
                    "is_medoid": bool(index == medoid_index),
                    "radius_cosine_distance": radius,
                    "distance_to_component_medoid": float(1.0 - embeddings[index] @ embeddings[medoid_index]),
                    "exact_only": bool(component_size == 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["component", "identity"], ignore_index=True)


def run_local_enrichment(
    published_reference: str | Path,
    clouds_dir: str | Path,
    source_dir: str | Path,
    results_dir: str | Path,
    *,
    draws: int = 500,
    local_neighbours: int = 10,
    graph_neighbours: int = 3,
    permutations_per_draw: int = 100,
    minimum_pool: int = 20,
    prevalence_tolerances: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50),
    seed: int = 0,
) -> dict[str, object]:
    """Perform the matched local-label test and freeze graph components/radii."""
    if draws < 100:
        raise ValueError("draws must be at least 100")
    output = Path(results_dir)
    if output.exists():
        raise FileExistsError(f"results directory must be new: {output}")
    anchors = _published_anchors(Path(published_reference))
    files = _cmv_labelled_clouds(Path(clouds_dir), Path(source_dir))
    counts, direct_anchor_set, embedding_columns = _count_publicness(files, set(anchors["identity"]))
    direct_anchor_ids = sorted(direct_anchor_set)
    pools, qc = _control_pools(direct_anchor_ids, counts, prevalence_tolerances=prevalence_tolerances, minimum_pool=minimum_pool)
    retained_ids = sorted(pools)
    if len(retained_ids) <= max(local_neighbours, graph_neighbours):
        raise RuntimeError("too few anchors retained for local enrichment and graph construction")
    matched_draws = _draw_control_sets(retained_ids, pools, draws=draws, seed=seed)
    selected_controls = set(np.concatenate(matched_draws).tolist())
    embeddings = _load_embeddings(files, set(retained_ids) | selected_controls, embedding_columns)
    summary, null = _local_label_enrichment(retained_ids, matched_draws, embeddings, neighbours=local_neighbours, permutations_per_draw=permutations_per_draw, seed=seed + 1)
    anchor_embeddings = _l2_normalise(np.vstack([embeddings[identity] for identity in retained_ids]))
    components = _mutual_knn_components(retained_ids, anchor_embeddings, neighbours=graph_neighbours)
    output.mkdir(parents=True)
    qc = anchors.merge(qc, on="identity", how="left")
    qc["direct_cloud_identity"] = qc["identity"].isin(direct_anchor_set)
    qc.to_csv(output / "emerson_cmv_local_anchor_qc.tsv", sep="\t", index=False)
    summary.to_csv(output / "emerson_cmv_local_enrichment.tsv", sep="\t", index=False)
    components.to_csv(output / "emerson_cmv_local_components.tsv", sep="\t", index=False)
    _plot_local_enrichment(summary, null, output / "emerson_cmv_local_enrichment.png")
    manifest = {
        "published_reference": str(published_reference),
        "clouds_dir": str(clouds_dir),
        "source_dir": str(source_dir),
        "n_cmv_labelled_p_clouds": len(files),
        "n_published_anchors": len(anchors),
        "n_retained_matched_anchors": len(retained_ids),
        "embedding": "existing frozen joint-tiny raw P clouds; L2-normalised before cosine distance",
        "local_label_test": {"neighbours": local_neighbours, "matched_draws": draws, "within_pair_label_permutations_per_draw": permutations_per_draw, "control_matching": "exact V gene and CDR3 amino-acid length; nearest P-donor prevalence tolerance"},
        "graph": {"type": "mutual k-nearest-neighbour connected components", "neighbours": graph_neighbours, "radius": "mean cosine distance from the component medoid to every other component member; singleton components are exact-only"},
        "seed": seed,
    }
    (output / "emerson_cmv_local_enrichment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(components.groupby("component_size").size().rename("n_components").to_string(), flush=True)
    return manifest


def _labelled_clouds(clouds_dir: Path, source_dir: Path, pattern: str) -> tuple[list[Path], dict[str, int]]:
    metadata = _metadata(source_dir, pattern.replace(".parquet", ".tsv"))
    labels = {row.sample: int(row.cmv_status == "positive") for row in metadata.itertuples() if row.cmv_status in {"positive", "negative"}}
    files = [path for path in sorted(clouds_dir.glob(pattern)) if path.stem in labels]
    if len(files) != len(labels):
        raise RuntimeError(f"cloud/source mismatch for {pattern}: {len(files)} clouds and {len(labels)} labelled source files")
    return files, labels


def _cloud_signature_counts(
    files: list[Path],
    labels: dict[str, int],
    embedding_columns: list[str],
    exact_identities: set[str],
    medoid_embeddings: np.ndarray,
    radii: np.ndarray,
) -> pd.DataFrame:
    """Count each unique productive cloud identity once under the frozen rule."""
    columns = [*_IDENTITY_COLUMNS, *embedding_columns]
    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=columns)
        unique: dict[str, np.ndarray] = {}
        for row in frame.itertuples(index=False, name=None):
            v_gene, cdr3aa, j_gene, *embedding = row
            identity = _identity(v_gene, cdr3aa, j_gene)
            if identity is not None and identity not in unique:
                unique[identity] = np.asarray(embedding, dtype=np.float32)
        identities = list(unique)
        values = _l2_normalise(np.vstack([unique[identity] for identity in identities])) if identities else np.empty((0, len(embedding_columns)), dtype=np.float32)
        exact = np.asarray([identity in exact_identities for identity in identities], dtype=bool)
        if len(values) and len(medoid_embeddings):
            distances = 1.0 - values @ medoid_embeddings.T
            local = (distances <= radii[None, :]).any(axis=1)
        else:
            local = np.zeros(len(identities), dtype=bool)
        total = exact | local
        rows.append({"sample": path.stem, "cmv_label": labels[path.stem], "n_unique_productive": len(identities), "k_exact_present": int(exact.sum()), "k_local_neighbour_present": int((local & ~exact).sum()), "k_exact_or_local_present": int(total.sum())})
        if index % 25 == 0 or index == len(files):
            print(f"cloud signature counts: {index}/{len(files)} clouds", flush=True)
    return pd.DataFrame(rows)


def _score_signature(frame: pd.DataFrame, count_column: str) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    fitted: dict[str, dict[str, float]] = {}
    for label, name in ((0, "negative"), (1, "positive")):
        subset = result.loc[result["cmv_label"].eq(label)]
        fitted[name] = _fit_beta_binomial(subset["n_unique_productive"].to_numpy(float), subset[count_column].to_numpy(float))
    result["cmv_probability"] = _posterior_probability(result["n_unique_productive"].to_numpy(float), result[count_column].to_numpy(float), fitted["negative"], fitted["positive"])
    return result, fitted


def _dual_roc_figure(frame: pd.DataFrame, path: Path, title: str) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    figure, axis = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    results: dict[str, float] = {}
    for column, label, colour in (("cmv_probability_exact", "exact anchors", "#4d4d4d"), ("cmv_probability_local", "exact or local neighbour", "#b2182b")):
        value = float(roc_auc_score(frame["cmv_label"], frame[column]))
        fpr, tpr, _ = roc_curve(frame["cmv_label"], frame[column])
        axis.plot(fpr, tpr, linewidth=2, color=colour, label=f"{label}: AUROC = {value:.3f}")
        results[column] = value
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    axis.legend(loc="lower right", frameon=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return results


def run_local_neighbour_classifier(
    local_enrichment_dir: str | Path,
    p_clouds_dir: str | Path,
    keck_clouds_dir: str | Path,
    source_dir: str | Path,
    results_dir: str | Path,
) -> dict[str, object]:
    """Fit P beta-binomial models for frozen exact-or-local cloud signatures and test Keck."""
    local_dir, p_clouds, keck_clouds, source, output = Path(local_enrichment_dir), Path(p_clouds_dir), Path(keck_clouds_dir), Path(source_dir), Path(results_dir)
    component_path = local_dir / "emerson_cmv_local_components.tsv"
    if not component_path.exists():
        raise FileNotFoundError(f"{component_path}: run local enrichment before classification")
    if output.exists():
        raise FileExistsError(f"results directory must be new: {output}")
    components = pd.read_csv(component_path, sep="\t")
    required = {"identity", "component_size", "is_medoid", "radius_cosine_distance"}
    if missing := required - set(components.columns):
        raise ValueError(f"{component_path}: missing required columns {sorted(missing)!r}")
    exact_identities = set(components["identity"])
    medoids = components.loc[components["is_medoid"] & components["component_size"].gt(1)].copy()
    p_files, p_labels = _labelled_clouds(p_clouds, source, "P*.parquet")
    keck_files, keck_labels = _labelled_clouds(keck_clouds, source, "Keck*.parquet")
    embedding_columns = _embedding_columns(p_files[0])
    medoid_vectors = _load_embeddings(p_files, set(medoids["identity"]), embedding_columns) if len(medoids) else {}
    medoid_embeddings = _l2_normalise(np.vstack([medoid_vectors[identity] for identity in medoids["identity"]])) if len(medoids) else np.empty((0, len(embedding_columns)), dtype=np.float32)
    radii = medoids["radius_cosine_distance"].to_numpy(dtype=np.float32)
    p_counts = _cloud_signature_counts(p_files, p_labels, embedding_columns, exact_identities, medoid_embeddings, radii)
    keck_counts = _cloud_signature_counts(keck_files, keck_labels, embedding_columns, exact_identities, medoid_embeddings, radii)
    p_exact, exact_model = _score_signature(p_counts, "k_exact_present")
    p_local, local_model = _score_signature(p_counts, "k_exact_or_local_present")
    p_counts["cmv_probability_exact"] = p_exact["cmv_probability"]
    p_counts["cmv_probability_local"] = p_local["cmv_probability"]
    keck_counts["cmv_probability_exact"] = _posterior_probability(keck_counts["n_unique_productive"].to_numpy(float), keck_counts["k_exact_present"].to_numpy(float), exact_model["negative"], exact_model["positive"])
    keck_counts["cmv_probability_local"] = _posterior_probability(keck_counts["n_unique_productive"].to_numpy(float), keck_counts["k_exact_or_local_present"].to_numpy(float), local_model["negative"], local_model["positive"])
    output.mkdir(parents=True)
    p_auc = _dual_roc_figure(p_counts, output / "emerson_cmv_local_p_apparent_roc.png", "Cloud signature: P (apparent)")
    keck_auc = _dual_roc_figure(keck_counts, output / "emerson_cmv_local_keck_external_roc.png", "Cloud signature: Keck external validation")
    p_counts.to_csv(output / "emerson_cmv_local_p_scores.tsv", sep="\t", index=False)
    keck_counts.to_csv(output / "emerson_cmv_local_keck_scores.tsv", sep="\t", index=False)
    manifest = {
        "local_enrichment_dir": str(local_dir),
        "p_clouds_dir": str(p_clouds),
        "keck_clouds_dir": str(keck_clouds),
        "source_dir": str(source),
        "signature": "each cloud identity is counted once if it is an exact published anchor or lies within any frozen non-singleton component medoid radius",
        "n_exact_anchors": len(exact_identities),
        "n_non_singleton_component_medoids": len(medoids),
        "p_beta_binomial_exact": exact_model,
        "p_beta_binomial_exact_or_local": local_model,
        "p_apparent_auroc": p_auc,
        "keck_external_auroc": keck_auc,
    }
    (output / "emerson_cmv_local_classifier_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


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
    parser.add_argument("--local-enrichment-results", help="new directory for the matched ten-NN label-enrichment test and frozen local components")
    parser.add_argument("--local-classifier-results", help="new directory for the frozen exact-or-local beta-binomial classifier results")
    parser.add_argument("--local-enrichment-input", help="completed local-enrichment result directory, required for --local-classifier-results")
    parser.add_argument("--keck-clouds-dir", help="Keck raw-cloud directory, required for --local-classifier-results")
    parser.add_argument("--local-neighbours", type=int, default=10, help="combined anchor/background neighbourhood size for the local-label test")
    parser.add_argument("--graph-neighbours", type=int, default=3, help="mutual-kNN graph degree used to freeze local components")
    parser.add_argument("--within-pair-permutations", type=int, default=100, help="matched-pair label permutations per control draw")
    args = parser.parse_args(argv)
    if args.local_enrichment_results or args.local_classifier_results or args.local_enrichment_input:
        if args.local_enrichment_results:
            if args.local_classifier_results or args.local_enrichment_input:
                parser.error("--local-enrichment-results cannot be combined with local-classifier arguments")
            if not all((args.published_reference, args.clouds_dir, args.emerson_tsv)):
                parser.error("--published-reference, --clouds-dir, and --emerson-tsv are required for --local-enrichment-results")
            run_local_enrichment(
                args.published_reference,
                args.clouds_dir,
                args.emerson_tsv,
                args.local_enrichment_results,
                draws=args.draws,
                local_neighbours=args.local_neighbours,
                graph_neighbours=args.graph_neighbours,
                permutations_per_draw=args.within_pair_permutations,
                minimum_pool=args.minimum_pool,
                prevalence_tolerances=tuple(args.prevalence_tolerances),
                seed=args.seed,
            )
            return
        if not all((args.local_classifier_results, args.local_enrichment_input, args.clouds_dir, args.keck_clouds_dir, args.emerson_tsv)):
            parser.error("--local-classifier-results, --local-enrichment-input, --clouds-dir, --keck-clouds-dir, and --emerson-tsv are required together")
        run_local_neighbour_classifier(args.local_enrichment_input, args.clouds_dir, args.keck_clouds_dir, args.emerson_tsv, args.local_classifier_results)
        return
    run_spatial_test(args.published_reference, args.clouds_dir, args.emerson_tsv, args.results, draws=args.draws, neighbours=tuple(args.neighbours), minimum_pool=args.minimum_pool, prevalence_tolerances=tuple(args.prevalence_tolerances), seed=args.seed)


if __name__ == "__main__":
    main()
