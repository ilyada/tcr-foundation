"""Nested, embedding-assisted CMV evaluation without label leakage.

The module compares three donor-level models: the literal Emerson
exact-clonotype beta-binomial score, that score plus matched direct local
hits, and that score plus direct hits supported by shared nearest neighbours.
All anchors, controls, reference identities, thresholds, and fitted
coefficients are derived inside the current training partition.  Validation,
outer-test, and Keck labels are used only after their predictions are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .emerson_cmv_exact import (
    KEY_SEPARATOR,
    _count_candidates,
    _fisher_statistics,
    _fit_beta_binomial,
    _metadata,
    _posterior_probability,
    _productive_keys,
    _public_gate,
)
from .emerson_cmv_spatial import _embedding_columns, _identity, _l2_normalise


_IDENTITY_COLUMNS = ("v_gene", "cdr3aa", "j_gene")


@dataclass(frozen=True)
class LocalConfiguration:
    """One predeclared local-evidence configuration selected by inner CV."""

    method: str
    controls: int | None = None
    direct_rank: int | None = None
    neighbours: int | None = None
    topology_rank: int | None = None
    logistic_c: float | None = None

    def complexity_key(self) -> tuple[int, int, int, int, int]:
        order = {"exact": 0, "direct": 1, "topology": 2}[self.method]
        return (
            order,
            self.controls or 0,
            self.direct_rank or 0,
            self.neighbours or 0,
            self.topology_rank or 0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "controls": self.controls,
            "direct_rank": self.direct_rank,
            "neighbours": self.neighbours,
            "topology_rank": self.topology_rank,
            "logistic_c": self.logistic_c,
        }


@dataclass
class TrainingState:
    """Objects fitted only from one current training partition."""

    exact_anchors: set[str]
    cloud_anchors: list[str]
    anchor_vectors: np.ndarray
    controls_by_anchor: dict[str, list[str]]
    embeddings: dict[str, np.ndarray]
    reference_ids: np.ndarray
    reference_vectors: np.ndarray
    reference_positions: dict[str, int]
    embedding_columns: list[str]
    negative_model: dict[str, float] | None
    positive_model: dict[str, float] | None


def _stratified_folds(labels: dict[str, int], folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    """Return deterministic donor-disjoint stratified folds."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    values = np.asarray(sorted(labels), dtype=object)
    y = np.asarray([labels[str(sample)] for sample in values], dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(set(values[train]), set(values[test])) for train, test in splitter.split(values, y)]


def _cloud_identity(exact_identity: str) -> str:
    """Convert a literal source identity to its cloud identity without broad harmonisation."""
    v_gene, cdr3aa, j_gene = exact_identity.split(KEY_SEPARATOR, 2)
    result = _identity(v_gene, cdr3aa, j_gene)
    if result is None:
        raise ValueError(f"invalid exact clonotype identity: {exact_identity!r}")
    return result


def _stable_hash(identity: str, seed: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{identity}".encode("utf-8"), digest_size=16).digest()


def _nearest_publicness_controls(
    anchors: list[str],
    publicness: dict[str, int],
    *,
    maximum_controls: int,
    seed: int,
) -> dict[str, list[str]]:
    """Choose V/length-matched controls nearest in donor publicness.

    Controls are selected deterministically.  The query patient is never
    filtered by publicness; publicness is used solely for anchor calibration.
    """
    if maximum_controls < 2:
        raise ValueError("maximum_controls must be at least two")
    anchor_set = set(anchors)
    pools: dict[tuple[str, int], list[str]] = defaultdict(list)
    for identity in publicness:
        if identity in anchor_set:
            continue
        v_gene, cdr3aa, _ = identity.split(KEY_SEPARATOR, 2)
        if v_gene != "unresolved":
            pools[(v_gene, len(cdr3aa))].append(identity)
    result: dict[str, list[str]] = {}
    for anchor in anchors:
        v_gene, cdr3aa, _ = anchor.split(KEY_SEPARATOR, 2)
        candidates = pools[(v_gene, len(cdr3aa))]
        ordered = sorted(
            candidates,
            key=lambda identity: (
                abs(publicness[identity] - publicness[anchor]),
                _stable_hash(identity, seed),
            ),
        )
        if len(ordered) >= maximum_controls:
            result[anchor] = ordered[:maximum_controls]
    return result


def _reference_identities(
    publicness: dict[str, int],
    anchors: list[str],
    *,
    limit: int,
    seed: int,
) -> list[str]:
    """Return a deterministic, outcome-blind public reference set."""
    if limit < len(anchors):
        raise ValueError("reference_limit must be at least the number of retained anchors")
    anchor_set = set(anchors)
    candidates = [identity for identity in publicness if identity not in anchor_set and identity.split(KEY_SEPARATOR, 1)[0] != "unresolved"]
    selected = sorted(candidates, key=lambda identity: _stable_hash(identity, seed))[: limit - len(anchors)]
    return sorted([*anchors, *selected])


def _load_embeddings(files: list[Path], wanted: set[str], embedding_columns: list[str]) -> dict[str, np.ndarray]:
    """Load one embedding per selected identity from training clouds only."""
    found: dict[str, np.ndarray] = {}
    columns = [*_IDENTITY_COLUMNS, *embedding_columns]
    for index, path in enumerate(files, start=1):
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=16_384):
            frame = batch.to_pandas()
            for row in frame.itertuples(index=False, name=None):
                v_gene, cdr3aa, j_gene, *values = row
                identity = _identity(v_gene, cdr3aa, j_gene)
                if identity in wanted and identity not in found:
                    found[identity] = np.asarray(values, dtype=np.float32)
        if index % 25 == 0 or index == len(files):
            print(f"reference embeddings: {index}/{len(files)} clouds, {len(found):,}/{len(wanted):,} identities", flush=True)
        if len(found) == len(wanted):
            break
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} selected identities were absent from the training clouds")
    return found


def _source_keys(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    """Memoise source identities because N and E are reused across configurations."""
    if path not in cache:
        cache[path] = _productive_keys(path)
    return cache[path]


def _fit_training_state(
    source_paths: list[Path],
    cloud_paths: list[Path],
    labels: dict[str, int],
    *,
    maximum_controls: int,
    reference_limit: int,
    p_threshold: float,
    bloom_bits: int,
    seed: int,
    source_cache: dict[Path, set[str]],
) -> TrainingState | None:
    """Construct all Fisher, control, and reference objects from training donors."""
    public = _public_gate(source_paths, bloom_bits=bloom_bits)
    counts, class_sizes, _ = _count_candidates(source_paths, labels, public)
    selected = _fisher_statistics(counts, class_sizes, p_threshold=p_threshold)
    exact_anchors = set(selected["clonotype_key"])
    if not exact_anchors:
        return None
    publicness: dict[str, int] = {}
    raw_by_cloud: dict[str, str] = {}
    for raw_identity, values in counts.items():
        if values[0] + values[1] < 2:
            continue
        cloud_identity = _cloud_identity(raw_identity)
        if cloud_identity in raw_by_cloud and raw_by_cloud[cloud_identity] != raw_identity:
            raise RuntimeError("allele-normalized cloud identities collide in the source data")
        raw_by_cloud[cloud_identity] = raw_identity
        publicness[cloud_identity] = int(values[0] + values[1])
    cloud_anchor_map = {_cloud_identity(identity): identity for identity in exact_anchors}
    cloud_anchors = sorted(cloud_anchor_map)
    controls = _nearest_publicness_controls(cloud_anchors, publicness, maximum_controls=maximum_controls, seed=seed)
    cloud_anchors = sorted(controls)
    if not cloud_anchors:
        return None
    exact_anchors = {cloud_anchor_map[identity] for identity in cloud_anchors}
    reference_ids = _reference_identities(publicness, cloud_anchors, limit=reference_limit, seed=seed)
    embedding_columns = _embedding_columns(cloud_paths[0])
    wanted = set(reference_ids)
    wanted.update(identity for values in controls.values() for identity in values)
    embeddings = _load_embeddings(cloud_paths, wanted, embedding_columns)
    anchor_vectors = _l2_normalise(np.vstack([embeddings[identity] for identity in cloud_anchors]))
    reference_vectors = _l2_normalise(np.vstack([embeddings[identity] for identity in reference_ids]))
    reference_positions = {identity: index for index, identity in enumerate(reference_ids)}
    training_counts = _exact_counts(source_paths, labels, exact_anchors, source_cache)
    negative_model = _fit_beta_binomial(
        training_counts.loc[training_counts["cmv_label"].eq(0), "n_unique_productive"].to_numpy(float),
        training_counts.loc[training_counts["cmv_label"].eq(0), "k_exact"].to_numpy(float),
    )
    positive_model = _fit_beta_binomial(
        training_counts.loc[training_counts["cmv_label"].eq(1), "n_unique_productive"].to_numpy(float),
        training_counts.loc[training_counts["cmv_label"].eq(1), "k_exact"].to_numpy(float),
    )
    return TrainingState(exact_anchors, cloud_anchors, anchor_vectors, controls, embeddings, np.asarray(reference_ids, dtype=object), reference_vectors, reference_positions, embedding_columns, negative_model, positive_model)


def _exact_counts(paths: list[Path], labels: dict[str, int], anchors: set[str], cache: dict[Path, set[str]]) -> pd.DataFrame:
    """Count literal exact anchors and total productive identities for each donor."""
    rows = []
    for path in paths:
        identities = _source_keys(path, cache)
        rows.append({"sample": path.stem, "cmv_label": labels[path.stem], "n_unique_productive": len(identities), "k_exact": len(identities & anchors)})
    return pd.DataFrame(rows)


def _nearest_reference_sets(
    query_ids: list[str],
    query_values: np.ndarray,
    state: TrainingState,
    neighbours: int,
) -> list[frozenset[str]]:
    """Return kNN identities in the frozen training reference, excluding self."""
    if not 1 <= neighbours < len(state.reference_ids):
        raise ValueError("neighbours must be positive and smaller than the reference set")
    values = _l2_normalise(query_values)
    similarity = values @ state.reference_vectors.T
    for row, identity in enumerate(query_ids):
        position = state.reference_positions.get(identity)
        if position is not None:
            similarity[row, position] = -np.inf
    indices = np.argpartition(similarity, kth=similarity.shape[1] - neighbours, axis=1)[:, -neighbours:]
    return [frozenset(map(str, state.reference_ids[row])) for row in indices]


def _rule(state: TrainingState, configuration: LocalConfiguration) -> tuple[np.ndarray, np.ndarray | None, list[frozenset[str]] | None]:
    """Derive matched direct and, when needed, topology thresholds for one configuration."""
    if configuration.method == "exact":
        raise ValueError("exact has no local rule")
    assert configuration.controls is not None and configuration.direct_rank is not None
    if configuration.controls > min(map(len, state.controls_by_anchor.values())):
        raise ValueError("configuration requests unavailable controls")
    if not 1 <= configuration.direct_rank <= configuration.controls:
        raise ValueError("direct_rank must lie in [1, controls]")
    direct = []
    control_ids: list[str] = []
    for index, anchor in enumerate(state.cloud_anchors):
        chosen = state.controls_by_anchor[anchor][: configuration.controls]
        control_ids.extend(chosen)
        values = _l2_normalise(np.vstack([state.embeddings[identity] for identity in chosen]))
        distances = np.sort(1.0 - values @ state.anchor_vectors[index])
        direct.append(float(distances[configuration.direct_rank - 1]))
    if configuration.method == "direct":
        return np.asarray(direct, dtype=np.float32), None, None
    assert configuration.neighbours is not None and configuration.topology_rank is not None
    if not 1 <= configuration.topology_rank <= configuration.controls:
        raise ValueError("topology_rank must lie in [1, controls]")
    anchor_sets = _nearest_reference_sets(state.cloud_anchors, state.anchor_vectors, state, configuration.neighbours)
    control_values = np.vstack([state.embeddings[identity] for identity in control_ids])
    control_sets = _nearest_reference_sets(control_ids, control_values, state, configuration.neighbours)
    thresholds = []
    cursor = 0
    for index in range(len(state.cloud_anchors)):
        supports = sorted((len(anchor_sets[index] & control_sets[cursor + offset]) for offset in range(configuration.controls)), reverse=True)
        thresholds.append(float(supports[configuration.topology_rank - 1]))
        cursor += configuration.controls
    return np.asarray(direct, dtype=np.float32), np.asarray(thresholds, dtype=np.float32), anchor_sets


def _local_features(
    cloud_path: Path,
    state: TrainingState,
    configuration: LocalConfiguration,
    *,
    query_neighbour_cache: dict[tuple[str, int], frozenset[str]],
) -> tuple[int, int]:
    """Count distinct direct and direct-plus-topology modules in one donor cloud."""
    if configuration.method == "exact":
        return 0, 0
    direct_thresholds, topology_thresholds, anchor_sets = _rule(state, configuration)
    direct_modules: set[int] = set()
    topology_modules: set[int] = set()
    anchor_set = set(state.cloud_anchors)
    columns = [*_IDENTITY_COLUMNS, *state.embedding_columns]
    for batch in pq.ParquetFile(cloud_path).iter_batches(columns=columns, batch_size=16_384):
        frame = batch.to_pandas()
        identities = [_identity(row[0], row[1], row[2]) for row in frame.itertuples(index=False, name=None)]
        keep = [index for index, identity in enumerate(identities) if identity is not None and identity not in anchor_set]
        if not keep:
            continue
        values = _l2_normalise(frame.iloc[keep, 3:].to_numpy(dtype=np.float32))
        distances = 1.0 - values @ state.anchor_vectors.T
        eligible = distances <= direct_thresholds[None, :]
        for local_index, source_index in enumerate(keep):
            if not eligible[local_index].any():
                continue
            assignment = int(np.argmin(np.where(eligible[local_index], distances[local_index], np.inf)))
            direct_modules.add(assignment)
            if configuration.method == "topology":
                assert topology_thresholds is not None and anchor_sets is not None and configuration.neighbours is not None
                identity = str(identities[source_index])
                cache_key = (identity, configuration.neighbours)
                neighbours = query_neighbour_cache.get(cache_key)
                if neighbours is None:
                    neighbours = _nearest_reference_sets([identity], values[local_index : local_index + 1], state, configuration.neighbours)[0]
                    query_neighbour_cache[cache_key] = neighbours
                if len(anchor_sets[assignment] & neighbours) >= topology_thresholds[assignment]:
                    topology_modules.add(assignment)
    return len(direct_modules), len(topology_modules)


def _feature_table(
    source_paths: list[Path],
    cloud_paths: dict[str, Path],
    labels: dict[str, int],
    state: TrainingState | None,
    configuration: LocalConfiguration,
    source_cache: dict[Path, set[str]],
) -> pd.DataFrame:
    """Create blinded donor features; labels are carried only for later AUROC."""
    if state is None:
        rows = [{"sample": path.stem, "cmv_label": labels[path.stem], "n_unique_productive": len(_source_keys(path, source_cache)), "k_exact": 0, "exact_probability": 0.5, "local_direct": np.nan, "local_topology": np.nan, "n_modules": 0} for path in source_paths]
        return pd.DataFrame(rows)
    exact = _exact_counts(source_paths, labels, state.exact_anchors, source_cache)
    exact["exact_probability"] = _posterior_probability(exact["n_unique_productive"].to_numpy(float), exact["k_exact"].to_numpy(float), state.negative_model, state.positive_model)
    neighbours: dict[tuple[str, int], frozenset[str]] = {}
    direct, topology = [], []
    for sample in exact["sample"]:
        value_direct, value_topology = _local_features(cloud_paths[str(sample)], state, configuration, query_neighbour_cache=neighbours)
        direct.append(value_direct / len(state.cloud_anchors))
        topology.append(value_topology / len(state.cloud_anchors))
    exact["local_direct"] = direct
    exact["local_topology"] = topology
    exact["n_modules"] = len(state.cloud_anchors)
    return exact


def _probability_log_odds(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-8, 1 - 1e-8)
    return np.log(clipped / (1 - clipped))


def _fit_predict(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    configuration: LocalConfiguration,
) -> np.ndarray:
    """Fit the selected donor model on labelled training donors and blind-score donors."""
    if configuration.method == "exact":
        return scoring["exact_probability"].to_numpy(float)
    local_column = "local_direct" if configuration.method == "direct" else "local_topology"
    if training[local_column].isna().any() or scoring[local_column].isna().any():
        raise RuntimeError("embedding extension is unavailable because no valid training modules were retained")
    x_train = np.column_stack((_probability_log_odds(training["exact_probability"].to_numpy(float)), training[local_column].to_numpy(float)))
    x_score = np.column_stack((_probability_log_odds(scoring["exact_probability"].to_numpy(float)), scoring[local_column].to_numpy(float)))
    classifier = LogisticRegression(C=float(configuration.logistic_c), penalty="l2", solver="liblinear", max_iter=10_000, random_state=0)
    classifier.fit(x_train, training["cmv_label"].to_numpy(int))
    return classifier.predict_proba(x_score)[:, 1]


def _configurations(
    controls: tuple[int, ...],
    direct_ranks: tuple[int, ...],
    neighbours: tuple[int, ...],
    topology_ranks: tuple[int, ...],
    logistic_cs: tuple[float, ...],
) -> list[LocalConfiguration]:
    """Enumerate the exact, direct, and direct-plus-topology comparators."""
    result = [LocalConfiguration("exact")]
    for control_count in controls:
        for direct_rank in direct_ranks:
            if direct_rank > control_count:
                continue
            for value_c in logistic_cs:
                result.append(LocalConfiguration("direct", control_count, direct_rank, logistic_c=value_c))
                for value_k in neighbours:
                    for topology_rank in topology_ranks:
                        if topology_rank <= control_count:
                            result.append(LocalConfiguration("topology", control_count, direct_rank, value_k, topology_rank, value_c))
    return result


def _select_configuration(rows: pd.DataFrame) -> LocalConfiguration:
    """Select maximum mean inner AUROC with the predeclared parsimonious tie-break."""
    summary = rows.groupby(["method", "controls", "direct_rank", "neighbours", "topology_rank", "logistic_c"], dropna=False, as_index=False).agg(mean_inner_auroc=("auroc", "mean"), n_inner_folds=("auroc", "size"))
    valid = summary.loc[np.isfinite(summary["mean_inner_auroc"])].copy()
    if valid.empty:
        return LocalConfiguration("exact")
    choices = []
    for row in valid.itertuples(index=False):
        configuration = LocalConfiguration(row.method, None if pd.isna(row.controls) else int(row.controls), None if pd.isna(row.direct_rank) else int(row.direct_rank), None if pd.isna(row.neighbours) else int(row.neighbours), None if pd.isna(row.topology_rank) else int(row.topology_rank), None if pd.isna(row.logistic_c) else float(row.logistic_c))
        choices.append((float(row.mean_inner_auroc), configuration.complexity_key(), configuration))
    best_auc = max(value[0] for value in choices)
    return min((value for value in choices if np.isclose(value[0], best_auc)), key=lambda value: value[1])[2]


def _paths_and_labels(source_dir: Path, clouds_dir: Path, pattern: str) -> tuple[list[Path], dict[str, Path], dict[str, int]]:
    metadata = _metadata(source_dir, pattern)
    labels = {row.sample: int(row.cmv_status == "positive") for row in metadata.itertuples() if row.cmv_status in {"positive", "negative"}}
    paths = [path for path in sorted(source_dir.glob(pattern)) if path.stem in labels]
    clouds = {path.stem: path for path in clouds_dir.glob("*.parquet") if path.stem in labels}
    if set(path.stem for path in paths) != set(clouds):
        raise RuntimeError(f"{pattern}: CMV-labelled source and cloud sample sets differ")
    return paths, clouds, labels


def run_nested_embedding_evaluation(
    clouds_dir: str | Path,
    source_dir: str | Path,
    temporary_dir: str | Path,
    results_dir: str | Path,
    *,
    outer_folds: int = 4,
    inner_folds: int = 3,
    controls: tuple[int, ...] = (8, 16, 32),
    direct_ranks: tuple[int, ...] = (1, 2),
    neighbours: tuple[int, ...] = (10, 20, 40),
    topology_ranks: tuple[int, ...] = (1, 2),
    logistic_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    reference_limit: int = 50_000,
    p_threshold: float = 1e-4,
    bloom_bits: int = 1 << 33,
    seed: int = 0,
) -> dict[str, object]:
    """Run nested P validation and one frozen full-P Keck evaluation.

    ``reference_limit`` is deliberately explicit.  It chooses a deterministic
    hash-selected subset of public training identities, never a pool containing
    a validation, outer-test, or Keck clonotype.
    """
    clouds, source = Path(clouds_dir), Path(source_dir)
    temporary, results = Path(temporary_dir), Path(results_dir)
    if temporary.exists() or results.exists():
        raise FileExistsError("temporary_dir and results_dir must be new")
    if not controls or not direct_ranks or not neighbours or not topology_ranks or not logistic_cs:
        raise ValueError("all candidate parameter grids must be non-empty")
    if reference_limit < max(neighbours) + 1:
        raise ValueError("reference_limit must exceed the largest neighbour count")
    p_paths, p_clouds, p_labels = _paths_and_labels(source, clouds, "P*.tsv")
    keck_paths, keck_clouds, keck_labels = _paths_and_labels(source, clouds, "Keck*.tsv")
    temporary.mkdir(parents=True)
    configurations = _configurations(controls, direct_ranks, neighbours, topology_ranks, logistic_cs)
    maximum_controls = max(controls)
    source_cache: dict[Path, set[str]] = {}
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[pd.DataFrame] = []
    outer_choices: list[dict[str, object]] = []
    for outer_index, (outer_train, outer_test) in enumerate(_stratified_folds(p_labels, outer_folds, seed)):
        inner_labels = {sample: p_labels[sample] for sample in outer_train}
        for inner_index, (inner_train, inner_validation) in enumerate(_stratified_folds(inner_labels, inner_folds, seed + outer_index + 1)):
            train_paths = [path for path in p_paths if path.stem in inner_train]
            validation_paths = [path for path in p_paths if path.stem in inner_validation]
            state = _fit_training_state(train_paths, [p_clouds[path.stem] for path in train_paths], {sample: p_labels[sample] for sample in inner_train}, maximum_controls=maximum_controls, reference_limit=reference_limit, p_threshold=p_threshold, bloom_bits=bloom_bits, seed=seed + 10_000 * outer_index + inner_index, source_cache=source_cache)
            for configuration in configurations:
                try:
                    training = _feature_table(train_paths, p_clouds, p_labels, state, configuration, source_cache)
                    validation = _feature_table(validation_paths, p_clouds, p_labels, state, configuration, source_cache)
                    score = _fit_predict(training, validation, configuration)
                    value = float(roc_auc_score(validation["cmv_label"], score))
                    status = "evaluated"
                except (RuntimeError, ValueError) as exc:
                    value, status = np.nan, type(exc).__name__
                inner_rows.append({"outer_fold": outer_index, "inner_fold": inner_index, **configuration.as_dict(), "auroc": value, "status": status})
        inner_frame = pd.DataFrame([row for row in inner_rows if row["outer_fold"] == outer_index])
        selected = _select_configuration(inner_frame.loc[inner_frame["status"].eq("evaluated")])
        outer_choices.append({"outer_fold": outer_index, **selected.as_dict()})
        train_paths = [path for path in p_paths if path.stem in outer_train]
        test_paths = [path for path in p_paths if path.stem in outer_test]
        state = _fit_training_state(train_paths, [p_clouds[path.stem] for path in train_paths], {sample: p_labels[sample] for sample in outer_train}, maximum_controls=maximum_controls, reference_limit=reference_limit, p_threshold=p_threshold, bloom_bits=bloom_bits, seed=seed + 100_000 + outer_index, source_cache=source_cache)
        train_features = _feature_table(train_paths, p_clouds, p_labels, state, selected, source_cache)
        test_features = _feature_table(test_paths, p_clouds, p_labels, state, selected, source_cache)
        selected_score = _fit_predict(train_features, test_features, selected)
        exact_score = _fit_predict(train_features, test_features, LocalConfiguration("exact"))
        test_features = test_features.assign(outer_fold=outer_index, selected_score=selected_score, exact_score=exact_score, selected_method=selected.method)
        outer_rows.append(test_features)
        print(f"nested CMV evaluation: outer fold {outer_index + 1}/{outer_folds}, selected {selected.method}", flush=True)
    inner_frame = pd.DataFrame(inner_rows)
    outer_frame = pd.concat(outer_rows, ignore_index=True)
    global_configuration = _select_configuration(inner_frame.loc[inner_frame["status"].eq("evaluated")])
    all_state = _fit_training_state(p_paths, [p_clouds[path.stem] for path in p_paths], p_labels, maximum_controls=maximum_controls, reference_limit=reference_limit, p_threshold=p_threshold, bloom_bits=bloom_bits, seed=seed + 1_000_000, source_cache=source_cache)
    all_training = _feature_table(p_paths, p_clouds, p_labels, all_state, global_configuration, source_cache)
    keck_features = _feature_table(keck_paths, keck_clouds, keck_labels, all_state, global_configuration, source_cache)
    keck_features["selected_score"] = _fit_predict(all_training, keck_features, global_configuration)
    keck_features["exact_score"] = _fit_predict(all_training, keck_features, LocalConfiguration("exact"))
    results.mkdir(parents=True)
    inner_frame.to_csv(temporary / "emerson_cmv_embedding_inner_folds.tsv", sep="\t", index=False)
    pd.DataFrame(outer_choices).to_csv(temporary / "emerson_cmv_embedding_outer_choices.tsv", sep="\t", index=False)
    outer_frame.to_csv(results / "emerson_cmv_embedding_p_oof_scores.tsv", sep="\t", index=False)
    keck_features.to_csv(results / "emerson_cmv_embedding_keck_scores.tsv", sep="\t", index=False)
    manifest = {
        "procedure": "nested P-only Fisher anchor discovery; training-only matched direct thresholds and shared-neighbour support; frozen external Keck score",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "controls": list(controls),
        "direct_ranks": list(direct_ranks),
        "neighbours": list(neighbours),
        "topology_ranks": list(topology_ranks),
        "logistic_cs": list(logistic_cs),
        "reference_limit": reference_limit,
        "reference_selection": "deterministic training-only hash selection among public identities",
        "p_threshold": p_threshold,
        "bloom_bits": bloom_bits,
        "seed": seed,
        "selected_full_p_configuration": global_configuration.as_dict(),
        "p_oof_exact_auroc": float(roc_auc_score(outer_frame["cmv_label"], outer_frame["exact_score"])),
        "p_oof_selected_auroc": float(roc_auc_score(outer_frame["cmv_label"], outer_frame["selected_score"])),
        "keck_exact_auroc": float(roc_auc_score(keck_features["cmv_label"], keck_features["exact_score"])),
        "keck_selected_auroc": float(roc_auc_score(keck_features["cmv_label"], keck_features["selected_score"])),
    }
    (results / "emerson_cmv_embedding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds-dir", required=True)
    parser.add_argument("--emerson-tsv", required=True)
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--controls", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--direct-ranks", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--neighbours", type=int, nargs="+", default=(10, 20, 40))
    parser.add_argument("--topology-ranks", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--logistic-cs", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--reference-limit", type=int, required=True)
    parser.add_argument("--p-threshold", type=float, default=1e-4)
    parser.add_argument("--bloom-bits", type=int, default=1 << 33)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_nested_embedding_evaluation(args.clouds_dir, args.emerson_tsv, args.tmp, args.results, outer_folds=args.outer_folds, inner_folds=args.inner_folds, controls=tuple(args.controls), direct_ranks=tuple(args.direct_ranks), neighbours=tuple(args.neighbours), topology_ranks=tuple(args.topology_ranks), logistic_cs=tuple(args.logistic_cs), reference_limit=args.reference_limit, p_threshold=args.p_threshold, bloom_bits=args.bloom_bits, seed=args.seed)


if __name__ == "__main__":
    main()
