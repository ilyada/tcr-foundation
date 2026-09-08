"""Build aligned raw/OAR repertoire descriptors from saved embedding clouds.

The module is deliberately independent of model inference: a cloud already contains the
Foundation-model coordinates.  It aligns every raw/OAR pair by clonotype identity, verifies that
the coordinates are equal, and changes only the saved ``w_log`` abundance weights.  Three
descriptor families are available:

* ``mean``: weighted first moment of the unit-normalised cloud;
* ``mean_cov``: weighted mean followed by the isometric upper triangle of its covariance;
* ``occupancy``: weighted soft-assignment mass over one or more fixed prototype codebooks.

Consequently, comparisons between raw and OAR outputs have a single controlled difference: the
abundance weights.  The builder keeps durable outputs compact (one table per branch and
descriptor plus a manifest); optional pair-level QC is intended only for temporary run folders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_IDENTITY_COLUMNS = ("sample", "chain", "v_gene", "j_gene", "cdr3aa")
_SUPPORTED_KINDS = ("mean", "mean_cov", "occupancy")


def embedding_columns(path: Path, prefix: str) -> list[str]:
    """Return numerically ordered embedding columns without reading cloud values."""
    names = pq.ParquetFile(path).schema.names
    columns = [name for name in names if name.startswith(prefix) and name[len(prefix):].isdigit()]
    if not columns:
        raise ValueError(f"{path}: no embedding columns with prefix {prefix!r}")
    return sorted(columns, key=lambda name: int(name[len(prefix):]))


def normalise_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalise cloud rows and reject invalid embeddings."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("embeddings must be finite and have positive norm")
    return values / norms


def normalised_weights(values: np.ndarray, *, path: Path) -> np.ndarray:
    """Validate and turn stored non-negative abundance values into probability weights."""
    weights = np.asarray(values, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{path}: w_log must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{path}: w_log must have positive total mass")
    return weights / total


def effective_n(weights: np.ndarray) -> float:
    """Return the inverse concentration of an already normalised abundance distribution."""
    return float(1.0 / np.square(weights).sum())


def align_oar_to_raw(raw: pd.DataFrame, oar: pd.DataFrame, *, raw_path: Path, oar_path: Path) -> pd.DataFrame:
    """Reorder OAR rows to raw order while retaining repeated clonotype rows exactly."""
    for frame, path in ((raw, raw_path), (oar, oar_path)):
        missing = sorted(set(_IDENTITY_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing identity columns {missing}")
    occurrence = "_paired_occurrence"
    raw_keyed = raw.assign(**{occurrence: raw.groupby(list(_IDENTITY_COLUMNS), sort=False, observed=True).cumcount()})
    oar_keyed = oar.assign(**{occurrence: oar.groupby(list(_IDENTITY_COLUMNS), sort=False, observed=True).cumcount()})
    paired_columns = [*_IDENTITY_COLUMNS, occurrence]
    raw_index = pd.MultiIndex.from_frame(raw_keyed.loc[:, paired_columns])
    oar_index = pd.MultiIndex.from_frame(oar_keyed.loc[:, paired_columns])
    if len(raw_index) != len(oar_index) or not raw_index.isin(oar_index).all() or not oar_index.isin(raw_index).all():
        raise ValueError(f"raw/OAR cloud identities differ: {raw_path.name} vs {oar_path.name}")
    return oar_keyed.set_index(paired_columns).loc[raw_index].reset_index().drop(columns=occurrence)


def weighted_mean(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the unit-normalised weighted first moment of a cloud."""
    mean = weights @ normalise_rows(embeddings)
    norm = float(np.linalg.norm(mean))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("weighted cloud mean has zero or invalid norm")
    return (mean / norm).astype(np.float32)


def weighted_mean_cov(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return weighted mean plus isometric upper-triangular covariance, L2-normalised.

    The calculation is numerically equivalent to the established repertoire
    ``mean_cov_weighted_np`` convention, including the ``sqrt(2)`` off-diagonal scaling.
    """
    values = normalise_rows(embeddings).astype(np.float64)
    mean = weights @ values
    centered = values - mean
    covariance = (weights[:, None] * centered).T @ centered
    upper = np.triu_indices(values.shape[1])
    packed = covariance[upper].copy()
    packed[upper[0] != upper[1]] *= np.sqrt(2.0)
    descriptor = np.concatenate((mean, packed))
    norm = float(np.linalg.norm(descriptor))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("weighted mean/covariance descriptor has zero or invalid norm")
    return (descriptor / norm).astype(np.float32)


def soft_assignments(embeddings: np.ndarray, prototypes: np.ndarray, *, temperature: float, chunk_size: int) -> np.ndarray:
    """Compute stable temperature-scaled soft prototype assignments in bounded batches."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    values = normalise_rows(embeddings)
    prototypes = normalise_rows(prototypes)
    assignments = np.empty((len(values), len(prototypes)), dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        logits = (values[start:stop] @ prototypes.T) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        assignments[start:stop] = logits / logits.sum(axis=1, keepdims=True)
    return assignments


def load_codebooks(paths: list[str | Path], temperature: float | None) -> tuple[dict[int, np.ndarray], float, dict[str, object]]:
    """Read prototype codebooks and their minimal provenance."""
    codebooks: dict[int, np.ndarray] = {}
    observed_temperatures: set[float] = set()
    provenance: dict[str, object] = {}
    for supplied in paths:
        directory = Path(supplied)
        payload_path, manifest_path = directory / "prototypes.npz", directory / "manifest.json"
        if not payload_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"{directory}: expected prototypes.npz and manifest.json")
        payload = np.load(payload_path)
        if not {"centroids", "clusters", "temperature"}.issubset(payload.files):
            raise ValueError(f"{payload_path}: missing a required codebook field")
        centroids = np.asarray(payload["centroids"], dtype=np.float32)
        clusters = int(np.asarray(payload["clusters"]).item())
        stored_temperature = float(np.asarray(payload["temperature"]).item())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if centroids.ndim != 2 or centroids.shape[0] != clusters:
            raise ValueError(f"{payload_path}: centroid shape is incompatible with clusters={clusters}")
        if int(manifest.get("clusters", -1)) != clusters:
            raise ValueError(f"{directory}: manifest and codebook disagree on the cluster count")
        if clusters in codebooks:
            raise ValueError(f"duplicate codebook for K={clusters}")
        codebooks[clusters] = normalise_rows(centroids)
        observed_temperatures.add(stored_temperature)
        provenance[str(clusters)] = {"directory": str(directory), "selected_identities": int(manifest["selected_identities"]), "embedding_dim": int(manifest["embedding_dim"])}
    if not codebooks:
        raise ValueError("occupancy descriptors require at least one codebook")
    if temperature is None:
        if len(observed_temperatures) != 1:
            raise ValueError("codebooks have different stored temperatures; supply --temperature explicitly")
        temperature = observed_temperatures.pop()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return dict(sorted(codebooks.items())), float(temperature), provenance


def _descriptor_row(*, source: str, sample: str, branch: str, n_clonotypes: int, n_eff: float, values: np.ndarray) -> dict[str, object]:
    return {"source": source, "sample": sample, "branch": branch, "n_clonotypes": n_clonotypes, "n_eff": n_eff, **{f"d{index}": float(value) for index, value in enumerate(values)}}


def build_paired_descriptors(
    raw_clouds: str | Path,
    oar_clouds: str | Path,
    out_dir: str | Path,
    *,
    kinds: tuple[str, ...] = ("mean", "mean_cov"),
    codebook_dirs: list[str | Path] | None = None,
    temperature: float | None = None,
    embedding_prefix: str = "e",
    chunk_size: int = 8192,
    limit: int | None = None,
    write_pair_qc: bool = False,
) -> dict[str, object]:
    """Write aligned raw/OAR tables for selected descriptor families.

    ``occupancy`` reuses the supplied codebooks and writes one table per ``K``.  Moment
    descriptors do not require reference points and therefore remain inexpensive in memory.
    """
    unknown = sorted(set(kinds) - set(_SUPPORTED_KINDS))
    if not kinds or unknown:
        raise ValueError(f"kinds must be a non-empty subset of {_SUPPORTED_KINDS}; unknown={unknown}")
    if len(set(kinds)) != len(kinds):
        raise ValueError("descriptor kinds must not be repeated")
    raw_root, oar_root, output = Path(raw_clouds), Path(oar_clouds), Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    raw_files = {path.name: path for path in raw_root.glob("P*.parquet")}
    oar_files = {path.name: path for path in oar_root.glob("P*.parquet")}
    if not raw_files or not oar_files:
        raise FileNotFoundError("raw and OAR directories must both contain P*.parquet clouds")
    if raw_files.keys() != oar_files.keys():
        only_raw, only_oar = sorted(raw_files.keys() - oar_files.keys()), sorted(oar_files.keys() - raw_files.keys())
        raise ValueError(f"raw/OAR file sets differ; raw-only={only_raw[:3]}, oar-only={only_oar[:3]}")
    names = sorted(raw_files)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        names = names[:limit]
    columns = embedding_columns(raw_files[names[0]], embedding_prefix)
    codebooks: dict[int, np.ndarray] = {}
    provenance: dict[str, object] = {}
    resolved_temperature: float | None = None
    if "occupancy" in kinds:
        codebooks, resolved_temperature, provenance = load_codebooks(codebook_dirs or [], temperature)
        for clusters, centroids in codebooks.items():
            if centroids.shape[1] != len(columns):
                raise ValueError(f"K={clusters}: centroid dimension {centroids.shape[1]} does not match cloud dimension {len(columns)}")
    elif codebook_dirs:
        raise ValueError("codebooks were supplied without requesting the occupancy descriptor")

    rows: dict[str, dict[str, list[dict[str, object]]]] = {kind: {"raw": [], "oar": []} for kind in kinds}
    qc_rows: list[dict[str, object]] = []
    read_columns = [*_IDENTITY_COLUMNS, "w_log", *columns]
    output.mkdir(parents=True)
    for index, name in enumerate(names, start=1):
        raw_path, oar_path = raw_files[name], oar_files[name]
        raw = pd.read_parquet(raw_path, columns=read_columns)
        oar = align_oar_to_raw(raw, pd.read_parquet(oar_path, columns=read_columns), raw_path=raw_path, oar_path=oar_path)
        samples = raw["sample"].astype(str).unique()
        if len(samples) != 1 or set(samples) != set(oar["sample"].astype(str).unique()):
            raise ValueError(f"{name}: expected one identical sample identifier in the paired clouds")
        raw_embeddings = raw.loc[:, columns].to_numpy(dtype=np.float32)
        oar_embeddings = oar.loc[:, columns].to_numpy(dtype=np.float32)
        maximum_difference = float(np.max(np.abs(raw_embeddings - oar_embeddings))) if len(raw_embeddings) else 0.0
        if maximum_difference > 1e-6:
            raise ValueError(f"{name}: raw/OAR embedding mismatch (max absolute difference {maximum_difference})")
        raw_weights = normalised_weights(raw["w_log"].to_numpy(), path=raw_path)
        oar_weights = normalised_weights(oar["w_log"].to_numpy(), path=oar_path)
        shared = {"source": Path(name).stem, "sample": samples[0], "n_clonotypes": int(len(raw))}
        raw_values: dict[str, np.ndarray] = {}
        oar_values: dict[str, np.ndarray] = {}
        if "mean" in kinds:
            raw_values["mean"], oar_values["mean"] = weighted_mean(raw_embeddings, raw_weights), weighted_mean(raw_embeddings, oar_weights)
        if "mean_cov" in kinds:
            raw_values["mean_cov"], oar_values["mean_cov"] = weighted_mean_cov(raw_embeddings, raw_weights), weighted_mean_cov(raw_embeddings, oar_weights)
        if "occupancy" in kinds:
            for clusters, prototypes in codebooks.items():
                assignment = soft_assignments(raw_embeddings, prototypes, temperature=resolved_temperature, chunk_size=chunk_size)
                raw_values[f"occupancy_k{clusters}"] = raw_weights @ assignment
                oar_values[f"occupancy_k{clusters}"] = oar_weights @ assignment
        for kind in ("mean", "mean_cov"):
            if kind in kinds:
                rows[kind]["raw"].append(_descriptor_row(**shared, branch="raw", n_eff=effective_n(raw_weights), values=raw_values[kind]))
                rows[kind]["oar"].append(_descriptor_row(**shared, branch="oar", n_eff=effective_n(oar_weights), values=oar_values[kind]))
        if "occupancy" in kinds:
            for clusters in codebooks:
                key = f"occupancy_k{clusters}"
                rows[key] = rows.get(key, {"raw": [], "oar": []})
                rows[key]["raw"].append(_descriptor_row(**shared, branch="raw", n_eff=effective_n(raw_weights), values=raw_values[key]))
                rows[key]["oar"].append(_descriptor_row(**shared, branch="oar", n_eff=effective_n(oar_weights), values=oar_values[key]))
        if write_pair_qc:
            for name_key in [*(["mean"] if "mean" in kinds else []), *(["mean_cov"] if "mean_cov" in kinds else []), *(f"occupancy_k{clusters}" for clusters in codebooks if "occupancy" in kinds)]:
                qc_rows.append({**shared, "descriptor": name_key, "n_eff_raw": effective_n(raw_weights), "n_eff_oar": effective_n(oar_weights), "max_embedding_abs_diff": maximum_difference, "raw_oar_l1": float(np.abs(raw_values[name_key] - oar_values[name_key]).sum())})
        if index % 25 == 0 or index == len(names):
            print(f"[{index}/{len(names)}] paired clouds processed", flush=True)

    table_names: list[str] = []
    for kind in ("mean", "mean_cov"):
        if kind in kinds:
            for branch in ("raw", "oar"):
                filename = f"{kind}_{branch}.parquet"
                pd.DataFrame(rows[kind][branch]).to_parquet(output / filename, index=False)
                table_names.append(filename)
    if "occupancy" in kinds:
        for clusters in codebooks:
            key = f"occupancy_k{clusters}"
            for branch in ("raw", "oar"):
                filename = f"{key}_{branch}.parquet"
                pd.DataFrame(rows[key][branch]).to_parquet(output / filename, index=False)
                table_names.append(filename)
    if write_pair_qc:
        pd.DataFrame(qc_rows).to_parquet(output / "pair_qc.parquet", index=False)
        table_names.append("pair_qc.parquet")
    manifest: dict[str, object] = {"raw_clouds": str(raw_root), "oar_clouds": str(oar_root), "n_pairs": len(names), "kinds": list(kinds), "embedding_prefix": embedding_prefix, "embedding_dim": len(columns), "chunk_size": chunk_size, "codebooks": provenance, "temperature": resolved_temperature, "tables": table_names}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-clouds", required=True)
    parser.add_argument("--oar-clouds", required=True)
    parser.add_argument("--out", required=True, help="new directory for paired descriptor tables")
    parser.add_argument("--kinds", nargs="+", choices=_SUPPORTED_KINDS, default=["mean", "mean_cov"])
    parser.add_argument("--codebooks", nargs="*", default=[], help="prototype directories; required for occupancy")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--embedding-prefix", default="e")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N paired clouds; for smoke tests")
    parser.add_argument("--write-pair-qc", action="store_true", help="write temporary pair-level validation records")
    args = parser.parse_args(argv)
    build_paired_descriptors(args.raw_clouds, args.oar_clouds, args.out, kinds=tuple(args.kinds), codebook_dirs=args.codebooks, temperature=args.temperature, embedding_prefix=args.embedding_prefix, chunk_size=args.chunk_size, limit=args.limit, write_pair_qc=args.write_pair_qc)


if __name__ == "__main__":
    main()
