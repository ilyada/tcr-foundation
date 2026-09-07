"""Construct paired raw/OAR soft-occupancy descriptors from saved repertoire clouds.

The input cloud coordinates are already frozen Foundation-model embeddings.  This module never
reloads the model or re-embeds a clonotype: it verifies the raw/OAR pairing, computes soft
prototype assignment once from their shared coordinates, and aggregates that assignment with each
branch's saved ``w_log`` weights.  Consequently, a raw/OAR descriptor difference is attributable
only to the branch weights, not to a different encoder or codebook.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


_IDENTITY_COLUMNS = ("sample", "chain", "v_gene", "j_gene", "cdr3aa")


def _embedding_columns(path: Path, prefix: str) -> list[str]:
    frame = pd.read_parquet(path, engine="pyarrow")
    columns = [name for name in frame.columns if name.startswith(prefix) and name[len(prefix):].isdigit()]
    if not columns:
        raise ValueError(f"{path}: no embedding columns with prefix {prefix!r}")
    return sorted(columns, key=lambda name: int(name[len(prefix):]))


def _load_codebooks(paths: list[str | Path], temperature: float | None) -> tuple[dict[int, np.ndarray], float, dict[str, object]]:
    codebooks: dict[int, np.ndarray] = {}
    observed_temperatures: set[float] = set()
    provenance: dict[str, object] = {}
    for supplied in paths:
        directory = Path(supplied)
        payload_path = directory / "prototypes.npz"
        manifest_path = directory / "manifest.json"
        if not payload_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"{directory}: expected prototypes.npz and manifest.json")
        payload = np.load(payload_path)
        if "centroids" not in payload or "clusters" not in payload or "temperature" not in payload:
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
        codebooks[clusters] = _normalize_rows(centroids)
        observed_temperatures.add(stored_temperature)
        provenance[str(clusters)] = {
            "directory": str(directory),
            "selected_identities": int(manifest["selected_identities"]),
            "embedding_dim": int(manifest["embedding_dim"]),
        }
    if not codebooks:
        raise ValueError("at least one codebook is required")
    if temperature is None:
        if len(observed_temperatures) != 1:
            raise ValueError("codebooks have different stored temperatures; supply --temperature explicitly")
        temperature = observed_temperatures.pop()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return dict(sorted(codebooks.items())), float(temperature), provenance


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("embeddings must be finite and have positive norm")
    return values / norms


def _normalised_weights(values: np.ndarray, *, path: Path) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{path}: w_log must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{path}: w_log must have positive total mass")
    return weights / total


def _effective_n(weights: np.ndarray) -> float:
    return float(1.0 / np.square(weights).sum())


def _align_oar_to_raw(raw: pd.DataFrame, oar: pd.DataFrame, *, raw_path: Path, oar_path: Path) -> pd.DataFrame:
    for frame, path in ((raw, raw_path), (oar, oar_path)):
        missing = sorted(set(_IDENTITY_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing identity columns {missing}")
        if frame.duplicated(list(_IDENTITY_COLUMNS)).any():
            raise ValueError(f"{path}: duplicate cloud identities prevent unambiguous raw/OAR alignment")
    raw_index = pd.MultiIndex.from_frame(raw.loc[:, _IDENTITY_COLUMNS])
    oar_index = pd.MultiIndex.from_frame(oar.loc[:, _IDENTITY_COLUMNS])
    if len(raw_index) != len(oar_index) or not raw_index.isin(oar_index).all() or not oar_index.isin(raw_index).all():
        raise ValueError(f"raw/OAR cloud identities differ: {raw_path.name} vs {oar_path.name}")
    return oar.set_index(list(_IDENTITY_COLUMNS)).loc[raw_index].reset_index()


def _soft_assignments(embeddings: np.ndarray, prototypes: np.ndarray, *, temperature: float, chunk_size: int) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    values = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    assignments = np.empty((len(values), len(prototypes)), dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        logits = (values[start:stop] @ prototypes.T) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        assignments[start:stop] = logits / logits.sum(axis=1, keepdims=True)
    return assignments


def build_paired_occupancies(
    raw_clouds: str | Path,
    oar_clouds: str | Path,
    codebook_dirs: list[str | Path],
    out_dir: str | Path,
    *,
    temperature: float | None = None,
    embedding_prefix: str = "e",
    chunk_size: int = 8192,
    limit: int | None = None,
) -> dict[str, object]:
    """Write aligned raw and OAR occupancy tables for one or more saved codebooks."""
    raw_root, oar_root, output = Path(raw_clouds), Path(oar_clouds), Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    raw_files = {path.name: path for path in raw_root.glob("P*.parquet")}
    oar_files = {path.name: path for path in oar_root.glob("P*.parquet")}
    if not raw_files or not oar_files:
        raise FileNotFoundError("raw and OAR directories must both contain P*.parquet clouds")
    if raw_files.keys() != oar_files.keys():
        only_raw = sorted(raw_files.keys() - oar_files.keys())
        only_oar = sorted(oar_files.keys() - raw_files.keys())
        raise ValueError(f"raw/OAR file sets differ; raw-only={only_raw[:3]}, oar-only={only_oar[:3]}")
    names = sorted(raw_files)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        names = names[:limit]

    codebooks, resolved_temperature, provenance = _load_codebooks(codebook_dirs, temperature)
    embedding_columns = _embedding_columns(raw_files[names[0]], embedding_prefix)
    for clusters, centroids in codebooks.items():
        if centroids.shape[1] != len(embedding_columns):
            raise ValueError(f"K={clusters}: centroid dimension {centroids.shape[1]} does not match cloud dimension {len(embedding_columns)}")

    raw_rows: dict[int, list[dict[str, object]]] = {clusters: [] for clusters in codebooks}
    oar_rows: dict[int, list[dict[str, object]]] = {clusters: [] for clusters in codebooks}
    qc_rows: dict[int, list[dict[str, object]]] = {clusters: [] for clusters in codebooks}
    read_columns = [*_IDENTITY_COLUMNS, "w_log", *embedding_columns]
    output.mkdir(parents=True)

    for index, name in enumerate(names, start=1):
        raw_path, oar_path = raw_files[name], oar_files[name]
        raw = pd.read_parquet(raw_path, columns=read_columns)
        oar = _align_oar_to_raw(raw, pd.read_parquet(oar_path, columns=read_columns), raw_path=raw_path, oar_path=oar_path)
        sample_values = raw["sample"].astype(str).unique()
        if len(sample_values) != 1 or set(sample_values) != set(oar["sample"].astype(str).unique()):
            raise ValueError(f"{name}: expected one identical sample identifier in the paired clouds")
        raw_embeddings = raw.loc[:, embedding_columns].to_numpy(dtype=np.float32)
        oar_embeddings = oar.loc[:, embedding_columns].to_numpy(dtype=np.float32)
        max_embedding_difference = float(np.max(np.abs(raw_embeddings - oar_embeddings))) if len(raw_embeddings) else 0.0
        if max_embedding_difference > 1e-6:
            raise ValueError(f"{name}: raw/OAR embedding mismatch (max absolute difference {max_embedding_difference})")
        raw_weights = _normalised_weights(raw["w_log"].to_numpy(), path=raw_path)
        oar_weights = _normalised_weights(oar["w_log"].to_numpy(), path=oar_path)
        shared = {
            "source": Path(name).stem,
            "sample": sample_values[0],
            "n_clonotypes": int(len(raw)),
            "n_eff": _effective_n(raw_weights),
        }
        for clusters, centroids in codebooks.items():
            assignment = _soft_assignments(raw_embeddings, centroids, temperature=resolved_temperature, chunk_size=chunk_size)
            raw_descriptor = raw_weights @ assignment
            oar_descriptor = oar_weights @ assignment
            columns = {f"d{item}": float(value) for item, value in enumerate(raw_descriptor)}
            raw_rows[clusters].append({**shared, "branch": "raw", **columns})
            oar_rows[clusters].append({**shared, "branch": "oar", "n_eff": _effective_n(oar_weights), **{f"d{item}": float(value) for item, value in enumerate(oar_descriptor)}})
            qc_rows[clusters].append({
                "source": Path(name).stem,
                "sample": sample_values[0],
                "n_clonotypes": int(len(raw)),
                "n_eff_raw": _effective_n(raw_weights),
                "n_eff_oar": _effective_n(oar_weights),
                "max_embedding_abs_diff": max_embedding_difference,
                "raw_oar_l1": float(np.abs(raw_descriptor - oar_descriptor).sum()),
                "raw_mass": float(raw_descriptor.sum()),
                "oar_mass": float(oar_descriptor.sum()),
            })
        if index % 25 == 0 or index == len(names):
            print(f"[{index}/{len(names)}] paired clouds processed", flush=True)

    for clusters in codebooks:
        pd.DataFrame(raw_rows[clusters]).to_parquet(output / f"occupancy_k{clusters}_raw.parquet", index=False)
        pd.DataFrame(oar_rows[clusters]).to_parquet(output / f"occupancy_k{clusters}_oar.parquet", index=False)
        pd.DataFrame(qc_rows[clusters]).to_parquet(output / f"occupancy_k{clusters}_pair_qc.parquet", index=False)
    manifest: dict[str, object] = {
        "raw_clouds": str(raw_root),
        "oar_clouds": str(oar_root),
        "n_pairs": len(names),
        "clusters": list(codebooks),
        "temperature": resolved_temperature,
        "embedding_prefix": embedding_prefix,
        "embedding_dim": len(embedding_columns),
        "chunk_size": chunk_size,
        "codebooks": provenance,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-clouds", required=True)
    parser.add_argument("--oar-clouds", required=True)
    parser.add_argument("--codebooks", nargs="+", required=True, help="directories containing prototypes.npz and manifest.json")
    parser.add_argument("--out", required=True, help="new directory for paired occupancy tables")
    parser.add_argument("--temperature", type=float, default=None, help="default: common temperature saved in codebooks")
    parser.add_argument("--embedding-prefix", default="e")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N paired clouds; intended for smoke tests")
    args = parser.parse_args()
    build_paired_occupancies(
        args.raw_clouds,
        args.oar_clouds,
        args.codebooks,
        args.out,
        temperature=args.temperature,
        embedding_prefix=args.embedding_prefix,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
