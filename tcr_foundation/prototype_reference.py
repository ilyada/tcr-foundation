"""Build a bounded-memory, publicness-stratified k-means reference for repertoire clouds.

The reference consists of beta-chain clonotype identities, not donor clouds.  Every identity is
selected by a stable BLAKE2b hash before it is retained, so its publicness can be counted exactly
over the streamed cloud files without materialising the cohort-wide clonotype table.  The selected
identities are then balanced across publicness strata before MiniBatchKMeans is fitted.

The command is intentionally label-free.  Raw and OAR clouds share clonotype embeddings, therefore
one reference codebook must be fitted once and reused for both occupancy-descriptor branches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans


_KEY_COLUMNS = ("chain", "v_gene", "cdr3aa")
_STRATA = (("private", 1, 1), ("low_public", 2, 3), ("mid_public", 4, 10), ("high_public", 11, None))


def stable_hash(key: str) -> int:
    """Return a reproducible unsigned 64-bit hash for a clonotype identity."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def publicness_stratum(n_donors: int) -> str:
    """Map an exact donor count to the predefined publicness strata."""
    if n_donors < 1:
        raise ValueError("n_donors must be positive")
    for label, lower, upper in _STRATA:
        if n_donors >= lower and (upper is None or n_donors <= upper):
            return label
    raise AssertionError(f"unhandled publicness value: {n_donors}")


def _embedding_columns(path: Path, prefix: str) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    columns = [name for name in names if name.startswith(prefix) and name[len(prefix):].isdigit()]
    if not columns:
        raise ValueError(f"{path}: no embedding columns with prefix {prefix!r}")
    return sorted(columns, key=lambda name: int(name[len(prefix):]))


def _identity_key(chain: str, v_gene: str, cdr3aa: str) -> str:
    return "\x1f".join((chain, v_gene, cdr3aa))


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_reference(
    clouds_dir: str | Path,
    out_dir: str | Path,
    *,
    clusters: int = 64,
    hash_modulus: int = 128,
    hash_remainder: int = 0,
    per_stratum: int = 125_000,
    seed: int = 0,
    minibatch_size: int = 10_000,
    embedding_prefix: str = "e",
) -> dict[str, object]:
    """Stream clouds, build a publicness-balanced reference pool, and fit a k-means codebook."""
    if clusters < 2:
        raise ValueError("clusters must be at least two")
    if hash_modulus < 1 or not 0 <= hash_remainder < hash_modulus:
        raise ValueError("hash remainder must lie in [0, hash_modulus)")
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")

    source = Path(clouds_dir)
    files = sorted(source.glob("P*.parquet"))
    if not files:
        raise FileNotFoundError(f"no P*.parquet files under {source}")
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    embedding_columns = _embedding_columns(files[0], embedding_prefix)
    registry: dict[str, int] = {}
    keys: list[str] = []
    embeddings: list[np.ndarray] = []
    donor_counts: list[int] = []

    for index, path in enumerate(files, start=1):
        columns = [*_KEY_COLUMNS, *embedding_columns]
        frame = pd.read_parquet(path, columns=columns)
        missing = sorted(set(_KEY_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing clonotype identity columns {missing}")
        frame = frame.drop_duplicates(list(_KEY_COLUMNS), keep="first")
        for row in frame.itertuples(index=False, name=None):
            chain, v_gene, cdr3aa, *embedding = row
            key = _identity_key(str(chain), str(v_gene), str(cdr3aa))
            if stable_hash(key) % hash_modulus != hash_remainder:
                continue
            position = registry.get(key)
            if position is None:
                registry[key] = len(keys)
                keys.append(key)
                embeddings.append(np.asarray(embedding, dtype=np.float32))
                donor_counts.append(1)
            else:
                donor_counts[position] += 1
        if index % 25 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] retained {len(keys):,} hashed identities", flush=True)

    if len(keys) < clusters:
        raise ValueError(f"only {len(keys)} retained identities for {clusters} clusters; lower --hash-modulus")

    counts = np.asarray(donor_counts, dtype=np.int32)
    stratum = np.asarray([publicness_stratum(int(value)) for value in counts], dtype=object)
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    selected_counts: dict[str, int] = {}
    for label, _, _ in _STRATA:
        available = np.flatnonzero(stratum == label)
        take = min(per_stratum, len(available))
        if take:
            selected_parts.append(rng.choice(available, size=take, replace=False))
        selected_counts[label] = int(take)
    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=int)
    if len(selected) < clusters:
        raise ValueError(f"only {len(selected)} balanced identities for {clusters} clusters")
    rng.shuffle(selected)

    reference = _l2_normalize(np.vstack([embeddings[item] for item in selected]).astype(np.float32, copy=False))
    print(f"fitting MiniBatchKMeans(k={clusters}) on {len(reference):,} balanced identities", flush=True)
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=minibatch_size,
        n_init=3,
        max_iter=200,
    )
    labels = model.fit_predict(reference)
    centroids = _l2_normalize(model.cluster_centers_.astype(np.float32, copy=False))

    selected_mask = np.zeros(len(keys), dtype=bool)
    selected_mask[selected] = True
    audit = pd.DataFrame({
        "chain": [key.split("\x1f", 2)[0] for key in keys],
        "v_gene": [key.split("\x1f", 2)[1] for key in keys],
        "cdr3aa": [key.split("\x1f", 2)[2] for key in keys],
        "n_donors": counts,
        "publicness_stratum": stratum,
        "selected_for_kmeans": selected_mask,
    })
    audit.to_parquet(output / "reference_candidates.parquet", index=False)
    np.savez(output / "prototypes.npz", centroids=centroids, clusters=clusters, seed=seed, temperature=0.1)
    cluster_sizes = np.bincount(labels, minlength=clusters)
    manifest: dict[str, object] = {
        "clouds_dir": str(source),
        "n_clouds": len(files),
        "identity": list(_KEY_COLUMNS),
        "embedding_prefix": embedding_prefix,
        "embedding_dim": len(embedding_columns),
        "hash": {"algorithm": "blake2b-64", "modulus": hash_modulus, "remainder": hash_remainder},
        "clusters": clusters,
        "seed": seed,
        "per_stratum": per_stratum,
        "candidate_identities": len(keys),
        "selected_identities": len(selected),
        "selected_by_stratum": selected_counts,
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_median": float(np.median(cluster_sizes)),
        "cluster_size_max": int(cluster_sizes.max()),
        "inertia": float(model.inertia_),
        "temperature": 0.1,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds-dir", required=True, help="raw cloud directory; one P*.parquet file per patient")
    parser.add_argument("--out", required=True, help="new directory for prototypes and reference audit")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--hash-modulus", type=int, default=128, help="retain identities with hash modulo this value")
    parser.add_argument("--hash-remainder", type=int, default=0)
    parser.add_argument("--per-stratum", type=int, default=125_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minibatch-size", type=int, default=10_000)
    parser.add_argument("--embedding-prefix", default="e")
    args = parser.parse_args()
    build_reference(
        args.clouds_dir,
        args.out,
        clusters=args.clusters,
        hash_modulus=args.hash_modulus,
        hash_remainder=args.hash_remainder,
        per_stratum=args.per_stratum,
        seed=args.seed,
        minibatch_size=args.minibatch_size,
        embedding_prefix=args.embedding_prefix,
    )


if __name__ == "__main__":
    main()
