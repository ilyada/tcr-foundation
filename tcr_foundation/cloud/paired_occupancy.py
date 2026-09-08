"""Compatibility entry point for paired occupancy descriptors.

New code should use :mod:`tcr_foundation.cloud_descriptors`, which builds moment and
prototype-occupancy descriptors through one validated raw/OAR pairing path.  This module retains
the established import and command-line interface for existing workflows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .descriptors import (
    align_oar_to_raw as _align_oar_to_raw,
    build_paired_descriptors,
    effective_n as _effective_n,
    embedding_columns as _embedding_columns,
    load_codebooks as _load_codebooks,
    normalise_rows as _normalize_rows,
    normalised_weights as _normalised_weights,
    soft_assignments as _soft_assignments,
)


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
    """Write aligned raw/OAR occupancy tables, preserving historical output names."""
    manifest = build_paired_descriptors(
        raw_clouds,
        oar_clouds,
        out_dir,
        kinds=("occupancy",),
        codebook_dirs=codebook_dirs,
        temperature=temperature,
        embedding_prefix=embedding_prefix,
        chunk_size=chunk_size,
        limit=limit,
        write_pair_qc=True,
    )
    output = Path(out_dir)
    pair_qc = pd.read_parquet(output / "pair_qc.parquet")
    for clusters in manifest["codebooks"]:
        pair_qc.query("descriptor == @descriptor", local_dict={"descriptor": f"occupancy_k{clusters}"}).drop(columns="descriptor").to_parquet(output / f"occupancy_k{clusters}_pair_qc.parquet", index=False)
    (output / "pair_qc.parquet").unlink()
    manifest["clusters"] = [int(clusters) for clusters in manifest["codebooks"]]
    manifest["tables"] = [name for name in manifest["tables"] if name != "pair_qc.parquet"]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-clouds", required=True)
    parser.add_argument("--oar-clouds", required=True)
    parser.add_argument("--codebooks", nargs="+", required=True, help="directories containing prototypes.npz and manifest.json")
    parser.add_argument("--out", required=True, help="new directory for paired occupancy tables")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--embedding-prefix", default="e")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N paired clouds; intended for smoke tests")
    args = parser.parse_args()
    build_paired_occupancies(args.raw_clouds, args.oar_clouds, args.codebooks, args.out, temperature=args.temperature, embedding_prefix=args.embedding_prefix, chunk_size=args.chunk_size, limit=args.limit)


if __name__ == "__main__":
    main()
