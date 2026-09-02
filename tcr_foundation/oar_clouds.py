"""Build OAR-corrected beta-chain Foundation-model clouds from patient files.

Each input file supplies its own ``out`` and ``stop`` OAR calibrator.  The
shared repertoire builder loads the encoder once, then emits one cloud parquet
per patient and shared OAR audit tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ._vendor.repertoire.encode_repertoires import build_clouds as _build_clouds


JOINT_TINY_HF_REPOSITORY = "argentel/tcr-foundation-joint-tiny"


def resolve_model(model_ref: str, *, output_dir: Path) -> str:
    """Resolve a local checkpoint or download an HF model into this run directory."""
    local = Path(model_ref)
    if local.is_dir():
        return str(local.resolve())
    repository = JOINT_TINY_HF_REPOSITORY if model_ref == "joint-tiny" else model_ref
    if "/" not in repository:
        raise ValueError("model must be a checkpoint directory, 'joint-tiny', or an HF repository identifier")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Hugging Face model download requires huggingface_hub") from exc
    return snapshot_download(repo_id=repository, local_dir=output_dir / "model")


def build_clouds(input_path: str | Path, *, model_ref: str = "joint-tiny", out_dir: str | Path, pattern: str = "P*.tsv",
                 min_unique_clonotypes: int = 15, device: str | None = None, batch_size: int = 512,
                 limit: int | None = None) -> pd.DataFrame:
    """Build OAR beta clouds through the shared raw/OAR repertoire builder."""
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    model_path = resolve_model(model_ref, output_dir=output)
    summary_frame, factors = _build_clouds(
        input_path, model_path=model_path, output_dir=output / "clouds", chain="beta", pattern=pattern,
        device=device, batch_size=batch_size, limit=limit, oar=True,
        min_unique_clonotypes=min_unique_clonotypes,
    )
    summary_frame.to_parquet(output / "oar_summary.parquet", index=False)
    if not factors.empty:
        factors.to_parquet(output / "oar_factors.parquet", index=False)
    manifest = {"input": str(input_path), "pattern": pattern, "model_ref": model_ref, "model_path": model_path, "chain": "TRB", "oar": True, "min_unique_clonotypes": min_unique_clonotypes, "batch_size": batch_size, "n_inputs": len(summary_frame), "n_success": int((summary_frame["status"] == "ok").sum())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary_frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build OAR-corrected beta-only Foundation-model clouds from patient files.")
    parser.add_argument("--input", required=True, help="one patient file or a directory of patient files")
    parser.add_argument("--model", default="joint-tiny", help="checkpoint directory, HF repository, or joint-tiny (default)")
    parser.add_argument("--out", required=True, help="new output directory; only clouds and shared controls are written")
    parser.add_argument("--glob", default="P*.tsv", help="directory input glob (default: P*.tsv)")
    parser.add_argument("--min-unique-clonotypes", type=int, default=15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    summary = build_clouds(args.input, model_ref=args.model, out_dir=args.out, pattern=args.glob,
                           min_unique_clonotypes=args.min_unique_clonotypes, device=args.device,
                           batch_size=args.batch_size, limit=args.limit)
    print(f"completed {int((summary['status'] == 'ok').sum())}/{len(summary)} patient files")


if __name__ == "__main__":
    main()
