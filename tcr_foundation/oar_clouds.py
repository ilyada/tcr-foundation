"""Build OAR-corrected beta-chain Foundation-model clouds from patient files.

Each input file is processed independently.  Its ``out`` and ``stop`` events
form the OAR calibrator; corrected productive clonotypes are kept in memory
and embedded immediately.  The only per-patient output is a cloud parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .oar import process_patient, read_patient


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


def _standardise_v_genes(genes: pd.Series) -> pd.Series:
    """Convert platform-specific V calls to tidytcells symbols for CDR1/2 lookup."""
    import tidytcells as tt

    def standardise(gene: str) -> str | None:
        try:
            return tt.tr.standardise(symbol=str(gene).split("*")[0], species="homosapiens")
        except Exception:
            return None

    mapping = {gene: standardise(gene) for gene in genes.astype(str).unique()}
    return genes.astype(str).map(mapping)


def embed_corrected_beta(productive: pd.DataFrame, *, model_path: str, device: str | None = None,
                         batch_size: int = 512) -> tuple[pd.DataFrame, dict[str, int]]:
    """Embed corrected productive TRB clonotypes with the maintained joint-tiny beta path."""
    # comet_ml must precede torch: importing torch first disables its auto-logging hooks.
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        pass
    import torch
    from repertoire.encode_repertoires import embed_clonotypes, resolve_cdr12
    from utils.benchmark_utils import _get_tokenizer, _load_perchain_joint_model

    frame = productive.copy()
    if set(frame["chain"].astype(str)) != {"TRB"}:
        raise ValueError("beta cloud construction requires only TRB productive clonotypes")
    n_corrected = len(frame)
    frame["v_gene"] = _standardise_v_genes(frame["v_gene"])
    frame = frame.dropna(subset=["v_gene"]).reset_index(drop=True)
    cdr12 = frame["v_gene"].map(resolve_cdr12)
    frame["cdr1"] = [value[0] for value in cdr12]
    frame["cdr2"] = [value[1] for value in cdr12]
    frame = frame.dropna(subset=["cdr1", "cdr2"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no productive clonotypes have a resolvable TRB germline sequence")

    target = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _get_tokenizer()
    model, config = _load_perchain_joint_model(model_path, target)
    embeddings = embed_clonotypes(model, tokenizer, config, frame["cdr1"].tolist(), frame["cdr2"].tolist(),
                                  frame["cdr3aa"].tolist(), "beta", target, batch_size)
    weights = np.log1p(frame["count"].to_numpy(dtype=float))
    frame["w_log"] = weights / weights.sum()
    cloud_columns = ["sample", "chain", "cdr3aa", "v_gene", "j_gene", "count_raw", "oar_v", "oar_v_source", "oar_j", "oar_j_source", "oar_coefficient", "count", "w_log"]
    cloud = pd.concat((frame[cloud_columns], pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])])), axis=1)
    return cloud, {"n_productive_corrected": n_corrected, "n_productive_embedded": len(cloud)}


def process_patient_file(path: str | Path, *, model_path: str, min_unique_clonotypes: int = 15,
                         device: str | None = None, batch_size: int = 512) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Process one patient file fully in memory and return cloud, OAR factors, and summary."""
    source = Path(path)
    events = read_patient(source, chain="TRB")
    events = events.loc[events["chain"].astype(str) == "TRB"].copy()
    if events.empty:
        raise ValueError("patient file contains no TRB events")
    factors, productive = process_patient(events, min_unique_clonotypes=min_unique_clonotypes)
    cloud, embedding_summary = embed_corrected_beta(productive, model_path=model_path, device=device, batch_size=batch_size)
    summary = {
        "source": source.name,
        "sample": str(cloud["sample"].iloc[0]),
        "n_event_rows": len(events),
        "n_nonproductive_rows": int(events["frame_type"].astype(str).isin(("out", "stop")).sum()),
        "n_oar_factors": len(factors),
        "oar_min": float(factors["oar"].min()),
        "oar_max": float(factors["oar"].max()),
        "n_oar_calibrated": int((factors["oar_source"] == "patient_nonproductive").sum()),
        "n_oar_neutral_insufficient": int((factors["oar_source"] == "insufficient_nonproductive").sum()),
        "n_oar_neutral_absent": int((factors["oar_source"] == "absent_nonproductive").sum()),
        **embedding_summary,
    }
    return cloud, factors, summary


def build_clouds(input_path: str | Path, *, model_ref: str = "joint-tiny", out_dir: str | Path, pattern: str = "P*.tsv",
                 min_unique_clonotypes: int = 15, device: str | None = None, batch_size: int = 512,
                 limit: int | None = None) -> pd.DataFrame:
    """Build one cloud per patient and one shared OAR control table for the whole run."""
    source = Path(input_path)
    files = [source] if source.is_file() else sorted(source.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no patient files matching {pattern!r} at {source}")
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    clouds_dir = output / "clouds"
    logs_dir = output / "logs"
    clouds_dir.mkdir(parents=True)
    logs_dir.mkdir()
    model_path = resolve_model(model_ref, output_dir=output)
    summaries: list[dict] = []
    factor_frames: list[pd.DataFrame] = []
    for index, patient_path in enumerate(files, 1):
        try:
            cloud, factors, summary = process_patient_file(patient_path, model_path=model_path,
                                                           min_unique_clonotypes=min_unique_clonotypes,
                                                           device=device, batch_size=batch_size)
            cloud.to_parquet(clouds_dir / f"{patient_path.stem}.parquet", index=False)
            factors = factors.assign(source=patient_path.name)
            factor_frames.append(factors)
            summary["status"] = "ok"
            print(f"[{index}/{len(files)}] {patient_path.name}: {summary['n_productive_embedded']} beta clonotypes")
        except Exception as exc:
            summary = {"source": patient_path.name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{index}/{len(files)}] {patient_path.name}: ERROR — {summary['error']}")
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_parquet(output / "oar_summary.parquet", index=False)
    if factor_frames:
        pd.concat(factor_frames, ignore_index=True).to_parquet(output / "oar_factors.parquet", index=False)
    manifest = {"input": str(source), "pattern": pattern, "model_ref": model_ref, "model_path": model_path, "chain": "TRB", "min_unique_clonotypes": min_unique_clonotypes, "batch_size": batch_size, "n_inputs": len(files), "n_success": int((summary_frame["status"] == "ok").sum())}
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
