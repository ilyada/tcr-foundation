"""
encode_repertoires.py — embed a folder of clonotype files into per-donor "clouds".

Generic encoder: take an input folder of VDJtools-style clonotype files, a filename suffix
(e.g. TRB), a chain, and a frozen foundation model, and write one parquet per donor holding the
per-clonotype embeddings (the donor's point cloud) plus log weights. No HLA / metadata is touched
here — the encoder stays generic (folder in -> folder out); labels are joined later by file stem.

Per matched file:
  1. read clonotypes (count, freq, cdr3aa, v, j);
  2. keep PRODUCTIVE cdr3aa only (clean amino acids — drops '_' frameshift / '*' stop);
  3. resolve CDR1/CDR2 from the V gene via tidytcells (the convention the embedder was trained on),
     cached per V gene; drop clonotypes whose V does not resolve;
  4. embed with the frozen per-chain model (model loaded ONCE, streamed over files);
  5. log weights w_log = log(1+count) / sum(log(1+count)); keep raw count + freq too;
  6. write output-dir/<input-stem>.parquet.

Run from scripts/ (so `utils` is importable). Runtime needs torch + transformers + tidytcells (cluster).

Usage (from scripts/):
    python encode_repertoires.py \
        --input-dir ../data/raw/FMBA_main \
        --chain beta \
        --model ../models/foundation/tcr-foundation-joint-tiny \
        --output-dir ../data/processed/clouds/FMBA_main_TRB
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# comet_ml must precede torch: importing torch first disables its auto-logging hooks.
try:
    import comet_ml  # noqa: F401
except ImportError:
    pass
import torch
import tidytcells.tr as tt_tr

# Make `utils` importable when launched from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ root for `from utils`
from utils.benchmark_utils import (  # noqa: E402  (path set above)
    _get_tokenizer,
    _joint_single_text,
    _joint_single_vtoken_text,
    _single_chain_position_ids,
    _load_perchain_joint_model,
)

# A productive CDR3 is a clean amino-acid string (no '_' frameshift, no '*' stop).
AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_RE = re.compile(f"^[{AA}]+$")

# Clonotype files vary in column naming (VDJtools vs MiXCR vs case). Map to canonical names.
COL_ALIASES = {
    "cdr3aa": ["cdr3aa", "cdr3", "cdr3_aa", "cdr3.aa", "aaseqcdr3", "amino.acid", "aminoacid"],
    "v":      ["v", "v_gene", "vgene", "bestvgene", "v.gene", "vsegm"],
    "j":      ["j", "j_gene", "jgene", "bestjgene", "j.gene", "jsegm"],
    "count":  ["uniquemoleculecount", "count", "clonecount", "cloncount", "readcount", "reads", "templates", "duplicate_count"],
    "freq":   ["freq", "frequency", "clonefraction", "proportion"],
}


class EmptyRepertoire(Exception):
    """A clonotype file with no data rows (header-only or empty) -- distinct from a real parse error."""


def _parse_first_gene(s):
    """Extract the first gene symbol from a MiXCR allVHitsWithScore field, e.g.
    'TRBV20-1*00(1521.5),TRBV20-1*01(...)' -> 'TRBV20-1'. None if unparseable."""
    s = str(s)
    if not s or s.lower() == "nan":
        return None
    g = s.split(",")[0].split("*")[0].split("(")[0].strip()
    return g or None


def _normalize_columns(df):
    """Lowercase/strip column names and resolve aliases -> canonical (cdr3aa, v, j, count, freq).
    Returns (df_with_canonical_cols, resolved_dict). cdr3aa and v are required; j/count/freq get
    sensible defaults if absent."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    resolved = {}
    for canon, aliases in COL_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                resolved[canon] = a
                break
    # MiXCR raw format: V/J live inside allVHitsWithScore / allJHitsWithScore, not a plain v/j column.
    if "v" not in resolved and "allvhitswithscore" in df.columns:
        df["v"] = df["allvhitswithscore"].map(_parse_first_gene)
        resolved["v"] = "v"
    if "j" not in resolved and "alljhitswithscore" in df.columns:
        df["j"] = df["alljhitswithscore"].map(_parse_first_gene)
        resolved["j"] = "j"
    if "cdr3aa" not in resolved or "v" not in resolved:
        raise ValueError(f"missing required column(s) cdr3aa/v -- file columns: {list(df.columns)}")
    df = df.rename(columns={resolved[c]: c for c in resolved})
    if "j" not in resolved:
        df["j"] = ""
    if "count" not in resolved:
        df["count"] = 1
    if "freq" not in resolved:
        df["freq"] = float("nan")
    return df


def is_productive(s) -> bool:
    return isinstance(s, str) and bool(_AA_RE.match(s))


_CDR_CACHE: dict = {}


def _get_aa_sequence(symbol: str):
    """tidytcells lookup, compatible with both the new `symbol=` and the old `gene=` kw."""
    try:
        return tt_tr.get_aa_sequence(symbol=symbol, species="homosapiens")
    except TypeError:
        return tt_tr.get_aa_sequence(gene=symbol, species="homosapiens")


def resolve_cdr12(v: str):
    """(CDR1, CDR2) amino-acid strings for a V gene via tidytcells IMGT lookup; cached. (None,None) if it fails.

    tidytcells needs an ALLELE-qualified symbol (e.g. TRBV2*01) -- a bare gene (TRBV2) returns nothing.
    CDR1/CDR2 are germline-encoded and allele-invariant in practice, so we default to the *01 reference."""
    if v in _CDR_CACHE:
        return _CDR_CACHE[v]
    out = (None, None)
    candidates = [v] if "*" in v else [f"{v}*01", v]
    for sym in candidates:
        try:
            r = _get_aa_sequence(sym)
        except Exception:
            continue
        if r and r.get("CDR1-IMGT") and r.get("CDR2-IMGT"):
            out = (r["CDR1-IMGT"], r["CDR2-IMGT"])
            break
    _CDR_CACHE[v] = out
    return out


def standardise_v_gene(v: str) -> str | None:
    """Map an input V call to a tidytcells symbol without changing cloud provenance."""
    try:
        return tt_tr.standardise(symbol=str(v).split("*")[0], species="homosapiens")
    except Exception:
        return None


@torch.no_grad()
def _embed_texts(model, tokenizer, texts, type_id, max_len, device, batch_size):
    """Tokenise single-chain `texts` -> per-chain position_ids + token_type -> mixed-pool z. Shared
    by the CDR1|CDR2|CDR3 and the V-token embedders (only the text builder differs). [N, hidden]."""
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        ttid = torch.full_like(ids, type_id)
        pos = torch.stack([_single_chain_position_ids(m) for m in enc["attention_mask"]]).to(device)
        _, z = model._encode(ids, mask, None, ttid, pos)   # threads per-chain positions + mixed pooling
        chunks.append(z.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def embed_clonotypes(model, tokenizer, jcfg, cdr1, cdr2, cdr3, chain, device, batch_size):
    """Per-chain embedding, legacy CDR1|CDR2|CDR3 grammar. Returns np.ndarray [N, hidden]."""
    max_len = jcfg.get("max_len_beta", 64)          # single-chain sequences fit in the beta budget
    type_id = 0 if chain == "alpha" else 1          # token_type marks the chain
    texts = [_joint_single_text(tokenizer, cdr1[i], cdr2[i], cdr3[i]) for i in range(len(cdr3))]
    return _embed_texts(model, tokenizer, texts, type_id, max_len, device, batch_size)


@torch.no_grad()
def embed_clonotypes_vtoken(model, tokenizer, jcfg, vtokens, cdr3, chain, device, batch_size):
    """Per-chain embedding for the V-token model: "[V:gene] | CDR3" (atomic V token in place of
    CDR1/CDR2). `vtokens` are the mapped V-token strings. Returns np.ndarray [N, hidden]."""
    max_len = jcfg.get("max_len_beta", 64)
    type_id = 0 if chain == "alpha" else 1
    texts = [_joint_single_vtoken_text(tokenizer, vtokens[i], cdr3[i]) for i in range(len(cdr3))]
    return _embed_texts(model, tokenizer, texts, type_id, max_len, device, batch_size)


def _embed_prepared_clonotypes(df, model, tokenizer, jcfg, chain, device, batch_size, *, oar: bool):
    """Embed a canonical productive table after raw or OAR weight preparation."""
    df = df.copy()
    required = {"sample", "chain", "cdr3aa", "v_gene", "j_gene", "count_raw", "count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"prepared clonotypes are missing required columns: {missing}")
    if set(df["chain"].astype(str)) != ({"TRB"} if chain == "beta" else {"TRA"}):
        raise ValueError(f"prepared clonotypes do not match requested {chain} chain")

    # Preserve the reported V call in the cloud, but use the same canonical
    # symbol for germline lookup in raw and OAR branches.
    raw_v_genes = df["v_gene"].astype(str)
    lookup = {gene: standardise_v_gene(gene) for gene in raw_v_genes.unique()}
    df["v_gene_lookup"] = raw_v_genes.map(lookup)
    c12 = df["v_gene_lookup"].map(lambda gene: resolve_cdr12(gene) if gene is not None else (None, None))
    df["cdr1"] = [c[0] for c in c12]
    df["cdr2"] = [c[1] for c in c12]
    df = df[df["cdr1"].notna() & df["cdr2"].notna()].reset_index(drop=True)
    if df.empty:
        return None, 0

    embeddings = embed_clonotypes(
        model, tokenizer, jcfg,
        df["cdr1"].tolist(), df["cdr2"].tolist(), df["cdr3aa"].tolist(),
        chain, device, batch_size,
    )
    log_counts = np.log1p(pd.to_numeric(df["count"], errors="raise").to_numpy(dtype=float))
    if not np.isfinite(log_counts).all() or log_counts.sum() <= 0:
        raise ValueError("cloud weights must be finite and positive after log1p")
    df["w_log"] = log_counts / log_counts.sum()
    df["oar"] = bool(oar)
    df["freq"] = df["count_raw"] / df.groupby(["sample", "chain"], observed=True)["count_raw"].transform("sum")
    for column, value in {
        "oar_v": 1.0, "oar_j": 1.0, "oar_coefficient": 1.0,
        "oar_v_source": "not_applied", "oar_j_source": "not_applied",
    }.items():
        if column not in df:
            df[column] = value
    columns = [
        "sample", "chain", "cdr3aa", "v_gene", "j_gene", "count_raw", "count", "freq", "w_log", "oar",
        "oar_v", "oar_v_source", "oar_j", "oar_j_source", "oar_coefficient",
    ]
    cloud = pd.concat((df[columns], pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])])), axis=1)
    return cloud, len(cloud)


def process_file(path, model, tokenizer, jcfg, chain, device, batch_size):
    """Build a raw-weight cloud from one legacy clonotype file.

    Returns ``(cloud_or_none, summary, factors)``.  ``factors`` is ``None`` in
    raw mode, which keeps the caller's audit interface identical to OAR mode.
    """
    from tcr_foundation.oar import process_patient, read_patient

    chain_label = "TRB" if chain == "beta" else "TRA"
    try:
        events = read_patient(path, chain=chain_label)
    except ValueError as exc:
        # VDJtools-style files lack frame_type and remain supported through the
        # historical reader below.  Adaptive, MiXCR, and canonical parquet
        # inputs take the same raw preparation path as OAR mode.
        if "unsupported TSV" not in str(exc):
            raise
    else:
        events = events.loc[events["chain"].astype(str) == chain_label].copy()
        if events.empty:
            raise EmptyRepertoire(f"no {chain_label} events")
        _, prepared = process_patient(events, oar=False)
        out, n_res = _embed_prepared_clonotypes(prepared, model, tokenizer, jcfg, chain, device, batch_size, oar=False)
        return out, {
            "n_event_rows": len(events),
            "n_nonproductive_rows": int(events["frame_type"].astype(str).isin(("out", "stop")).sum()),
            "n_productive_raw": len(prepared),
            "n_productive_embedded": n_res,
        }, None

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        raise EmptyRepertoire("empty file (no columns)")
    if len(df) == 0:
        raise EmptyRepertoire("header only, no clonotypes")
    df = _normalize_columns(df)
    n_raw = len(df)

    df = df[df["cdr3aa"].map(is_productive)].copy()
    n_prod = len(df)

    prepared = pd.DataFrame({
        "sample": Path(path).stem,
        "chain": "TRB" if chain == "beta" else "TRA",
        "cdr3aa": df["cdr3aa"].values,
        "v_gene": df["v"].values,
        "j_gene": df["j"].values,
        "count_raw": pd.to_numeric(df["count"], errors="raise").values,
        "count": pd.to_numeric(df["count"], errors="raise").values,
    })
    out, n_res = _embed_prepared_clonotypes(prepared, model, tokenizer, jcfg, chain, device, batch_size, oar=False)
    return out, {"n_event_rows": n_raw, "n_productive_raw": n_prod, "n_productive_embedded": n_res}, None


def process_oar_file(path, model, tokenizer, jcfg, chain, device, batch_size, *, min_unique_clonotypes=15):
    """Build an OAR-corrected cloud through the same CDR lookup and embedder as raw mode."""
    from tcr_foundation.oar import process_patient, read_patient

    source = Path(path)
    chain_label = "TRB" if chain == "beta" else "TRA"
    events = read_patient(source, chain=chain_label)
    events = events.loc[events["chain"].astype(str) == chain_label].copy()
    if events.empty:
        raise ValueError(f"{source.name}: no {chain_label} events")
    factors, productive = process_patient(events, min_unique_clonotypes=min_unique_clonotypes)
    out, n_embedded = _embed_prepared_clonotypes(productive, model, tokenizer, jcfg, chain, device, batch_size, oar=True)
    return out, {
        "n_event_rows": len(events),
        "n_nonproductive_rows": int(events["frame_type"].astype(str).isin(("out", "stop")).sum()),
        "n_oar_factors": len(factors),
        "n_oar_calibrated": int((factors["oar_source"] == "patient_nonproductive").sum()),
        "n_oar_neutral_insufficient": int((factors["oar_source"] == "insufficient_nonproductive").sum()),
        "n_oar_neutral_absent": int((factors["oar_source"] == "absent_nonproductive").sum()),
        "n_productive_corrected": len(productive),
        "n_productive_embedded": n_embedded,
    }, factors


def build_clouds(input_path, *, model_path, output_dir, chain="beta", pattern=None, device=None, batch_size=512,
                 limit=None, overwrite=False, oar=False, min_unique_clonotypes=15):
    """Build raw or OAR-weighted clouds through one shared model instance.

    The raw and OAR branches converge before CDR resolution, tokenisation, and
    embedding.  Thus `oar` is the sole intentional difference between the two
    cloud types.  The returned factor table is empty in raw mode.
    """
    source = Path(input_path)
    token = {"beta": "TRB", "alpha": "TRA"}[chain]
    selected_pattern = pattern or f"*{token}*.txt"
    files = [source] if source.is_file() else sorted(source.glob(selected_pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no files matching {selected_pattern!r} at {source}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found {len(files)} files matching {selected_pattern} | chain={chain} | oar={oar} | device={target}")
    if target.type == "cpu":
        print("WARNING: running on CPU — embedding will be slow.")
    tokenizer = _get_tokenizer()
    model, jcfg = _load_perchain_joint_model(model_path, target)

    summaries, factor_frames = [], []
    for index, path in enumerate(files, 1):
        stem = path.stem
        out_path = output / f"{stem}.parquet"
        if out_path.exists() and not overwrite:
            print(f"[{index}/{len(files)}] skip (exists) {stem}")
            summaries.append({"source": path.name, "status": "skipped", "oar": bool(oar)})
            continue
        try:
            if oar:
                cloud, summary, factors = process_oar_file(
                    path, model, tokenizer, jcfg, chain, target, batch_size,
                    min_unique_clonotypes=min_unique_clonotypes,
                )
            else:
                cloud, summary, factors = process_file(path, model, tokenizer, jcfg, chain, target, batch_size)
        except EmptyRepertoire as exc:
            summaries.append({"source": path.name, "status": "empty", "oar": bool(oar), "error": str(exc)})
            print(f"[{index}/{len(files)}] {path.name}: EMPTY ({exc})")
            continue
        except Exception as exc:
            summaries.append({"source": path.name, "status": "error", "oar": bool(oar), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(files)}] {path.name}: ERROR — {type(exc).__name__}: {exc}")
            continue
        if cloud is None:
            summaries.append({"source": path.name, "status": "empty", "oar": bool(oar), **summary})
            print(f"[{index}/{len(files)}] {path.name}: 0 V-resolved clonotypes")
            continue
        cloud.to_parquet(out_path, index=False)
        summaries.append({"source": path.name, "sample": str(cloud["sample"].iloc[0]), "status": "ok", "oar": bool(oar), **summary})
        if factors is not None:
            factor_frames.append(factors.assign(source=path.name))
        print(f"[{index}/{len(files)}] {path.name}: {summary['n_productive_embedded']} {chain} clonotypes")

    summary_frame = pd.DataFrame(summaries)
    factors = pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    return summary_frame, factors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="one clonotype file or a folder of clonotype files")
    ap.add_argument("--chain", choices=["alpha", "beta"], required=True,
                    help="chain to embed: sets token_type and the default filename pattern")
    ap.add_argument("--glob", default=None, help="override the filename glob")
    ap.add_argument("--model", required=True, help="frozen foundation checkpoint directory")
    ap.add_argument("--output-dir", required=True, help="where per-donor cloud parquets are written")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N files (debug)")
    ap.add_argument("--overwrite", action="store_true", help="re-encode even if the output parquet exists")
    ap.add_argument("--oar", action="store_true", help="estimate patient-specific OARs from out/stop events before weighting productive clonotypes")
    ap.add_argument("--min-unique-clonotypes", type=int, default=15, help="minimum non-productive clonotypes per V/J segment in OAR mode")
    args = ap.parse_args()

    summary, factors = build_clouds(
        args.input_dir, model_path=args.model, output_dir=args.output_dir, chain=args.chain, pattern=args.glob,
        device=args.device, batch_size=args.batch_size, limit=args.limit, overwrite=args.overwrite,
        oar=args.oar, min_unique_clonotypes=args.min_unique_clonotypes,
    )
    output = Path(args.output_dir)
    summary.to_parquet(output / ("oar_summary.parquet" if args.oar else "cloud_summary.parquet"), index=False)
    if args.oar and not factors.empty:
        factors.to_parquet(output / "oar_factors.parquet", index=False)
    manifest = {
        "input": str(args.input_dir), "pattern": args.glob, "model_path": str(args.model), "chain": args.chain,
        "oar": bool(args.oar), "min_unique_clonotypes": args.min_unique_clonotypes if args.oar else None,
        "batch_size": args.batch_size, "n_inputs": len(summary), "n_success": int((summary["status"] == "ok").sum()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"completed {manifest['n_success']}/{len(summary)} files")


if __name__ == "__main__":
    main()
