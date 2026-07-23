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
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import tidytcells.tr as tt_tr

# Make `utils` importable when launched from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
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


def process_file(path, model, tokenizer, jcfg, chain, device, batch_size):
    """Read one clonotype file -> productive filter -> tidytcells CDR1/2 -> embed -> DataFrame.
    Returns (df_or_None, (n_raw, n_productive, n_resolved))."""
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

    c12 = df["v"].astype(str).map(resolve_cdr12)
    df["cdr1"] = [c[0] for c in c12]
    df["cdr2"] = [c[1] for c in c12]
    df = df[df["cdr1"].notna() & df["cdr2"].notna()].reset_index(drop=True)
    n_res = len(df)
    if n_res == 0:
        return None, (n_raw, n_prod, n_res)

    Z = embed_clonotypes(
        model, tokenizer, jcfg,
        df["cdr1"].tolist(), df["cdr2"].tolist(), df["cdr3aa"].tolist(),
        chain, device, batch_size,
    )

    # log weighting: w_log = log(1+count) / sum(log(1+count))
    w = np.log1p(df["count"].astype(float).values)
    w_log = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))

    H = Z.shape[1]
    out = pd.DataFrame({
        "cdr3aa": df["cdr3aa"].values,
        "v_gene": df["v"].values,
        "j_gene": df["j"].values,
        "count":  df["count"].values,
        "freq":   df["freq"].values,
        "w_log":  w_log,
    })
    emb = pd.DataFrame(Z, columns=[f"e{i}" for i in range(H)])
    return pd.concat([out, emb], axis=1), (n_raw, n_prod, n_res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="folder of clonotype .txt files")
    ap.add_argument("--chain", choices=["alpha", "beta"], required=True,
                    help="chain to embed: sets token_type AND selects files (beta -> *TRB*.txt, alpha -> *TRA*.txt)")
    ap.add_argument("--glob", default=None,
                    help="override the filename glob (default derived from --chain: *TRB*.txt / *TRA*.txt)")
    ap.add_argument("--model", required=True, help="frozen foundation dir (backbone + poolers/joint.pt + joint_config.json)")
    ap.add_argument("--output-dir", required=True, help="where per-donor parquets are written")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N files (debug)")
    ap.add_argument("--overwrite", action="store_true", help="re-encode even if the output parquet exists")
    args = ap.parse_args()

    chain = args.chain
    token = {"beta": "TRB", "alpha": "TRA"}[chain]
    pattern = args.glob or f"*{token}*.txt"

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, pattern)))
    if args.limit:
        files = files[:args.limit]
    print(f"Found {len(files)} files matching {pattern}  |  chain={chain}  |  device={device}")
    if device.type == "cpu":
        print("WARNING: running on CPU — embedding will be slow.")

    tokenizer = _get_tokenizer()
    model, jcfg = _load_perchain_joint_model(args.model, device)

    tot_raw = tot_prod = tot_res = 0
    empty, failed = [], []
    for k, path in enumerate(files, 1):
        stem = os.path.basename(path)[:-4]                      # strip .txt; stem == metadata file.name minus .txt
        out_path = os.path.join(args.output_dir, stem + ".parquet")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{k}/{len(files)}] skip (exists) {stem}")
            continue
        try:
            out, (n_raw, n_prod, n_res) = process_file(path, model, tokenizer, jcfg, chain, device, args.batch_size)
        except EmptyRepertoire as ex:                           # no data -> not an error, just empty
            print(f"[{k}/{len(files)}] {stem}: EMPTY ({ex}) — skipped")
            empty.append(stem)
            continue
        except Exception as ex:                                 # a genuinely malformed file must not kill the run
            print(f"[{k}/{len(files)}] {stem}: PARSE ERROR — {ex}")
            failed.append(stem)
            continue
        tot_raw += n_raw
        tot_prod += n_prod
        tot_res += n_res
        if out is None:
            print(f"[{k}/{len(files)}] {stem}: 0 usable clonotypes — skipped")
            continue
        out.to_parquet(out_path, index=False)
        print(f"[{k}/{len(files)}] {stem}: {n_raw} -> productive {n_prod} -> V-resolved {n_res}")

    n_unres = sum(1 for v in _CDR_CACHE.values() if v[0] is None)
    print(f"\nDONE -> {args.output_dir}")
    print(f"  clonotypes: {tot_raw} total | productive {tot_prod} ({100*tot_prod/max(tot_raw,1):.1f}%) "
          f"| V-resolved {tot_res} ({100*tot_res/max(tot_raw,1):.1f}%)")
    print(f"  unique V genes seen: {len(_CDR_CACHE)} (unresolved by tidytcells: {n_unres})")
    if n_unres:
        print("  unresolved V genes:", sorted(v for v in _CDR_CACHE if _CDR_CACHE[v][0] is None))
    if empty:
        print(f"  EMPTY (no clonotypes): {len(empty)} files")
    if failed:
        print(f"  PARSE ERRORS: {len(failed)} files: {failed[:20]}{' ...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
