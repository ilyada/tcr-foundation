"""
rep_data.py -- cloud + cohort-metadata loading for repertoire analysis.

A "cloud" is one donor's per-clonotype parquet with the frozen embedding columns e0..e127 plus w_log (log-count
weight), cdr3aa, v_gene, j_gene. Loading is memory-safe by design: read ONE cloud at a time and discard it
(holding ~1900 clouds in RAM caused MemoryError before). For descriptor building, use iter_clouds and accumulate
only the small descriptor vectors, never the raw point clouds.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EDIM = 128
EMB_COLS = [f"e{i}" for i in range(EDIM)]


def cloud_files(cloud_dir):
    """{stem: path} for every *.parquet in a cloud directory (stem = filename without .parquet)."""
    return {os.path.basename(f)[:-len(".parquet")]: f for f in glob.glob(os.path.join(cloud_dir, "*.parquet"))}


def embedding_cols(path):
    """Detect the embedding columns e0..e{d-1} present in a cloud (any d -> supports 128-d ours, 64-d SCEPTR, ...)."""
    e = [c for c in pq.read_schema(path).names if re.fullmatch(r"e\d+", c)]
    return sorted(e, key=lambda c: int(c[1:])) or EMB_COLS


def load_cloud(path, cap=None, seed=0, cols=("w_log", "v_gene")):
    """Load one cloud -> (Z [N,d] float32 L2-normalized, w [N] float32 sum-1, extra dict). Embedding dim d is
    auto-detected from the file (model-agnostic).

    `cols` names the non-embedding columns to also return (in `extra`). `cap` optionally subsamples very deep
    clouds for memory (embedding columns only read once)."""
    emb = embedding_cols(path)
    df = pd.read_parquet(path, columns=emb + list(cols))
    if cap and len(df) > cap:
        df = df.sample(cap, random_state=seed)
    Z = df[emb].to_numpy(np.float32)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    w = df["w_log"].to_numpy(np.float32) if "w_log" in cols else np.full(len(df), 1.0 / len(df), np.float32)
    w = w / (w.sum() + 1e-8)
    extra = {c: df[c].to_numpy() for c in cols if c != "w_log"}
    return Z, w, extra


def load_clouds_ram(cloud_dir, stems, cap=None, fp16=True):
    """Load a chosen set of clouds into RAM as {stem: (Z, w)} (Z fp16 to halve memory). Use only for a bounded set
    (e.g. one cohort's train split), never the whole corpus."""
    files = cloud_files(cloud_dir)
    out = {}
    for s in stems:
        if s in files:
            Z, w, _ = load_cloud(files[s], cap=cap, cols=("w_log",))
            out[s] = (Z.astype(np.float16) if fp16 else Z, w)
    return out


# ----------------------------------------------------------------------------- cohort metadata
def load_vaccine_meta(path):
    """FMBA_vaccine metadata -> DataFrame[stem, subject, timepoint, vaccine].
    subject links before/after (= sample.id without the last 2 chars); tab-separated with a leading index col."""
    m = pd.read_csv(path, sep="\t")
    m.columns = [c.strip().lstrip("#") for c in m.columns]
    m["stem"] = m["file.name"].map(lambda p: os.path.basename(str(p)).replace(".txt", ""))
    m["subject"] = m["sample.id"].astype(str).str[:-2]
    return m[["stem", "subject", "timepoint", "vaccine"]]


VACCINE_TP = {"before_vaccination": 0, "20d_after_vaccination": 1}


def paired_subjects(meta, cloud_dir):
    """Subjects with BOTH timepoints present as clouds -> {subject: {0: stem_before, 1: stem_after, 'vaccine': v}}."""
    files = cloud_files(cloud_dir)
    m = meta[meta["stem"].isin(files) & meta["timepoint"].isin(VACCINE_TP)]
    out = {}
    for _, r in m.iterrows():
        out.setdefault(r["subject"], {})[VACCINE_TP[r["timepoint"]]] = r["stem"]
        out[r["subject"]]["vaccine"] = r["vaccine"]
    return {s: d for s, d in out.items() if 0 in d and 1 in d}
