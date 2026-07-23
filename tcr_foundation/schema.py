"""
schema.py -- one canonical clonotype table for the whole library, + ingestion + germline resolution.

Every encoder / featurizer / distance consumes a DataFrame with these CANONICAL columns:
    v_gene   (str, required)   e.g. "TRBV20-1"
    cdr3     (str, required)   junction amino-acid string, e.g. "CASS...F"
    j_gene   (str, optional)
    count    (float, optional; defaults to 1)   clonal abundance
    cdr1, cdr2 (str, optional) germline loops -- filled by resolve_germline() when an encoder needs them

`ingest()` maps a messy input frame (VDJtools / MiXCR / AIRR / our cloud parquets) onto this schema by
auto-detecting columns from a small alias table (or you pass explicit names). `resolve_germline()` fills
cdr1/cdr2 from the V gene via tidytcells -- imported lazily so pure-sequence work never pulls in that dep.

No torch/transformers here: this is the light core.
"""
from __future__ import annotations

import pandas as pd

V_GENE, CDR3, J_GENE, COUNT, CDR1, CDR2 = "v_gene", "cdr3", "j_gene", "count", "cdr1", "cdr2"
REQUIRED = (V_GENE, CDR3)
CANONICAL = (V_GENE, CDR3, J_GENE, COUNT, CDR1, CDR2)

# Common column spellings across formats -> canonical name. First match wins (case-insensitive).
_ALIASES = {
    V_GENE: ["v_gene", "v", "bestvgene", "v_call", "vgene", "v_b_gene", "trbv", "v_segm", "allvhitswithscore"],
    CDR3:   ["cdr3", "cdr3aa", "cdr3_aa", "cdr3b", "aaseqcdr3", "junction_aa", "cdr3_b_aa", "cdr3amino"],
    J_GENE: ["j_gene", "j", "bestjgene", "j_call", "jgene", "j_b_gene", "trbj", "j_segm"],
    COUNT:  ["count", "cloneCount", "clonecount", "reads", "duplicate_count", "freq", "frequency"],
    CDR1:   ["cdr1", "cdr1aa", "aaseqcdr1"],
    CDR2:   ["cdr2", "cdr2aa", "aaseqcdr2"],
}


def _detect(columns, wanted):
    lower = {str(c).lower(): c for c in columns}
    for alias in _ALIASES[wanted]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def ingest(df: pd.DataFrame, *, v_gene=None, cdr3=None, j_gene=None, count=None,
           cdr1=None, cdr2=None) -> pd.DataFrame:
    """Map an arbitrary clonotype frame onto the canonical schema. Explicit column names override
    auto-detection. Missing `count` -> 1.0. Returns a new frame with only the canonical columns present."""
    picks = {V_GENE: v_gene, CDR3: cdr3, J_GENE: j_gene, COUNT: count, CDR1: cdr1, CDR2: cdr2}
    out = {}
    for canon, explicit in picks.items():
        src = explicit if explicit is not None else _detect(df.columns, canon)
        if src is not None and src in df.columns:
            out[canon] = df[src].values
    for req in REQUIRED:
        if req not in out:
            raise ValueError(f"ingest: could not find a '{req}' column (looked for {_ALIASES[req]}); "
                             f"pass it explicitly. Available: {list(df.columns)}")
    res = pd.DataFrame(out)
    res[V_GENE] = res[V_GENE].astype(str)
    res[CDR3] = res[CDR3].astype(str)
    if COUNT not in res:
        res[COUNT] = 1.0
    res[COUNT] = pd.to_numeric(res[COUNT], errors="coerce").fillna(1.0).astype(float)
    return res


def read(path: str, **ingest_kwargs) -> pd.DataFrame:
    """Read a parquet (.parquet) or tab-separated (.tsv/.txt) clonotype file and ingest it to canonical."""
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    return ingest(df, **ingest_kwargs)


def resolve_germline(df: pd.DataFrame) -> pd.DataFrame:
    """Fill cdr1/cdr2 from the V gene via tidytcells (IMGT lookup, cached). Rows whose V gene does not
    resolve get None in cdr1/cdr2. Reuses the existing scripts implementation; imported lazily so the
    pure-sequence path never imports torch/tidytcells."""
    from repertoire.encode_repertoires import resolve_cdr12  # lazy: heavy import chain
    c12 = df[V_GENE].astype(str).map(resolve_cdr12)
    res = df.copy()
    res[CDR1] = [c[0] for c in c12]
    res[CDR2] = [c[1] for c in c12]
    return res
