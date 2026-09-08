"""
schema.py -- one canonical clonotype table for the whole library, + ingestion + germline resolution.

Every encoder / featurizer / distance consumes a DataFrame with these CANONICAL columns:
    v_gene   (str, required)   e.g. "TRBV20-1"
    cdr3     (str, required)   junction amino-acid string, e.g. "CASS...F"
    j_gene   (str, optional)
    count    (float, optional; defaults to 1)   clonal abundance
    cdr1, cdr2 (str, optional) germline loops -- filled by resolve_germline() when an encoder needs them
    cdr3_nt  (str, optional)   junction NUCLEOTIDE string -- the join key to the recombination-event table
    productive (bool, optional) in-frame and stop-free

`ingest()` maps a messy input frame (VDJtools / MiXCR / AIRR / our cloud parquets) onto this schema by
auto-detecting columns from a small alias table (or you pass explicit names). `resolve_germline()` fills
cdr1/cdr2 from the V gene via tidytcells -- imported lazily so pure-sequence work never pulls in that dep.

cdr3_nt / productive exist because everything probabilistic lives at the level of the recombination EVENT
(see events.py), and an amino-acid clonotype is a many-to-one image of it: one CDR3 string is produced by
many distinct nucleotide rearrangements. Carrying the nucleotide junction on the clonotype row is what makes
that join exact instead of approximate. `productive` is carried rather than filtered on: the non-productive
rearrangements ARE the pre-selection null, and dropping them at ingest is precisely the defect that left
every assembled cohort in this project without a single out-of-frame row.

No torch/transformers here: this is the light core.
"""
from __future__ import annotations

import pandas as pd

V_GENE, CDR3, J_GENE, COUNT, CDR1, CDR2 = "v_gene", "cdr3", "j_gene", "count", "cdr1", "cdr2"
CDR3_NT, PRODUCTIVE = "cdr3_nt", "productive"
REQUIRED = (V_GENE, CDR3)
CANONICAL = (V_GENE, CDR3, J_GENE, COUNT, CDR1, CDR2, CDR3_NT, PRODUCTIVE)

# Common column spellings across formats -> canonical name. First match wins (case-insensitive).
_ALIASES = {
    V_GENE: ["v_gene", "v", "bestvgene", "v_call", "vgene", "v_b_gene", "trbv", "v_segm", "allvhitswithscore"],
    CDR3:   ["cdr3", "cdr3aa", "cdr3_aa", "cdr3b", "aaseqcdr3", "junction_aa", "cdr3_b_aa", "cdr3amino"],
    J_GENE: ["j_gene", "j", "bestjgene", "j_call", "jgene", "j_b_gene", "trbj", "j_segm"],
    COUNT:  ["count", "cloneCount", "clonecount", "reads", "duplicate_count", "freq", "frequency"],
    CDR1:   ["cdr1", "cdr1aa", "aaseqcdr1"],
    CDR2:   ["cdr2", "cdr2aa", "aaseqcdr2"],
    # cdr3nt means DIFFERENT things per platform in this project's older code -- the whole rearrangement on
    # Adaptive, the CDR3 only on MiXCR -- so the canonical name is explicit and the alias list is narrow.
    CDR3_NT: ["cdr3_nt", "cdr3nt", "nseqcdr3", "junction", "cdr3_rearrangement", "cdr3_b_nucseq"],
    PRODUCTIVE: ["productive", "is_productive"],
}

# Strings that count as True for `productive`. Anything else non-null reads as False.
_TRUEISH = frozenset({"true", "t", "yes", "y", "1", "in", "productive"})


def _detect(columns, wanted):
    lower = {str(c).lower(): c for c in columns}
    for alias in _ALIASES[wanted]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a productivity column to bool from bool / 0-1 / text, without inventing a default."""
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin(_TRUEISH)


def ingest(df: pd.DataFrame, *, v_gene=None, cdr3=None, j_gene=None, count=None,
           cdr1=None, cdr2=None, cdr3_nt=None, productive=None) -> pd.DataFrame:
    """Map an arbitrary clonotype frame onto the canonical schema. Explicit column names override
    auto-detection. Missing `count` -> 1.0. Returns a new frame with only the canonical columns present."""
    picks = {V_GENE: v_gene, CDR3: cdr3, J_GENE: j_gene, COUNT: count, CDR1: cdr1, CDR2: cdr2,
             CDR3_NT: cdr3_nt, PRODUCTIVE: productive}
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
    if CDR3_NT in res:
        res[CDR3_NT] = res[CDR3_NT].astype(str).str.upper()
    if PRODUCTIVE in res:
        res[PRODUCTIVE] = _as_bool(res[PRODUCTIVE])
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
