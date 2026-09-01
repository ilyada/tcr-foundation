"""
events.py -- the canonical recombination-EVENT table, per-platform readers, and the corpus build.

A level BELOW the clonotype. One row is one rearrangement, described the way the recombination machinery
actually produced it: which V, D and J segments were joined, how many nucleotides were chewed off each end,
and how many were inserted at each junction. The amino-acid CDR3 is a many-to-one image of this; the event
is where the probabilities live.

WHY THIS EXISTS, AND WHY IT KEEPS THE OUT-OF-FRAME ROWS
------------------------------------------------------
The pre-selection null of a donor is their own NON-PRODUCTIVE rearrangements: frameshifted or stop-carrying,
so no receptor was ever built from them and the thymus never selected them. They come from the same tube,
the same library and the same depth as the productive ones, which is why technical factors cancel between
the two. Every extraction path in this project so far filtered them away at read time
(`scripts/utils/data_build/build_cohort_parquet.py:84,114` keeps `frame_type == "In"`), so no assembled
cohort contains a single out-of-frame row. This module carries `frame_type` as a column instead. Filtering
is a query, never a build step.

WHY THE EVENT COORDINATES AND NOT THE NUCLEOTIDE STRING
-------------------------------------------------------
Productivity is a deterministic function of the sequence: the junction length modulo three, plus the
presence of an in-frame stop. Measured on Emerson `Keck0001_MC1.tsv`, the junction length is divisible by
three in 105,015 of 105,015 productive rows and in none of the 23,725 out-of-frame ones. So any model that
can COUNT -- and an autoregressive model over nucleotides counts by construction, its position index is the
counter -- can reconstruct the class, and a likelihood ratio conditioned on the class then measures the
frame rule rather than selection. In the event coordinates the junction length is a SUM of the coordinates,
so a model whose coordinate groups are conditionally independent given the segments cannot represent
"the sum is divisible by three" at all. That is the same factorisation IGoR uses.

THE SENTINEL, MEASURED NOT ASSUMED
----------------------------------
Adaptive writes -1 for an insertion index when there is no insertion. On `Keck0001_MC1.tsv` (131,030 rows):
`n1_index < 0` in 16,318 rows and `ins_n1 == 0` in all 16,318; where `n1_index >= 0` the identity
`ins_n1 == d_index - n1_index` holds in all 114,712. Same for n2: 22,114 sentinel rows all with
`ins_n2 == 0`, and 108,916 rows where `ins_n2 == j_index - n2_index` holds exactly. No exceptions in either.
Read naively -- without the sentinel -- those identities appear to hold in only 83-88% of rows, which is
what a silent off-by-something looks like. `validate_events()` re-checks this on every corpus built.

The same -1 convention is reused for every column that can be absent, so "not reported" never masquerades
as a zero: a D segment is unassigned in 21.5% of Emerson rows (28,175 of 131,030) and in 22.1% of MiXCR
rows measured independently on FMBA, and for those rows the n1/n2 split does not exist -- only the total
insertion between V and J does.

GENE NAMES ARE STORED RAW
-------------------------
Adaptive spells genes `TCRBV05-01`, MiXCR `TRBV5-1*00`. Canonicalisation is a transformation that can be
redone at any time, and the frame-leak diagnostics condition on the segment as a category, so nothing here
needs it. `canonicalize_genes()` is offered separately and imports tidytcells lazily.

CLI
    python -m tcr_foundation.events --input DIR --out DIR --platform adaptive [--limit N] [--report-only]
"""
from __future__ import annotations

import glob
import os
import json
import re

import numpy as np
import pandas as pd

# -- the canonical event schema ------------------------------------------------------------------------

FRAME_IN, FRAME_OUT, FRAME_STOP = "in", "out", "stop"

EVENT_COLUMNS = (
    "donor", "sample", "library", "timepoint", "chain",
    "v_gene", "d_gene", "j_gene",
    "v_resolved", "d_resolved", "j_resolved",
    "frame_type", "cdr3_length",
    "del_v", "del_d5", "del_d3", "del_j",
    "ins_n1", "ins_n2", "ins_vj",
    "p_v", "p_d5", "p_d3", "p_j",
    "v_index", "n1_index", "n2_index", "d_index", "j_index",
    "d_present",
    "rearrangement", "templates",
)

#: Every integer column uses -1 for "not reported / not applicable", following Adaptive's own convention.
MISSING = -1

_INT_COLS = ("cdr3_length", "del_v", "del_d5", "del_d3", "del_j", "ins_n1", "ins_n2", "ins_vj",
             "p_v", "p_d5", "p_d3", "p_j", "v_index", "n1_index", "n2_index", "d_index", "j_index")


#: What a platform writes when it could not call a gene.
UNCALLED = frozenset({"", "nan", "none", "unresolved", "unknown"})


def best_gene(df: pd.DataFrame, side: str) -> pd.Series:
    """The gene call to model, preferring the resolved column over the raw one.

    Adaptive fills `v_gene` with the literal string "unresolved" whenever the read cannot separate members
    of a family, but still reports the family in `v_resolved` -- measured on Emerson, that is 20.32 percent
    of rearrangements. Modelling `v_gene` therefore throws a fifth of the V calls into an unresolved class
    that carries no information, when a family-level call was available all along.

    A family-level call (TCRBV20 with no member) is NOT a defect: it means several templates are compatible,
    which is exactly the ambiguity the scenario marginalisation is there to sum over.

    Both columns stay in the corpus. Which one to model is a modelling decision, so it lives here and not in
    the reader -- the stored table remains a faithful copy of what the platform reported.
    """
    raw = df[f"{side}_gene"].astype(str).str.strip()
    res = df[f"{side}_resolved"].astype(str).str.strip()
    use_res = ~res.str.lower().isin(UNCALLED)
    return res.where(use_res, raw)


def _to_int(series, fill=MISSING) -> np.ndarray:
    """Numeric coercion that turns blanks into the sentinel rather than into NaN or 0."""
    return pd.to_numeric(series, errors="coerce").fillna(fill).astype("int32").to_numpy()


def _finalize(frame: dict, donor: str, sample: str, chain: str, library: str | None = None,
              timepoint: str = "") -> pd.DataFrame:
    """Assemble, order and downcast one sample's rows to the canonical schema."""
    n = len(frame["rearrangement"])
    frame.setdefault("donor", np.repeat(donor, n))
    frame.setdefault("sample", np.repeat(sample, n))
    frame.setdefault("library", np.repeat(library or sample, n))
    frame.setdefault("timepoint", np.repeat(timepoint, n))
    frame.setdefault("chain", np.repeat(chain, n))
    for col in EVENT_COLUMNS:
        if col not in frame:
            frame[col] = np.full(n, MISSING if col in _INT_COLS else "")
    df = pd.DataFrame({c: frame[c] for c in EVENT_COLUMNS})
    for c in _INT_COLS:
        df[c] = df[c].astype("int32")
    df["templates"] = pd.to_numeric(df["templates"], errors="coerce").fillna(1).astype("int32")
    df["d_present"] = df["d_present"].astype(bool)
    for c in ("donor", "sample", "library", "timepoint", "chain", "v_gene", "d_gene", "j_gene",
              "v_resolved", "d_resolved", "j_resolved", "frame_type"):
        df[c] = df[c].astype("category")
    return df


def _load_identity_map(path: str) -> pd.DataFrame:
    """Read a sample-to-donor map with required sample and donor columns.

    Library and timepoint are optional for cross-sectional corpora. If absent, the library defaults to the
    sample and timepoint is empty. A longitudinal or OAR analysis must supply both fields explicitly; this
    function preserves them in every event row instead of attempting to infer biology from a file name.
    """
    identity = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, sep="\t")
    return _normalise_identity_map(identity, source=path)


def _normalise_identity_map(identity: pd.DataFrame, source: str = "<in-memory>") -> pd.DataFrame:
    """Validate and index a sample-to-identity table supplied by a file or a test fixture."""
    required = {"sample", "donor"}
    missing = required - set(identity.columns)
    if missing:
        raise ValueError(f"identity map {source!r} is missing required columns {sorted(missing)}")
    keep = [c for c in ("sample", "donor", "library", "timepoint") if c in identity.columns]
    identity = identity[keep].copy()
    for c in keep:
        identity[c] = identity[c].fillna("").astype(str).str.strip()
    if (identity["sample"] == "").any() or (identity["donor"] == "").any():
        raise ValueError("identity map contains an empty sample or donor")
    if identity["sample"].duplicated().any():
        bad = sorted(identity.loc[identity["sample"].duplicated(keep=False), "sample"].unique())[:5]
        raise ValueError(f"identity map has non-unique samples: {bad}")
    if "library" not in identity:
        identity["library"] = identity["sample"]
    if "timepoint" not in identity:
        identity["timepoint"] = ""
    if (identity["library"] == "").any():
        raise ValueError("identity map contains an empty library")
    return identity.set_index("sample")[["donor", "library", "timepoint"]]


def _apply_identity(df: pd.DataFrame, identity: pd.DataFrame | None) -> pd.DataFrame:
    """Attach recorded biological and technical identities to a one-sample event table."""
    if identity is None:
        return df
    samples = pd.Index(df["sample"].astype(str).unique())
    missing = samples.difference(identity.index)
    if len(missing):
        raise ValueError(f"identity map has no row for sample(s) {missing.tolist()[:5]}")
    out = df.copy()
    mapped = identity.loc[out["sample"].astype(str)]
    out["donor"] = mapped["donor"].to_numpy()
    out["library"] = mapped["library"].to_numpy()
    out["timepoint"] = mapped["timepoint"].to_numpy()
    for c in ("donor", "library", "timepoint"):
        out[c] = out[c].astype("category")
    return out


# -- Adaptive (immuneACCESS v2 export; Emerson) --------------------------------------------------------

#: 25 of the 118 columns. Reading only these is what makes a 370 GB corpus affordable.
ADAPTIVE_COLS = [
    "sample_name", "rearrangement", "amino_acid", "frame_type", "rearrangement_type", "templates",
    "cdr3_length", "v_gene", "d_gene", "j_gene",
    "v_deletions", "d5_deletions", "d3_deletions", "j_deletions", "n1_insertions", "n2_insertions",
    "v_index", "n1_index", "n2_index", "d_index", "j_index",
    "v_resolved", "d_resolved", "j_resolved", "sample_tags",
]

#: Sample-level columns, identical on every row -- taken from the first row only.
ADAPTIVE_SAMPLE_COLS = [
    "sample_name", "sample_tags", "counting_method", "primer_set", "total_templates",
    "productive_templates", "total_rearrangements", "productive_rearrangements",
    "outofframe_rearrangements", "stop_rearrangements",
]

_ADAPTIVE_FRAME = {"in": FRAME_IN, "out": FRAME_OUT, "stop": FRAME_STOP}


class AdaptiveReader:
    """Adaptive immuneACCESS export -> canonical event table. The coordinates are columns, so this is a
    read plus the -1 sentinel rule; nothing is derived and nothing is inferred."""

    platform = "adaptive"

    def __init__(self, chain: str = "TRB"):
        self.chain = chain

    def read(self, path: str) -> pd.DataFrame:
        head = pd.read_csv(path, sep="\t", nrows=0)
        use = [c for c in ADAPTIVE_COLS if c in head.columns]
        missing = set(ADAPTIVE_COLS) - set(use)
        if {"frame_type", "rearrangement", "cdr3_length"} & missing:
            raise ValueError(f"{os.path.basename(path)}: not an Adaptive export "
                             f"(missing {sorted({'frame_type','rearrangement','cdr3_length'} & missing)})")
        raw = pd.read_csv(path, sep="\t", usecols=use, low_memory=False)

        sample = str(raw["sample_name"].iloc[0]) if "sample_name" in raw and len(raw) else \
            os.path.basename(path).split(".")[0]
        ft = raw["frame_type"].astype(str).str.strip().str.lower().map(_ADAPTIVE_FRAME)
        d_res = raw["d_resolved"].astype(str).str.strip() if "d_resolved" in raw else pd.Series([""] * len(raw))
        d_present = (d_res != "") & (d_res.str.lower() != "nan")

        ins_n1 = _to_int(raw["n1_insertions"])
        ins_n2 = _to_int(raw["n2_insertions"])
        frame = {
            "rearrangement": raw["rearrangement"].astype(str).to_numpy(),
            "templates": raw["templates"].to_numpy() if "templates" in raw else np.ones(len(raw)),
            "frame_type": ft.fillna("").to_numpy(),
            "cdr3_length": _to_int(raw["cdr3_length"]),
            "v_gene": raw.get("v_gene", pd.Series([""] * len(raw))).astype(str).to_numpy(),
            "d_gene": raw.get("d_gene", pd.Series([""] * len(raw))).astype(str).to_numpy(),
            "j_gene": raw.get("j_gene", pd.Series([""] * len(raw))).astype(str).to_numpy(),
            "v_resolved": raw.get("v_resolved", pd.Series([""] * len(raw))).astype(str).to_numpy(),
            "d_resolved": d_res.to_numpy(),
            "j_resolved": raw.get("j_resolved", pd.Series([""] * len(raw))).astype(str).to_numpy(),
            "del_v": _to_int(raw["v_deletions"]),
            "del_d5": _to_int(raw["d5_deletions"]),
            "del_d3": _to_int(raw["d3_deletions"]),
            "del_j": _to_int(raw["j_deletions"]),
            "ins_n1": ins_n1,
            "ins_n2": ins_n2,
            # Total inserted nucleotides between V and J: defined whether or not a D was assigned, which is
            # what makes D-less rows comparable with D-bearing ones on the one coordinate they share.
            "ins_vj": np.where((ins_n1 >= 0) & (ins_n2 >= 0), ins_n1 + ins_n2, MISSING),
            "v_index": _to_int(raw["v_index"]),
            "n1_index": _to_int(raw["n1_index"]),
            "n2_index": _to_int(raw["n2_index"]),
            "d_index": _to_int(raw["d_index"]),
            "j_index": _to_int(raw["j_index"]),
            "d_present": d_present.to_numpy(),
        }
        return _finalize(frame, donor=sample, sample=sample, chain=self.chain)


# -- MiXCR (FMBA, T1D) ---------------------------------------------------------------------------------

MIXCR_COLS = ["cloneCount", "readCount", "uniqueMoleculeCount", "nSeqCDR3", "aaSeqCDR3",
              "allVHitsWithScore", "allDHitsWithScore", "allJHitsWithScore", "refPoints"]

#: refPoints is a 22-field colon-separated string; only 9..18 are ever populated (verified on 293,952 rows).
_RP_V_TRIM, _RP_V_END = 10, 11
_RP_D_BEG, _RP_D5_TRIM, _RP_D3_TRIM, _RP_D_END = 12, 13, 14, 15
_RP_J_BEG, _RP_J_TRIM, _RP_CDR3_END = 16, 17, 18

_SCORE_RE = re.compile(r"\([\d.eE+-]+\)$")


def _first_gene(hits: str) -> str:
    """MiXCR packs V/J into `allVHitsWithScore` as `GENE*allele(score),GENE*allele(score)`. Best hit first."""
    if not isinstance(hits, str) or not hits:
        return ""
    return _SCORE_RE.sub("", hits.split(",")[0]).strip()


class MixcrReader:
    """MiXCR clonotype table -> canonical event table, with the coordinates recovered from `refPoints`.

    Validated on 293,952 rows of FMBA_vaccine: the length identity
    `v_end + ins_n1 + (d_end - d_beg) + ins_n2 + (cdr3_len - j_beg) == cdr3_len` holds in 229,006 of 229,006
    D-bearing rows, and `v_end + ins_vj + (cdr3_len - j_beg) == cdr3_len` in 64,946 of 64,946 D-less rows.

    A POSITIVE trim value is a P-nucleotide, not a deletion -- checked by reverse-complement against the
    reconstructed germline end (V 96.5%, J 95.2% agreement). So deletions take max(0, -trim) and the
    palindromic nucleotides are kept as their own coordinates."""

    platform = "mixcr"

    def __init__(self, chain: str = "TRB"):
        self.chain = chain

    def read(self, path: str) -> pd.DataFrame:
        head = pd.read_csv(path, sep="\t", nrows=0)
        use = [c for c in MIXCR_COLS if c in head.columns]
        if "refPoints" not in use or "nSeqCDR3" not in use:
            raise ValueError(f"{os.path.basename(path)}: no refPoints/nSeqCDR3 -- not a MiXCR clonotype table")
        raw = pd.read_csv(path, sep="\t", usecols=use, low_memory=False)
        sample = os.path.basename(path).split(".")[0]

        rp = raw["refPoints"].astype(str).str.split(":", expand=True)
        g = lambda i: _to_int(rp[i]) if i in rp.columns else np.full(len(raw), MISSING)

        v_trim, v_end = g(_RP_V_TRIM), g(_RP_V_END)
        d_beg, d5_trim, d3_trim, d_end = g(_RP_D_BEG), g(_RP_D5_TRIM), g(_RP_D3_TRIM), g(_RP_D_END)
        j_beg, j_trim = g(_RP_J_BEG), g(_RP_J_TRIM)
        cdr3_len = raw["nSeqCDR3"].astype(str).str.len().to_numpy()

        d_present = d_beg != MISSING
        ins_n1 = np.where(d_present, d_beg - v_end, MISSING)
        ins_n2 = np.where(d_present, j_beg - d_end, MISSING)
        ins_vj = j_beg - v_end  # the V-J gap; equals ins_n1 + ins_n2 + (d_end - d_beg) when D is present

        aa = raw["aaSeqCDR3"].astype(str) if "aaSeqCDR3" in raw else pd.Series([""] * len(raw))
        frame_type = np.where(aa.str.contains("_", regex=False), FRAME_OUT,
                              np.where(aa.str.contains("*", regex=False), FRAME_STOP, FRAME_IN))

        count_col = next((c for c in ("uniqueMoleculeCount", "cloneCount", "readCount") if c in raw), None)
        frame = {
            "rearrangement": raw["nSeqCDR3"].astype(str).to_numpy(),
            "templates": raw[count_col].to_numpy() if count_col else np.ones(len(raw)),
            "frame_type": frame_type,
            "cdr3_length": cdr3_len.astype("int32"),
            "v_gene": raw.get("allVHitsWithScore", pd.Series([""] * len(raw))).map(_first_gene).to_numpy(),
            "d_gene": raw.get("allDHitsWithScore", pd.Series([""] * len(raw))).map(_first_gene).to_numpy(),
            "j_gene": raw.get("allJHitsWithScore", pd.Series([""] * len(raw))).map(_first_gene).to_numpy(),
            "del_v": np.maximum(0, -v_trim),
            "del_d5": np.where(d_present, np.maximum(0, -d5_trim), MISSING),
            "del_d3": np.where(d_present, np.maximum(0, -d3_trim), MISSING),
            "del_j": np.maximum(0, -j_trim),
            "p_v": np.maximum(0, v_trim),
            "p_d5": np.where(d_present, np.maximum(0, d5_trim), MISSING),
            "p_d3": np.where(d_present, np.maximum(0, d3_trim), MISSING),
            "p_j": np.maximum(0, j_trim),
            "ins_n1": ins_n1,
            "ins_n2": ins_n2,
            "ins_vj": ins_vj,
            "v_index": v_end,
            "n1_index": np.where(d_present, v_end, MISSING),
            "n2_index": np.where(d_present, d_end, MISSING),
            "d_index": d_beg,
            "j_index": j_beg,
            "d_present": d_present,
        }
        return _finalize(frame, donor=sample, sample=sample, chain=self.chain)


READERS = {"adaptive": AdaptiveReader, "mixcr": MixcrReader}


# -- validation ----------------------------------------------------------------------------------------

def validate_events(df: pd.DataFrame, platform: str = "adaptive") -> dict:
    """Re-derive the invariants the readers rely on and report them as counts, never as a silent clamp.

    Returns a dict of counters. Every `*_ok` entry must equal its `*_n` partner; a shortfall means the
    reader's understanding of the format is wrong, not that the data is noisy."""
    out = {"rows": int(len(df))}
    for cls in (FRAME_IN, FRAME_OUT, FRAME_STOP):
        out[f"frame_{cls}"] = int((df["frame_type"] == cls).sum())
    out["d_absent"] = int((~df["d_present"]).sum())

    # The frame class must be exactly what the junction length says: productive <=> length divisible by 3.
    mod = (df["cdr3_length"].to_numpy() % 3)
    for cls in (FRAME_IN, FRAME_OUT, FRAME_STOP):
        m = (df["frame_type"] == cls).to_numpy()
        out[f"{cls}_mod0"] = int((mod[m] == 0).sum())
        out[f"{cls}_n"] = int(m.sum())
    out["mod3_rule_ok"] = int(out["in_mod0"] == out["in_n"]
                              and out["stop_mod0"] == out["stop_n"]
                              and out["out_mod0"] == 0)

    if platform == "adaptive":
        # The -1 sentinel: a negative insertion index means there was no insertion, and only then does the
        # gap identity not apply. Read without it, these come out at 83-88% and look like a parse bug.
        for tag, ins, idx, far in (("n1", "ins_n1", "n1_index", "d_index"),
                                   ("n2", "ins_n2", "n2_index", "j_index")):
            i, x, f = df[ins].to_numpy(), df[idx].to_numpy(), df[far].to_numpy()
            neg = x < 0
            out[f"{tag}_sentinel_n"] = int(neg.sum())
            out[f"{tag}_sentinel_ok"] = int((i[neg] == 0).sum())
            out[f"{tag}_gap_n"] = int((~neg).sum())
            out[f"{tag}_gap_ok"] = int((i[~neg] == (f[~neg] - x[~neg])).sum())
    else:
        # MiXCR: the coordinates must reconstruct the junction length exactly, both with and without a D.
        d = df["d_present"].to_numpy()
        L, v, jb = df["cdr3_length"].to_numpy(), df["v_index"].to_numpy(), df["j_index"].to_numpy()
        n1, n2, db, de = (df["ins_n1"].to_numpy(), df["ins_n2"].to_numpy(),
                          df["d_index"].to_numpy(), df["n2_index"].to_numpy())
        out["len_withD_n"] = int(d.sum())
        out["len_withD_ok"] = int((v[d] + n1[d] + (de[d] - db[d]) + n2[d] + (L[d] - jb[d]) == L[d]).sum())
        out["len_noD_n"] = int((~d).sum())
        out["len_noD_ok"] = int((v[~d] + df["ins_vj"].to_numpy()[~d] + (L[~d] - jb[~d]) == L[~d]).sum())
    return out


def _report(counts: dict) -> str:
    pairs = [(k[:-3], counts[k], counts.get(k[:-3] + "_n")) for k in counts if k.endswith("_ok")
             and isinstance(counts.get(k[:-3] + "_n"), int)]
    lines = [f"  rows {counts['rows']}  in {counts.get('frame_in',0)}  out {counts.get('frame_out',0)}"
             f"  stop {counts.get('frame_stop',0)}  D-absent {counts.get('d_absent',0)}"]
    for name, ok, n in pairs:
        flag = "OK " if ok == n else "FAIL"
        lines.append(f"  {flag} {name}: {ok}/{n}")
    lines.append(f"  {'OK ' if counts.get('mod3_rule_ok') else 'FAIL'} mod3 rule "
                 f"(in+stop all divisible by 3, out none)")
    return "\n".join(lines)


# -- sample tags ---------------------------------------------------------------------------------------

def parse_sample_tags(tags: str) -> dict:
    """Adaptive packs the donor-level labels into one comma-separated `Key:Value` string.

    Known typing and INFERRED typing are kept in separate fields on purpose: the inferred HLA in this corpus
    was produced by a model fitted to these very repertoires (Emerson et al. 2017), so scoring a repertoire
    model against it is circular. `hla_known` is the label; `hla_inferred` is a covariate at best."""
    out = {"cohort": "", "cmv": "", "hla_known": "", "hla_inferred": "",
           "age": MISSING, "sex": "", "ethnic_group": "", "racial_group": ""}
    if not isinstance(tags, str) or not tags:
        return out
    known, inferred = [], []
    for item in tags.split(","):
        if ":" not in item:
            continue
        key, _, val = item.partition(":")
        key, val = key.strip(), val.strip()
        if key == "Cohort":
            out["cohort"] = val
        elif key == "Virus Diseases" and val.startswith("Cytomegalovirus"):
            out["cmv"] = "+" if val.rstrip().endswith("+") else ("-" if val.rstrip().endswith("-") else "")
        elif key.startswith("HLA MHC class"):
            known.append(val)
        elif key == "Inferred HLA type":
            inferred.append(val.replace("Inferred ", ""))
        elif key == "Age":
            m = re.match(r"(\d+)", val)
            if m:
                out["age"] = int(m.group(1))
        elif key == "Biological Sex":
            out["sex"] = val
        elif key == "Ethnic Group":
            out["ethnic_group"] = val
        elif key == "Racial Group":
            out["racial_group"] = val
    out["hla_known"] = ";".join(sorted(set(known)))
    out["hla_inferred"] = ";".join(sorted(set(inferred)))
    return out


def read_sample_metadata(path: str) -> dict:
    """Donor-level row for one Adaptive file: the parsed tags plus the export's own depth counters."""
    head = pd.read_csv(path, sep="\t", nrows=0)
    use = [c for c in ADAPTIVE_SAMPLE_COLS if c in head.columns]
    row = pd.read_csv(path, sep="\t", usecols=use, nrows=1)
    rec = {c: (row[c].iloc[0] if c in row else "") for c in ADAPTIVE_SAMPLE_COLS}
    sample = str(rec.get("sample_name") or os.path.basename(path).split(".")[0])
    meta = {"donor": sample, "sample": sample, "file": os.path.basename(path)}
    meta.update(parse_sample_tags(rec.get("sample_tags", "")))
    for c in ADAPTIVE_SAMPLE_COLS:
        if c.endswith(("_templates", "_rearrangements")) or c in ("counting_method", "primer_set"):
            meta[c] = rec.get(c, "")
    return meta


# -- corpus build --------------------------------------------------------------------------------------

def build_corpus(input_dir: str, out_dir: str, platform: str = "adaptive", pattern: str = "*.tsv",
                 chain: str = "TRB", limit: int | None = None, report_only: bool = False,
                 identity_map: str | None = None,
                 overwrite: bool = False, shard: int = 0, nshards: int = 1) -> dict:
    """Read every raw sample file into the canonical event table, one parquet per library.

    Partitioned by library so productive and non-productive rearrangements from one tube stay together.
    The optional identity map supplies the biological donor and, for longitudinal data, the timepoint. Both
    classes go into the SAME table with
    `frame_type` as a column -- a separate "non-productive" file would reproduce, at the storage layer, the
    very split that made the null unavailable in the first place, and the two halves would drift apart on
    the first re-run.

    Returns the aggregate validation counters; per-file counters are printed as it goes."""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    files = [f for f in files if os.path.basename(f) != "metadata.tsv"]
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {input_dir}")
    if nshards > 1:
        # Stride, not a contiguous block: file size varies several-fold across the corpus, so a stride
        # gives every task a similar mix and the array finishes together instead of trailing one long task.
        files = files[shard::nshards]
        print(f"shard {shard}/{nshards}: {len(files)} files")

    reader = READERS[platform](chain=chain)
    identity = _load_identity_map(identity_map) if identity_map else None
    ev_dir = os.path.join(out_dir, "events")
    if not report_only:
        os.makedirs(ev_dir, exist_ok=True)

    total: dict = {}
    metas, skipped = [], []
    for i, path in enumerate(files, 1):
        stem = os.path.basename(path).split(".")[0]
        out_path = os.path.join(ev_dir, f"{stem}.parquet")
        if not report_only and os.path.exists(out_path) and not overwrite:
            print(f"[{i}/{len(files)}] {stem}: exists, skipped")
            skipped.append((stem, "output already exists; rerun with --overwrite to rebuild"))
            continue
        try:
            df = reader.read(path)
        except Exception as exc:                      # one malformed file must not kill an 8-hour build
            print(f"[{i}/{len(files)}] {stem}: SKIPPED -- {exc}")
            skipped.append((stem, str(exc)))
            continue
        df = _apply_identity(df, identity)

        before = len(df)
        df = df.drop_duplicates(subset=["sample", "v_gene", "d_gene", "j_gene", "rearrangement"])
        counts = validate_events(df, platform=platform)
        counts["dedup_removed"] = before - len(df)
        for k, v in counts.items():
            total[k] = total.get(k, 0) + (v if isinstance(v, int) else 0)

        if not report_only:
            df.to_parquet(out_path, index=False, compression="snappy")
            if platform == "adaptive":
                meta = read_sample_metadata(path)
                for c in ("donor", "library", "timepoint"):
                    meta[c] = str(df[c].iloc[0])
                metas.append(meta)
        dup = counts["dedup_removed"]
        print(f"[{i}/{len(files)}] {stem}: {before} rows" + (f" (-{dup} duplicate)" if dup else ""))
        print(_report(counts))

    if not report_only:
        manifest = {"platform": platform, "input_pattern": pattern, "identity_map": identity_map,
                    "n_input_files": len(files), "n_skipped": len(skipped),
                    "skipped": [{"sample": sample, "error": error} for sample, error in skipped]}
        manifest_path = os.path.join(out_dir, "build_manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"wrote build manifest -> {manifest_path}")

    if not report_only and metas:
        meta_df = pd.DataFrame(metas)
        # One file per shard; array tasks would otherwise overwrite each other. merge_sample_metadata()
        # concatenates them once the array is done.
        name = "samples.parquet" if nshards == 1 else f"samples_shard{shard:03d}.parquet"
        meta_df.to_parquet(os.path.join(out_dir, name), index=False)
        print(f"\nwrote {len(meta_df)} donor metadata rows -> {os.path.join(out_dir, name)}")
        if "cmv" in meta_df:
            print("  CMV:", meta_df["cmv"].replace("", "unknown").value_counts().to_dict())
            print("  cohort:", meta_df["cohort"].replace("", "unknown").value_counts().to_dict())

    print("\n=== corpus totals ===")
    print(_report(total))
    print(f"  duplicate rows removed: {total.get('dedup_removed', 0)}")
    if skipped:
        print(f"  files skipped: {len(skipped)} -> {skipped[:5]}")
    return total


def canonicalize_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw platform gene spellings onto the encoder's vocabulary via tidytcells (lazy import).

    Kept out of the build so the stored corpus stays a faithful copy of what the platform reported; a
    canonicalisation can be re-run, a discarded spelling cannot be recovered."""
    import tidytcells as tt  # lazy: only needed when joining to the amino-acid side

    def std(x):
        try:
            return tt.tr.standardise(gene=str(x), species="homosapiens")
        except Exception:
            return None

    res = df.copy()
    for col in ("v_gene", "d_gene", "j_gene"):
        uniq = {g: std(g) for g in res[col].astype(str).unique()}
        res[col] = res[col].astype(str).map(uniq)
    return res


def main():
    import argparse
    p = argparse.ArgumentParser(description="Build the canonical recombination-event corpus.")
    p.add_argument("--input", required=True, help="directory of raw per-sample files")
    p.add_argument("--out", help="output directory (events/ + samples.parquet); omit with --report-only")
    p.add_argument("--platform", default="adaptive", choices=sorted(READERS))
    p.add_argument("--pattern", default="*.tsv")
    p.add_argument("--chain", default="TRB")
    p.add_argument("--identity-map", default=None,
                   help="TSV or parquet with sample, donor, and optional library/timepoint columns")
    p.add_argument("--limit", type=int, default=None, help="read only the first N files (smoke test)")
    p.add_argument("--report-only", action="store_true", help="validate and print, write nothing")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shard", type=int, default=0, help="this task's index in a SLURM job array")
    p.add_argument("--nshards", type=int, default=1)
    a = p.parse_args()
    if not a.report_only and not a.out:
        p.error("--out is required unless --report-only")
    build_corpus(a.input, a.out or "", platform=a.platform, pattern=a.pattern, chain=a.chain,
                 limit=a.limit, report_only=a.report_only, identity_map=a.identity_map, overwrite=a.overwrite,
                 shard=a.shard, nshards=a.nshards)


if __name__ == "__main__":
    main()
