"""OAR correction of productive clonotype weights.

For each sample and receptor chain, non-productive rearrangements provide a
within-sample calibration set for multiplex-PCR amplification bias.  The
over-amplification rate (OAR) of a V or J gene is its template-count
frequency divided by its unique-clonotype frequency among the non-productive
events.  Productive template counts are divided by the product of their V and
J OARs.  The module never changes its inputs: it writes raw and corrected
weights side by side.

The calculation follows Smirnova et al., eLife 2023, doi:10.7554/eLife.69157.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .events import AdaptiveReader, FRAME_IN, FRAME_OUT, FRAME_STOP, MixcrReader, best_gene


NONPRODUCTIVE_FRAMES = frozenset(("out", "stop"))
PRODUCTIVE_FRAME = "in"
_REQUIRED = frozenset(("sample", "chain", "frame_type", "v_gene", "j_gene", "rearrangement", "templates"))
_CODONS = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _validate_events(frame: pd.DataFrame, *, label: str, allowed_frames: frozenset[str]) -> pd.DataFrame:
    """Validate the canonical columns needed for OAR without altering events."""
    missing = sorted(_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{label} input is missing required columns {missing}")
    out = frame.copy()
    out["frame_type"] = out["frame_type"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(out["frame_type"]) - allowed_frames)
    if unexpected:
        raise ValueError(f"{label} input contains frame types outside {sorted(allowed_frames)}: {unexpected}")
    for column in ("sample", "chain", "v_gene", "j_gene", "rearrangement"):
        values = out[column].astype(object)
        out[column] = values.where(values.notna(), "").astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"{label} input contains an empty {column!r}")
    out["templates"] = pd.to_numeric(out["templates"], errors="coerce")
    if (~np.isfinite(out["templates"]) | (out["templates"] <= 0)).any():
        raise ValueError(f"{label} input contains a non-positive or non-finite template count")
    return out


def _axis_oar(nonproductive: pd.DataFrame, gene_column: str, gene_axis: str,
              min_unique_clonotypes: int) -> pd.DataFrame:
    """Estimate normalized OAR values for one gene axis within every sample."""
    groups = ["sample", "chain"]
    clone_columns = [*groups, "v_gene", "j_gene", "rearrangement"]
    unique = nonproductive.drop_duplicates(clone_columns)
    expected = unique.groupby([*groups, gene_column], observed=True).size().rename("n_unique_clonotypes")
    expected = expected.to_frame().join(unique.groupby(groups, observed=True).size().rename("n_unique_total"))
    expected["expected_frequency"] = expected["n_unique_clonotypes"] / expected["n_unique_total"]

    observed = nonproductive.groupby([*groups, gene_column], observed=True)["templates"].sum().rename("template_count")
    observed = observed.to_frame().join(nonproductive.groupby(groups, observed=True)["templates"].sum().rename("template_total"))
    observed["observed_frequency"] = observed["template_count"] / observed["template_total"]

    result = expected.join(observed).reset_index().rename(columns={gene_column: "gene"})
    result["oar_raw"] = result["observed_frequency"] / result["expected_frequency"]
    normalized = result["oar_raw"] / result.groupby(groups, observed=True)["oar_raw"].transform("mean")
    sufficient = result["n_unique_clonotypes"] >= min_unique_clonotypes
    # iROAR's min_outframe rule: sparse gene-specific calibrators receive a
    # neutral factor rather than a noisy patient or population estimate.
    result["oar"] = normalized.where(sufficient, 1.0)
    result["oar_source"] = np.where(sufficient, "patient_nonproductive", "insufficient_nonproductive")
    result.insert(2, "gene_axis", gene_axis)
    return result[["sample", "chain", "gene_axis", "gene", "n_unique_clonotypes", "n_unique_total", "template_count", "template_total", "expected_frequency", "observed_frequency", "oar_raw", "oar", "oar_source"]]


def estimate_oar(nonproductive: pd.DataFrame, *, min_unique_clonotypes: int = 15) -> pd.DataFrame:
    """Estimate per-sample V and J OAR factors from non-productive events.

    ``min_unique_clonotypes`` applies to every V or J segment separately,
    following iROAR's ``min_outframe`` rule.  Sparse segment-specific
    calibrators receive ``OAR = 1`` and an explicit audit label; no
    population-level coefficient is substituted.
    """
    if min_unique_clonotypes < 1:
        raise ValueError("min_unique_clonotypes must be at least one")
    events = _validate_events(nonproductive, label="non-productive", allowed_frames=NONPRODUCTIVE_FRAMES)
    return pd.concat((
        _axis_oar(events, "v_gene", "V", min_unique_clonotypes),
        _axis_oar(events, "j_gene", "J", min_unique_clonotypes),
    ), ignore_index=True)


def correct_productive_weights(productive: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Return productive events with raw and OAR-corrected, sample-normalized weights."""
    events = _validate_events(productive, label="productive", allowed_frames=frozenset((PRODUCTIVE_FRAME,)))
    required_factor_columns = {"sample", "chain", "gene_axis", "gene", "oar", "oar_source"}
    missing = sorted(required_factor_columns - set(factors.columns))
    if missing:
        raise ValueError(f"OAR factors are missing required columns {missing}")
    out = events.copy()
    for axis, gene_column in (("V", "v_gene"), ("J", "j_gene")):
        axis_factors = factors.loc[factors["gene_axis"] == axis, ["sample", "chain", "gene", "oar", "oar_source"]].rename(columns={"gene": gene_column, "oar": f"oar_{axis.lower()}", "oar_source": f"oar_{axis.lower()}_source"})
        out = out.merge(axis_factors, on=["sample", "chain", gene_column], how="left", validate="many_to_one")
    for axis in ("v", "j"):
        out[f"oar_{axis}"] = out[f"oar_{axis}"].fillna(1.0)
        out[f"oar_{axis}_source"] = out[f"oar_{axis}_source"].fillna("absent_nonproductive")
    out["oar_coefficient"] = out["oar_v"] * out["oar_j"]
    out["templates_raw"] = out["templates"].astype(float)
    out["templates_oar"] = out["templates_raw"] / out["oar_coefficient"]
    groups = ["sample", "chain"]
    out["weight_raw"] = out["templates_raw"] / out.groupby(groups, observed=True)["templates_raw"].transform("sum")
    out["weight_oar"] = out["templates_oar"] / out.groupby(groups, observed=True)["templates_oar"].transform("sum")
    return out


def read_patient(path: str | Path, *, chain: str = "TRB") -> pd.DataFrame:
    """Read one patient from canonical parquet, Adaptive TSV, or MiXCR TSV.

    The returned canonical frame must retain both productive and non-productive
    rearrangements.  One-file processing is deliberate: OAR factors are
    estimated only from the patient whose productive weights they correct.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"patient file does not exist: {source}")
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
        if "cdr3aa" not in frame and "amino_acid" in frame:
            frame["cdr3aa"] = frame["amino_acid"]
    else:
        header = pd.read_csv(source, sep="\t", nrows=0).columns
        columns = set(header)
        if {"nSeqCDR3", "aaSeqCDR3", "refPoints"} <= columns:
            frame = MixcrReader(chain=chain).read(str(source))
            frame["cdr3aa"] = pd.read_csv(source, sep="\t", usecols=["aaSeqCDR3"])["aaSeqCDR3"].to_numpy()
        elif {"frame_type", "rearrangement", "cdr3_length"} <= columns:
            frame = AdaptiveReader(chain=chain).read(str(source))
            if "amino_acid" in columns:
                frame["cdr3aa"] = pd.read_csv(source, sep="\t", usecols=["amino_acid"], low_memory=False)["amino_acid"].to_numpy()
        else:
            raise ValueError(f"{source.name}: unsupported TSV; expected Adaptive or MiXCR columns")
    missing = sorted(_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{source.name}: canonical parquet is missing required columns {missing}")
    return frame


def translate_cdr3nt(rearrangements: pd.Series) -> pd.Series:
    """Translate productive CDR3 nucleotide junctions and reject malformed codons."""
    translated: list[str] = []
    for nt in rearrangements.astype(str):
        sequence = nt.upper().replace("U", "T")
        if len(sequence) == 0 or len(sequence) % 3:
            raise ValueError(f"productive CDR3nt has invalid length: {nt!r}")
        amino_acids = "".join(_CODONS.get(sequence[i:i + 3], "?") for i in range(0, len(sequence), 3))
        if "?" in amino_acids or "*" in amino_acids:
            raise ValueError(f"productive CDR3nt cannot be translated cleanly: {nt!r}")
        translated.append(amino_acids)
    return pd.Series(translated, index=rearrangements.index, dtype="string")


def _productive_cdr3aa(events: pd.DataFrame) -> pd.Series:
    """Use the platform's productive CDR3 amino-acid call, with nt fallback."""
    if "cdr3aa" not in events.columns:
        return translate_cdr3nt(events["rearrangement"])
    amino_acids = events["cdr3aa"].astype(object)
    clean = amino_acids.where(amino_acids.notna(), "").astype(str).str.strip().str.upper()
    missing = clean.eq("") | clean.str.contains(r"[*_]", regex=True)
    if missing.any():
        clean.loc[missing] = translate_cdr3nt(events.loc[missing, "rearrangement"])
    if clean.str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True).any():
        raise ValueError("productive CDR3aa contains unsupported amino-acid symbols")
    return clean.astype("string")


def _add_absent_productive_factors(factors: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    """Add neutral audit rows for productive genes absent from the calibrator."""
    additions: list[pd.DataFrame] = []
    for axis, gene_column in (("V", "v_gene"), ("J", "j_gene")):
        source_column = f"oar_{axis.lower()}_source"
        absent = corrected.loc[corrected[source_column] == "absent_nonproductive", ["sample", "chain", gene_column]].drop_duplicates()
        if absent.empty:
            continue
        rows = pd.DataFrame({
            "sample": absent["sample"].to_numpy(), "chain": absent["chain"].to_numpy(),
            "gene_axis": axis, "gene": absent[gene_column].to_numpy(),
            "n_unique_clonotypes": np.nan, "n_unique_total": np.nan,
            "template_count": np.nan, "template_total": np.nan,
            "expected_frequency": np.nan, "observed_frequency": np.nan,
            "oar_raw": np.nan, "oar": 1.0, "oar_source": "absent_nonproductive",
        })
        additions.append(rows)
    return pd.concat([factors, *additions], ignore_index=True) if additions else factors


def process_patient(frame: pd.DataFrame, *, min_unique_clonotypes: int = 15, oar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one patient's productive clonotypes for a raw or OAR cloud.

    All intermediate quantities remain in memory.  With ``oar=True``, ``out``
    and ``stop`` events calibrate V/J factors that correct productive template
    counts.  With ``oar=False``, the same productive-event reader and output
    schema are used, but the effective count equals the raw template count and
    all OAR fields are explicitly neutral.
    """
    event_frame = frame.copy()
    if {"v_resolved", "j_resolved"} <= set(event_frame.columns):
        event_frame["v_gene"] = best_gene(event_frame, "v")
        event_frame["j_gene"] = best_gene(event_frame, "j")
    event_frame["frame_type"] = event_frame["frame_type"].astype(str).str.strip().str.lower()
    allowed = NONPRODUCTIVE_FRAMES | frozenset((PRODUCTIVE_FRAME,))
    unexpected = sorted(set(event_frame["frame_type"]) - allowed)
    if unexpected:
        raise ValueError(f"patient input contains unsupported frame types: {unexpected}")
    productive = event_frame.loc[event_frame["frame_type"] == PRODUCTIVE_FRAME].copy()
    if productive.empty:
        raise ValueError("patient input has no productive in-frame rearrangements")
    if oar:
        nonproductive = event_frame.loc[event_frame["frame_type"].isin(NONPRODUCTIVE_FRAMES)].copy()
        if nonproductive.empty:
            raise ValueError("patient input has no non-productive out/stop rearrangements")
        factors = estimate_oar(nonproductive, min_unique_clonotypes=min_unique_clonotypes)
        corrected = correct_productive_weights(productive, factors)
        factors = _add_absent_productive_factors(factors, corrected)
    else:
        corrected = productive.copy()
        corrected["templates_raw"] = corrected["templates"].astype(float)
        corrected["templates_oar"] = corrected["templates_raw"]
        corrected["oar_v"] = 1.0
        corrected["oar_j"] = 1.0
        corrected["oar_coefficient"] = 1.0
        corrected["oar_v_source"] = "not_applied"
        corrected["oar_j_source"] = "not_applied"
        factors = pd.DataFrame(columns=["sample", "chain", "gene_axis", "gene", "n_unique_clonotypes", "n_unique_total", "template_count", "template_total", "expected_frequency", "observed_frequency", "oar_raw", "oar", "oar_source"])
    corrected["cdr3aa"] = _productive_cdr3aa(corrected)
    corrected["count_raw"] = corrected["templates_raw"]
    corrected["count"] = corrected["templates_oar"]
    log_counts = np.log1p(corrected["count"].to_numpy(dtype=float))
    corrected["w_log"] = log_counts / corrected.groupby(["sample", "chain"], observed=True)["count"].transform(lambda x: np.log1p(x).sum()).to_numpy()
    columns = ["sample", "chain", "cdr3aa", "v_gene", "j_gene", "count_raw", "oar_v", "oar_v_source", "oar_j", "oar_j_source", "oar_coefficient", "count", "w_log"]
    return factors, corrected[columns]


def read_parquet_input(path: str | Path) -> pd.DataFrame:
    """Read one parquet file or all top-level parquet files in a directory."""
    root = Path(path)
    paths = [root] if root.is_file() else sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found at {root}")
    return pd.concat((pd.read_parquet(item) for item in paths), ignore_index=True)


def run(nonproductive_path: str | Path, productive_path: str | Path, out_dir: str | Path, *, min_unique_clonotypes: int = 15) -> tuple[Path, Path]:
    """Write OAR factors and corrected productive events to a new output directory."""
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    factors = estimate_oar(read_parquet_input(nonproductive_path), min_unique_clonotypes=min_unique_clonotypes)
    corrected = correct_productive_weights(read_parquet_input(productive_path), factors)
    output.mkdir(parents=True)
    factor_path = output / "oar_factors.parquet"
    weights_path = output / "productive_weights_oar.parquet"
    factors.to_parquet(factor_path, index=False)
    corrected.to_parquet(weights_path, index=False)
    manifest = {"nonproductive": str(nonproductive_path), "productive": str(productive_path), "min_unique_clonotypes": min_unique_clonotypes, "factor_file": factor_path.name, "weights_file": weights_path.name}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return factor_path, weights_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Correct one patient's productive clonotypes from its own non-productive OAR calibrator.")
    parser.add_argument("--input", required=True, help="one canonical parquet, Adaptive TSV, or MiXCR TSV patient file")
    parser.add_argument("--out", required=True, help="corrected productive clonotype parquet")
    parser.add_argument("--chain", default="TRB", help="chain label for raw TSV input (default: TRB)")
    parser.add_argument("--min-unique-clonotypes", type=int, default=15, help="minimum unique non-productive clonotypes per V/J segment (default: 15)")
    args = parser.parse_args(argv)
    factors, corrected = process_patient(read_patient(args.input, chain=args.chain), min_unique_clonotypes=args.min_unique_clonotypes)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"output file already exists: {output}")
    corrected.to_parquet(output, index=False)
    print(f"sample(s): {sorted(corrected['sample'].astype(str).unique())}")
    print(f"OAR factors retained in memory: {len(factors)}")
    print(f"wrote corrected productive clonotypes -> {output}")


if __name__ == "__main__":
    main()
