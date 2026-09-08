"""Fit IGoR models from MiXCR TSV or canonical event parquet inputs.

IGoR receives only an indexed list of CDR3 nucleotide sequences. This module
converts supported input tables to ``frame_type`` and ``cdr3nt``, selects
non-productive events, and creates that transient IGoR input only for the duration
of an inference. V/D/J usage is inferred by IGoR from its reference bundle.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..recombination.events import FRAME_OUT, FRAME_STOP


# Without `@dataclass`, `Scope()` would create an empty object and the four fields would have to be 
# assigned one by one. `@dataclass` permits `Scope(name, source_files, counts, sequences)` instead. 
# `frozen=True` prevents changing the fields after the IGoR task has been created.
@dataclass(frozen=True)
class Scope:
    # One complete IGoR task. `build_scopes()` creates these objects, and
    # `fit_scope()` consumes them without needing to inspect the input files again.
    name: str
    source_files: tuple[str, ...]
    source_sequence_counts: tuple[tuple[str, int], ...]
    sequences: tuple[str, ...]
    # ``individual`` means one input library, ``patient`` a union of one or more
    # libraries from the same donor, and ``group`` a cohort-wide pooled fit.
    kind: str = "individual"


def _safe_name(path: Path) -> str:
    # Convert an input filename into a name safe to use as an output directory.
    parts = re.split(r"[\W_]+", path.stem)
    return "_".join(part for part in parts if part)


def _frame_type_from_mixcr(aa: pd.Series) -> pd.Series:
    # MiXCR marks an out-of-frame CDR3 with `_` and a stop-carrying CDR3 with `*`.
    aa = aa.fillna("").astype(str)
    return pd.Series(np.where(aa.str.contains("_", regex=False), FRAME_OUT,
                              np.where(aa.str.contains("*", regex=False), FRAME_STOP, "in")), index=aa.index)


def _read_mixcr(path: Path) -> pd.DataFrame:
    # MiXCR already exports CDR3nt directly, so only its nucleotide and frame fields
    # are needed. The return value is normalized to the common two-column table.
    header = pd.read_csv(path, sep="\t", nrows=0)
    required = {"nSeqCDR3", "aaSeqCDR3"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"{path}: MiXCR TSV lacks {sorted(missing)}")
    raw = pd.read_csv(path, sep="\t", usecols=["nSeqCDR3", "aaSeqCDR3"], low_memory=False)
    return pd.DataFrame({"frame_type": _frame_type_from_mixcr(raw["aaSeqCDR3"]),
                         "cdr3nt": raw["nSeqCDR3"].astype(str).str.upper()})


def _read_canonical_parquet(path: Path) -> pd.DataFrame:
    # In a canonical event table, CDR3nt is a slice of `rearrangement`: it starts at
    # `v_index` and spans `cdr3_length` nucleotides. Invalid coordinates are excluded.
    required = ["frame_type", "rearrangement", "v_index", "cdr3_length"]
    frame = pd.read_parquet(path, columns=required)
    sequence = frame["rearrangement"].astype(str).str.upper()
    start, length = frame["v_index"].astype(int), frame["cdr3_length"].astype(int)
    valid = (start >= 0) & (length > 0) & (start + length <= sequence.str.len())
    return pd.DataFrame({"frame_type": frame.loc[valid, "frame_type"].astype(str),
                         "cdr3nt": [s[a:a + n] for s, a, n in zip(sequence[valid], start[valid], length[valid])]},
                        index=frame.index[valid])


def read_input(path: str | Path) -> pd.DataFrame:
    """Read one supported file into the two columns needed for IGoR preparation."""
    path = Path(path)
    # The filename extension selects the appropriate reader; both readers return the
    # same columns, allowing all downstream code to be format-independent.
    if path.suffix.lower() == ".tsv":
        frame = _read_mixcr(path)
    elif path.suffix.lower() == ".parquet":
        frame = _read_canonical_parquet(path)
    else:
        raise ValueError(f"{path}: expected MiXCR .tsv or canonical .parquet")
    # IGoR should receive only complete A/C/G/T strings, not ambiguous or malformed reads.
    return frame.loc[frame["cdr3nt"].str.fullmatch("[ACGT]+", na=False)].copy()


def input_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        # Recursive discovery permits a directory tree with patient subdirectories.
        files = sorted([*path.rglob("*.tsv"), *path.rglob("*.parquet")])
        if files:
            return files
    raise FileNotFoundError(f"{path}: expected a supported file or a directory containing .tsv/.parquet files")


def _patient_scope_name(donor: str) -> str:
    """Convert an explicit donor identifier into a safe output-directory name."""
    parts = re.split(r"[\W_]+", donor)
    name = "_".join(part for part in parts if part)
    if not name:
        raise ValueError(f"cannot derive an output name from donor identifier {donor!r}")
    return name


def _patient_identifier(path: Path, patient_pattern: re.Pattern[str]) -> str:
    """Extract one declared donor identifier from a library filename."""
    match = patient_pattern.search(path.stem)
    if match is None:
        raise ValueError(f"{path}: filename does not match the patient grouping pattern")
    donor = match.group(1)
    if not donor:
        raise ValueError(f"{path}: patient grouping pattern captured an empty donor identifier")
    return donor


def build_scopes(path: str | Path, mode: str, *, patient_pattern: str | None = None) -> list[Scope]:
    """Create one scope per file (individual) or one pooled directory scope (group).
    
    De-duplication occurs within an input file. In group mode, repetitions across
    files remain independent observations and are retained in the IGoR list.
    """
    files = input_files(path)
    per_file = []
    seen_names: set[str] = set()
    for file in files:
        name = _safe_name(file)
        if name in seen_names:
            raise ValueError(f"duplicate output name {name!r}; rename one input file")
        seen_names.add(name)
        frame = read_input(file)
        # A clonotype is counted once within its source file. Read and UMI counts are
        # deliberately excluded because this fit estimates rearrangement diversity,
        # rather than clonal abundance or PCR-amplified abundance.
        sequences = tuple(sorted(frame.loc[frame["frame_type"].isin((FRAME_OUT, FRAME_STOP)), "cdr3nt"].unique()))
        if sequences:
            per_file.append((file, sequences))
    if not per_file:
        raise ValueError("no valid non-productive CDR3nt sequences found")
    if mode == "individual":
        # Each input file becomes one independent model and one output directory.
        return [Scope(_safe_name(file), (str(file),), ((str(file), len(sequences)),), sequences, "individual")
                for file, sequences in per_file]
    if mode == "group":
        # Duplicates across files remain: they are observations from distinct source
        # samples and must not be globally deduplicated before a group-level fit.
        return [Scope("group", tuple(str(file) for file, _ in per_file),
                      tuple((str(file), len(sequences)) for file, sequences in per_file),
                      tuple(sequence for _, sequences in per_file for sequence in sequences), "group")]
    if mode == "patient":
        if patient_pattern is None:
            raise ValueError("patient mode requires a --patient-pattern with one donor capture group")
        pattern = re.compile(patient_pattern)
        if pattern.groups != 1:
            raise ValueError("patient grouping pattern must contain exactly one capturing group for the donor identifier")
        # Build the union from raw library inputs, rather than trying to average two
        # fitted IGoR models. A patient with one source library is represented by the
        # same path; a patient with CD4 and CD8 libraries receives their sequence union.
        sources_by_donor: dict[str, list[tuple[Path, tuple[str, ...]]]] = {}
        for file in files:
            donor = _patient_identifier(file, pattern)
            frame = read_input(file)
            sequences = tuple(sorted(frame.loc[frame["frame_type"].isin((FRAME_OUT, FRAME_STOP)), "cdr3nt"].unique()))
            sources_by_donor.setdefault(donor, []).append((file, sequences))
        scopes: list[Scope] = []
        used_names: set[str] = set()
        for donor in sorted(sources_by_donor):
            sources = sources_by_donor[donor]
            scope_name = _patient_scope_name(donor)
            if scope_name in used_names:
                raise ValueError(f"patient identifiers produce duplicate output name {scope_name!r}")
            used_names.add(scope_name)
            # A nucleotide rearrangement shared by CD4 and CD8 is one unique patient
            # observation, so library depth never changes its IGoR weight.
            union = tuple(sorted({sequence for _, sequences in sources for sequence in sequences}))
            if not union:
                continue
            scopes.append(Scope(scope_name, tuple(str(file) for file, _ in sources),
                                tuple((str(file), len(sequences)) for file, sequences in sources),
                                union, "patient"))
        if not scopes:
            raise ValueError("no valid non-productive CDR3nt sequences found after patient pooling")
        return scopes
    raise ValueError(f"unknown mode {mode!r}")


def igor_commands(igor: str, work_dir: Path, batch: str, input_csv: Path, *, iterations: int, threads: int) -> list[list[str]]:
    # IGoR uses three separate commands: index the input list, align it to its germline
    # reference bundle, then estimate recombination parameters by EM inference.
    root = [igor, "-set_wd", str(work_dir), "-batch", batch, "-threads", str(threads)]
    model = ["-species", "human", "-chain", "beta"]
    return [root + ["-read_seqs", str(input_csv)], root + model + ["-align", "--all", "--ntCDR3"],
            root + model + ["-infer", "--N_iter", str(iterations)]]


def _igor_version(igor: str) -> str:
    # Store the executable version in the manifest because model files can depend on it.
    return subprocess.run([igor, "-version"], check=True, capture_output=True, text=True).stdout.strip()


def _native_igor_complete(destination: Path) -> bool:
    """Return whether IGoR has written the two native final parameter files.

    A process can be interrupted after IGoR has finished but before this wrapper
    writes ``manifest.json``. Native final files are therefore the completion
    criterion for recovery; the manifest is provenance, not an inference result.
    """
    inference = destination / "model_inference"
    return (inference / "final_marginals.txt").is_file() and (inference / "final_parms.txt").is_file()


def _interrupted_destination(destination: Path) -> Path:
    """Return a non-conflicting recoverable name for one interrupted scope."""
    base = destination.with_name(f"{destination.name}.interrupted")
    candidate, index = base, 1
    while candidate.exists():
        candidate = destination.with_name(f"{destination.name}.interrupted.{index}")
        index += 1
    return candidate


def fit_scope(scope: Scope, *, igor: str, out_dir: str, iterations: int, igor_threads: int, keep_temp: bool,
              resume: bool, restart_incomplete: bool = False) -> Path:
    """Run one scope in an independent persistent IGoR working directory."""
    destination = Path(out_dir) / scope.name
    if destination.exists():
        # Native final files, rather than a wrapper manifest, establish that IGoR
        # finished. This prevents an interruption after inference from needlessly
        # repeating a completed donor on recovery.
        if resume and _native_igor_complete(destination):
            return destination
        if resume and restart_incomplete:
            # IGoR has no supported continuation interface for a partially written
            # fit. Preserve its alignment and EM artefacts for audit, then rerun
            # only this unfinished donor in a new directory with the original name.
            shutil.move(str(destination), str(_interrupted_destination(destination)))
        elif resume:
            raise FileExistsError(f"{destination} is incomplete; rerun with --restart-incomplete to archive and refit only this donor")
        else:
            raise FileExistsError(f"{destination} already exists; choose a new output directory")
    destination.mkdir(parents=True)
    # The temporary CSV stays separate from persistent IGoR outputs and can be removed
    # without deleting alignments, inference iterations, or the final model files.
    temporary = Path(tempfile.mkdtemp(prefix=f"{scope.name}_", dir=destination))
    input_csv = temporary / "sequences.csv"
    try:
        # This is the only file passed directly to IGoR: an index and a CDR3nt sequence.
        pd.DataFrame({"seq_index": range(len(scope.sequences)), "sequence": scope.sequences}).to_csv(input_csv, sep=";", index=False)
        commands = igor_commands(igor, destination, "model", input_csv, iterations=iterations, threads=igor_threads)
        for command in commands:
            subprocess.run(command, check=True)
        manifest = {"scope": scope.name, "scope_kind": scope.kind, "source_files": list(scope.source_files),
                    "n_unique_nonproductive_cdr3nt_per_source": dict(scope.source_sequence_counts),
                    "n_sequences_submitted_to_igor": len(scope.sequences), "igor_version": _igor_version(igor),
                    "commands": commands, "temporary_input_retained": keep_temp}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        # The input is also removed after a failed run unless debugging explicitly keeps it.
        if not keep_temp:
            shutil.rmtree(temporary, ignore_errors=True)
    return destination


def fit_scopes(scopes: list[Scope], *, igor: str, out_dir: str | Path, iterations: int, workers: int,
               igor_threads: int, keep_temp: bool, resume: bool, restart_incomplete: bool = False) -> list[Path]:
    """Fit independent scopes concurrently without oversubscribing IGoR OpenMP threads."""
    available = os.cpu_count() or 1
    if workers < 1 or igor_threads < 1:
        raise ValueError("workers and igor_threads must be positive")
    if workers * igor_threads > available:
        # Otherwise, concurrent IGoR processes would request more OpenMP threads than
        # the host has CPUs, which usually slows rather than accelerates inference.
        raise ValueError(f"workers * igor_threads exceeds available CPUs ({workers} * {igor_threads} > {available})")
    kwargs = {"igor": igor, "out_dir": str(out_dir), "iterations": iterations,
              "igor_threads": igor_threads, "keep_temp": keep_temp, "resume": resume,
              "restart_incomplete": restart_incomplete}
    if len(scopes) == 1:
        return [fit_scope(scopes[0], **kwargs)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Individual tasks write to separate directories and can therefore run in parallel.
        futures = [pool.submit(fit_scope, scope, **kwargs) for scope in scopes]
        return [future.result() for future in futures]


def main() -> None:
    # The command line defines the data source, fitting mode, and resource allocation.
    parser = argparse.ArgumentParser(description="Fit individual or group IGoR models from MiXCR TSV or canonical parquet.")
    parser.add_argument("--input", required=True, help="one .tsv/.parquet file or a directory of such files")
    parser.add_argument("--mode", required=True, choices=("individual", "patient", "group"))
    parser.add_argument("--patient-pattern", help="regular expression with exactly one capture group for the donor identifier; required for --mode patient")
    parser.add_argument("--out", required=True, help="new directory for persistent IGoR outputs")
    parser.add_argument("--igor", required=True, help="path to the IGoR executable")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="concurrent independent fits")
    parser.add_argument("--igor-threads", type=int, default=1, help="OpenMP threads per IGoR fit")
    parser.add_argument("--keep-temp", action="store_true", help="retain transient IGoR CSV inputs for debugging")
    parser.add_argument("--resume", action="store_true", help="skip scopes with native final IGoR output, even if an interruption prevented manifest writing")
    parser.add_argument("--restart-incomplete", action="store_true", help="with --resume, move each incomplete scope to a recoverable .interrupted directory and refit only that donor")
    args = parser.parse_args()
    scopes = build_scopes(args.input, args.mode, patient_pattern=args.patient_pattern)
    if args.restart_incomplete and not args.resume:
        parser.error("--restart-incomplete requires --resume")
    outputs = fit_scopes(scopes, igor=args.igor, out_dir=args.out, iterations=args.iterations,
                         workers=args.workers, igor_threads=args.igor_threads, keep_temp=args.keep_temp,
                         resume=args.resume, restart_incomplete=args.restart_incomplete)
    print("\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
