"""Discover exact CMV-associated TCR beta clonotypes in Emerson P and test Keck.

The discovery unit is an exact productive clonotype identity: observed V gene,
CDR3 amino-acid sequence, and observed J gene.  P samples are used only for
discovery.  Keck samples are read only after the P-derived candidate list is
frozen.  Prototype coordinates are deliberately absent from candidate
selection; a later annotation step may use them to interpret the frozen list.

The P cloud files contain only productive clonotypes and are considerably
smaller than the source TSV collection.  A two-pass Bloom-filter gate avoids
retaining the large set of private clonotypes in memory: it admits only
identities observed at least twice, followed by an exact second-pass count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score


CMV_TAG = re.compile(r"(?:^|,)Virus Diseases:Cytomegalovirus ([+-])(?:,|$)")
KEY_SEPARATOR = "\x1f"
_CLOUD_COLUMNS = ["sample", "v_gene", "j_gene", "cdr3aa"]


def _sample_status(path: Path) -> str:
    """Read the CMV status from the first record of one Emerson TSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        record = next(csv.DictReader(handle, delimiter="\t"), None)
    if record is None:
        raise ValueError(f"{path}: no data row")
    tags = ",".join(str(record.get(column) or "") for column in ("sample_catalog_tags", "sample_rich_tags", "sample_tags"))
    calls = set(CMV_TAG.findall(tags))
    if len(calls) > 1:
        raise ValueError(f"{path}: conflicting CMV calls")
    return {"+": "positive", "-": "negative"}.get(next(iter(calls), None), "missing")


def _status_table(source_dir: Path, pattern: str) -> pd.DataFrame:
    """Return one CMV call per matching source sample."""
    rows = [{"sample": path.stem, "cmv_status": _sample_status(path)} for path in sorted(source_dir.glob(pattern))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"{source_dir}: no {pattern} files")
    if frame["sample"].duplicated().any():
        raise ValueError(f"{source_dir}: duplicate sample identifiers for {pattern}")
    return frame


def _key(v_gene: str, cdr3aa: str, j_gene: str) -> str | None:
    """Construct a non-empty exact clonotype key with canonical whitespace/case."""
    v_gene, cdr3aa, j_gene = str(v_gene).strip(), str(cdr3aa).strip().upper(), str(j_gene).strip()
    if not v_gene or not cdr3aa or not j_gene or cdr3aa.lower() == "nan":
        return None
    return KEY_SEPARATOR.join((v_gene, cdr3aa, j_gene))


def _keys_from_cloud(path: Path, *, retries: int = 5) -> tuple[str, set[str]]:
    """Read one productive cloud, retrying transient filesystem failures."""
    failure: OSError | None = None
    for attempt in range(retries):
        try:
            frame = pd.read_parquet(path, columns=_CLOUD_COLUMNS)
            break
        except OSError as exc:
            failure = exc
            time.sleep(3 * (attempt + 1))
    else:
        assert failure is not None
        raise failure
    samples = frame["sample"].astype(str).unique()
    if len(samples) != 1:
        raise ValueError(f"{path}: expected one sample, found {samples.tolist()}")
    keys = {_key(v, c, j) for v, c, j in frame[["v_gene", "cdr3aa", "j_gene"]].itertuples(index=False, name=None)}
    keys.discard(None)
    return str(samples[0]), keys


def _all_keys_from_tsv(path: Path, *, chunksize: int = 100_000, retries: int = 5) -> set[str]:
    """Read all productive identities from a source TSV, used only for missing clouds."""
    usecols = ["frame_type", "amino_acid", "v_gene", "j_gene"]
    failure: OSError | None = None
    for attempt in range(retries):
        try:
            keys: set[str] = set()
            for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize, low_memory=False):
                productive = chunk.loc[chunk["frame_type"].astype(str).str.strip().str.lower().eq("in")]
                keys.update(_key(v, c, j) for v, c, j in productive[["v_gene", "amino_acid", "j_gene"]].itertuples(index=False, name=None))
            keys.discard(None)
            return keys
        except OSError as exc:
            failure = exc
            time.sleep(3 * (attempt + 1))
    assert failure is not None
    raise failure


def _p_records(cloud_dir: Path, source_dir: Path, labels: dict[str, str]) -> list[tuple[str, Path, str]]:
    """Use clouds where available and source TSV fallback only for missing P clouds."""
    clouds = {path.stem: path for path in cloud_dir.glob("P*.parquet")}
    records = [(sample, path, "cloud") for sample, path in clouds.items() if sample in labels]
    source_samples = {path.stem: path for path in source_dir.glob("P*.tsv")}
    missing = sorted(set(labels) - set(clouds))
    absent = sorted(set(missing) - set(source_samples))
    if absent:
        raise FileNotFoundError(f"P samples missing both cloud and source TSV: {absent[:5]}")
    records.extend((sample, source_samples[sample], "source_fallback") for sample in missing)
    return sorted(records)


def _record_keys(sample: str, path: Path, kind: str) -> set[str]:
    """Read identities from one P cloud or the corresponding source fallback."""
    if kind == "cloud":
        observed_sample, keys = _keys_from_cloud(path)
        if observed_sample != sample:
            raise ValueError(f"{path}: expected sample {sample}, found {observed_sample}")
        return keys
    if kind == "source_fallback":
        return _all_keys_from_tsv(path)
    raise ValueError(f"unknown P record type {kind!r}")


def _hash_positions(key: str, bloom_bits: int) -> tuple[int, int]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    first = int.from_bytes(digest[:8], "little") % bloom_bits
    second = int.from_bytes(digest[8:], "little") % bloom_bits
    return first, second


def _is_set(bits: bytearray, position: int) -> bool:
    return bool(bits[position >> 3] & (1 << (position & 7)))


def _set(bits: bytearray, position: int) -> None:
    bits[position >> 3] |= 1 << (position & 7)


def _candidate_gate(records: Iterable[tuple[str, Path, str]], *, bloom_bits: int) -> set[str]:
    """Return identities seen at least twice, with a negligible false-positive gate."""
    if bloom_bits < 8:
        raise ValueError("bloom_bits must be at least 8")
    bits = bytearray((bloom_bits + 7) // 8)
    candidates: set[str] = set()
    for index, (sample, path, kind) in enumerate(records, start=1):
        keys = _record_keys(sample, path, kind)
        for key in keys:
            first, second = _hash_positions(key, bloom_bits)
            if _is_set(bits, first) and _is_set(bits, second):
                candidates.add(key)
            _set(bits, first)
            _set(bits, second)
        if index % 25 == 0:
            print(f"candidate gate: {index} clouds, {len(candidates)} provisional identities", flush=True)
    return candidates


def _exact_p_counts(records: Iterable[tuple[str, Path, str]], labels: dict[str, str], candidates: set[str]) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Count donor-level P carriers for each gated identity exactly."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    donor_counts = {"positive": 0, "negative": 0}
    for index, (sample, path, kind) in enumerate(records, start=1):
        keys = _record_keys(sample, path, kind)
        status = labels.get(sample)
        if status not in donor_counts:
            continue
        donor_counts[status] += 1
        for key in keys & candidates:
            counts[key][0 if status == "positive" else 1] += 1
        if index % 25 == 0:
            print(f"exact count: {index} clouds", flush=True)
    return counts, donor_counts


def _bh(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for a one-dimensional finite array."""
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _split_key(key: str) -> tuple[str, str, str]:
    return tuple(key.split(KEY_SEPARATOR, 2))  # type: ignore[return-value]


def _p_statistics(counts: dict[str, list[int]], donor_counts: dict[str, int], minimum_donors: int) -> pd.DataFrame:
    """Compute exact P association statistics after outcome-blind donor filter."""
    n_positive, n_negative = donor_counts["positive"], donor_counts["negative"]
    rows = []
    for key, (positive, negative) in counts.items():
        total = positive + negative
        if total < minimum_donors:
            continue
        table = [[positive, n_positive - positive], [negative, n_negative - negative]]
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        v_gene, cdr3aa, j_gene = _split_key(key)
        rows.append({"clonotype_key": key, "v_gene": v_gene, "cdr3aa": cdr3aa, "j_gene": j_gene, "p_positive_carriers": positive, "p_negative_carriers": negative, "p_total_carriers": total, "p_positive_prevalence": positive / n_positive, "p_negative_prevalence": negative / n_negative, "p_risk_difference": positive / n_positive - negative / n_negative, "p_odds_ratio": float(odds_ratio), "p_fisher_p": float(p_value)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no clonotypes passed the outcome-blind donor-occurrence filter")
    frame["p_bh_fdr_q"] = _bh(frame["p_fisher_p"].to_numpy(dtype=float))
    return frame.sort_values(["p_bh_fdr_q", "p_fisher_p", "p_total_carriers", "clonotype_key"], ascending=[True, True, False, True]).reset_index(drop=True)


def _keys_from_tsv(path: Path, candidate_keys: set[str], *, chunksize: int = 100_000, retries: int = 5) -> set[str]:
    """Read productive source rows and retain only membership in a fixed candidate set."""
    usecols = ["frame_type", "amino_acid", "v_gene", "j_gene"]
    failure: OSError | None = None
    for attempt in range(retries):
        try:
            found: set[str] = set()
            for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize, low_memory=False):
                productive = chunk.loc[chunk["frame_type"].astype(str).str.strip().str.lower().eq("in")]
                for v_gene, cdr3aa, j_gene in productive[["v_gene", "amino_acid", "j_gene"]].itertuples(index=False, name=None):
                    key = _key(v_gene, cdr3aa, j_gene)
                    if key in candidate_keys:
                        found.add(key)
            return found
        except OSError as exc:
            failure = exc
            time.sleep(3 * (attempt + 1))
    assert failure is not None
    raise failure


def _keck_statistics(source_dir: Path, candidate_keys: set[str], metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fixed P candidate list to Keck and return clone- and donor-level records."""
    labels = dict(zip(metadata["sample"], metadata["cmv_status"]))
    carrier_rows, donor_rows = [], []
    for index, path in enumerate(sorted(source_dir.glob("Keck*.tsv")), start=1):
        sample = path.stem
        status = labels.get(sample, "missing")
        if status in ("positive", "negative"):
            found = _keys_from_tsv(path, candidate_keys)
            donor_rows.append({"sample": sample, "cmv_status": status, "n_candidates_present": len(found)})
            carrier_rows.extend({"sample": sample, "cmv_status": status, "clonotype_key": key} for key in found)
        if index % 10 == 0:
            print(f"Keck scan: {index} files", flush=True)
    donors = pd.DataFrame(donor_rows)
    carriers = pd.DataFrame(carrier_rows)
    if donors.empty:
        raise ValueError("no Keck donors with known CMV status")
    clone_rows = []
    for key in sorted(candidate_keys):
        subset = carriers.loc[carriers["clonotype_key"].eq(key)] if not carriers.empty else carriers
        positive = int(subset["cmv_status"].eq("positive").sum()) if not subset.empty else 0
        negative = int(subset["cmv_status"].eq("negative").sum()) if not subset.empty else 0
        clone_rows.append({"clonotype_key": key, "keck_positive_carriers": positive, "keck_negative_carriers": negative, "keck_positive_prevalence": positive / int(donors["cmv_status"].eq("positive").sum()), "keck_negative_prevalence": negative / int(donors["cmv_status"].eq("negative").sum())})
    return pd.DataFrame(clone_rows, columns=["clonotype_key", "keck_positive_carriers", "keck_negative_carriers", "keck_positive_prevalence", "keck_negative_prevalence"]), donors


def run_discovery(cloud_dir: str | Path, source_dir: str | Path, out_dir: str | Path, *, minimum_donors: int = 2, fdr_threshold: float = 0.05, bloom_bits: int = 1 << 33) -> dict[str, object]:
    """Discover in P and externally evaluate the frozen list in Keck."""
    if minimum_donors < 2:
        raise ValueError("minimum_donors must be at least 2")
    if not 0 < fdr_threshold <= 1:
        raise ValueError("fdr_threshold must be in (0, 1]")
    clouds, source, output = Path(cloud_dir), Path(source_dir), Path(out_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    p_metadata = _status_table(source, "P*.tsv")
    keck_metadata = _status_table(source, "Keck*.tsv")
    p_labels = dict(zip(p_metadata.loc[p_metadata["cmv_status"].isin(["positive", "negative"]), "sample"], p_metadata.loc[p_metadata["cmv_status"].isin(["positive", "negative"]), "cmv_status"]))
    cloud_paths = sorted(clouds.glob("P*.parquet"))
    if not cloud_paths:
        raise FileNotFoundError(f"{clouds}: no P*.parquet clouds")
    records = _p_records(clouds, source, p_labels)
    output.mkdir(parents=True)
    candidates = _candidate_gate(records, bloom_bits=bloom_bits)
    counts, donor_counts = _exact_p_counts(records, p_labels, candidates)
    p_stats = _p_statistics(counts, donor_counts, minimum_donors)
    selected = p_stats.loc[(p_stats["p_bh_fdr_q"] <= fdr_threshold) & (p_stats["p_risk_difference"] > 0)].copy()
    frozen_keys = set(selected["clonotype_key"])
    keck_stats, keck_donors = _keck_statistics(source, frozen_keys, keck_metadata)
    selected = selected.merge(keck_stats, on="clonotype_key", how="left", validate="one_to_one")
    if len(keck_donors["cmv_status"].unique()) == 2 and len(frozen_keys):
        y = keck_donors["cmv_status"].eq("positive").to_numpy(dtype=int)
        signature_auc = float(roc_auc_score(y, keck_donors["n_candidates_present"].to_numpy(dtype=float)))
    else:
        signature_auc = float("nan")
    p_stats.to_parquet(output / "p_public_clonotype_statistics.parquet", index=False)
    p_metadata.to_parquet(output / "p_metadata.parquet", index=False)
    keck_metadata.to_parquet(output / "keck_metadata.parquet", index=False)
    keck_donors.to_parquet(output / "keck_donor_signature.parquet", index=False)
    selected.to_csv(output / "cmv_candidate_clonotypes.tsv", sep="\t", index=False)
    manifest = {"cloud_dir": str(clouds), "source_dir": str(source), "minimum_donors": minimum_donors, "fdr_threshold": fdr_threshold, "bloom_bits": bloom_bits, "n_p_clouds": len(cloud_paths), "n_p_source_fallbacks": sum(kind == "source_fallback" for _, _, kind in records), "n_p_cmv_positive": donor_counts["positive"], "n_p_cmv_negative": donor_counts["negative"], "n_p_public_tested": len(p_stats), "n_selected_positive": len(selected), "n_keck_cmv_positive": int(keck_donors["cmv_status"].eq("positive").sum()), "n_keck_cmv_negative": int(keck_donors["cmv_status"].eq("negative").sum()), "keck_signature_auroc": signature_auc}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-clouds", required=True, help="raw productive P cloud directory")
    parser.add_argument("--emerson-tsv", required=True, help="Emerson P and Keck TSV directory")
    parser.add_argument("--out", required=True, help="new temporary output directory")
    parser.add_argument("--minimum-donors", type=int, default=2, help="outcome-blind P carrier threshold")
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--bloom-bits", type=int, default=1 << 33, help="Bloom-filter size in bits")
    args = parser.parse_args(argv)
    run_discovery(args.p_clouds, args.emerson_tsv, args.out, minimum_donors=args.minimum_donors, fdr_threshold=args.fdr_threshold, bloom_bits=args.bloom_bits)


if __name__ == "__main__":
    main()
