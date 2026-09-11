"""Reproduce the Emerson et al. CMV exact-clonotype classifier from source TSVs.

This module deliberately does not use embeddings, repertoire clouds, clone
abundances, or any learned representation.  It reproduces the published
two-stage statistical procedure on the native Emerson files:

1. keep productive TCR beta rearrangements (``frame_type == 'In'``) and use
   the exact ``(V gene, CDR3 amino-acid sequence, J gene)`` identity;
2. discover CMV-enriched public identities in P donors with one-sided Fisher's
   exact test, a donor-incidence threshold of two, and ``p < 1e-4``;
3. fit class-specific beta-binomial distributions to the number of discovered
   identities observed out of all distinct productive identities; and
4. evaluate the frozen P-derived model on Keck donors.

The temporary output may contain large diagnostics.  The compact tables,
metrics, and ROC figure intended for inspection are written separately under
the requested results directory by the caller.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit
from scipy.stats import hypergeom
from sklearn.metrics import auc, roc_auc_score, roc_curve


CMV_TAG = re.compile(r"(?:^|,)Virus Diseases:Cytomegalovirus ([+-])(?:,|$)")
KEY_SEPARATOR = "\x1f"
_TSV_COLUMNS = ["frame_type", "amino_acid", "v_gene", "j_gene"]


def _sample_status(path: Path) -> str:
    """Extract one CMV call from the first source TSV row."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        record = next(csv.DictReader(handle, delimiter="\t"), None)
    if record is None:
        raise ValueError(f"{path}: no data row")
    tags = ",".join(str(record.get(column) or "") for column in ("sample_catalog_tags", "sample_rich_tags", "sample_tags"))
    calls = set(CMV_TAG.findall(tags))
    if len(calls) > 1:
        raise ValueError(f"{path}: conflicting CMV calls")
    return {"+": "positive", "-": "negative"}.get(next(iter(calls), None), "missing")


def _metadata(source_dir: Path, pattern: str) -> pd.DataFrame:
    """Return source-file CMV calls without depending on external metadata files."""
    rows = [{"sample": path.stem, "cmv_status": _sample_status(path)} for path in sorted(source_dir.glob(pattern))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"{source_dir}: no files matching {pattern}")
    return frame


def _key(v_gene: object, cdr3aa: object, j_gene: object) -> str | None:
    """Construct the published exact TCR beta identity without gene harmonization."""
    v_text, cdr3_text, j_text = str(v_gene).strip(), str(cdr3aa).strip().upper(), str(j_gene).strip()
    if not v_text or not cdr3_text or not j_text or cdr3_text.lower() == "nan":
        return None
    return KEY_SEPARATOR.join((v_text, cdr3_text, j_text))


def _productive_keys(path: Path, *, chunksize: int = 100_000, retries: int = 5) -> set[str]:
    """Read one native repertoire as a set of distinct productive identities."""
    failure: OSError | None = None
    for attempt in range(retries):
        try:
            keys: set[str] = set()
            for chunk in pd.read_csv(path, sep="\t", usecols=_TSV_COLUMNS, chunksize=chunksize, low_memory=False):
                productive = chunk.loc[chunk["frame_type"].astype(str).str.strip().str.lower().eq("in")]
                keys.update(_key(v, c, j) for v, c, j in productive[["v_gene", "amino_acid", "j_gene"]].itertuples(index=False, name=None))
            keys.discard(None)
            return keys
        except OSError as exc:
            failure = exc
            time.sleep(3 * (attempt + 1))
    assert failure is not None
    raise failure


def _hash_positions(key: str, bloom_bits: int) -> tuple[int, int]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest[:8], "little") % bloom_bits, int.from_bytes(digest[8:], "little") % bloom_bits


def _is_set(bits: bytearray, position: int) -> bool:
    return bool(bits[position >> 3] & (1 << (position & 7)))


def _set(bits: bytearray, position: int) -> None:
    bits[position >> 3] |= 1 << (position & 7)


def _public_gate(paths: Iterable[Path], *, bloom_bits: int) -> set[str]:
    """Keep only identities observed in at least two donors, using bounded memory."""
    if bloom_bits < 8:
        raise ValueError("bloom_bits must be at least 8")
    bits = bytearray((bloom_bits + 7) // 8)
    candidates: set[str] = set()
    for index, path in enumerate(paths, start=1):
        for key in _productive_keys(path):
            first, second = _hash_positions(key, bloom_bits)
            if _is_set(bits, first) and _is_set(bits, second):
                candidates.add(key)
            _set(bits, first)
            _set(bits, second)
        if index % 25 == 0:
            print(f"public gate: {index} repertoires, {len(candidates)} provisional identities", flush=True)
    return candidates


def _count_candidates(paths: Iterable[Path], labels: dict[str, int], candidates: set[str]) -> tuple[dict[str, list[int]], dict[str, int], dict[str, tuple[int, int]]]:
    """Obtain exact donor incidences and per-donor total/diagnostic counts."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    class_sizes = {0: 0, 1: 0}
    donor_counts: dict[str, tuple[int, int]] = {}
    for index, path in enumerate(paths, start=1):
        label = labels[path.stem]
        keys = _productive_keys(path)
        class_sizes[label] += 1
        for key in keys & candidates:
            counts[key][label] += 1
        donor_counts[path.stem] = (len(keys), 0)
        if index % 25 == 0:
            print(f"exact incidence: {index} repertoires", flush=True)
    return counts, class_sizes, donor_counts


def _fisher_statistics(counts: dict[str, list[int]], class_sizes: dict[str, int] | dict[int, int], *, p_threshold: float) -> pd.DataFrame:
    """Calculate one-sided Fisher probabilities vectorially for public identities."""
    keys = np.array(list(counts), dtype=object)
    if not len(keys):
        raise ValueError("no public clonotypes remained after exact counting")
    positives = np.fromiter((counts[key][1] for key in keys), dtype=np.int32, count=len(keys))
    negatives = np.fromiter((counts[key][0] for key in keys), dtype=np.int32, count=len(keys))
    totals = positives + negatives
    keep = totals >= 2
    keys, positives, negatives, totals = keys[keep], positives[keep], negatives[keep], totals[keep]
    n_positive, n_negative = int(class_sizes[1]), int(class_sizes[0])
    # The one-sided Fisher exact probability is the hypergeometric survival
    # probability for at least the observed number of positive carriers.
    p_values = hypergeom.sf(positives - 1, n_positive + n_negative, n_positive, totals)
    selected = (p_values < p_threshold) & (positives / n_positive > negatives / n_negative)
    rows = []
    for key, positive, negative, total, p_value, is_selected in zip(keys[selected], positives[selected], negatives[selected], totals[selected], p_values[selected], selected[selected], strict=True):
        v_gene, cdr3aa, j_gene = key.split(KEY_SEPARATOR, 2)
        rows.append({"clonotype_key": key, "v_gene": v_gene, "cdr3aa": cdr3aa, "j_gene": j_gene, "p_positive_carriers": int(positive), "p_negative_carriers": int(negative), "p_total_carriers": int(total), "p_positive_prevalence": float(positive / n_positive), "p_negative_prevalence": float(negative / n_negative), "p_fisher_one_sided": float(p_value), "selected": bool(is_selected)})
    return pd.DataFrame(rows).sort_values(["p_fisher_one_sided", "p_positive_carriers", "clonotype_key"], ascending=[True, False, True], ignore_index=True)


def _neg_log_likelihood(params: np.ndarray, n_values: np.ndarray, k_values: np.ndarray) -> float:
    """Negative beta-binomial log likelihood up to the class-invariant term."""
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return float("inf")
    return float(-(np.sum(betaln(k_values + alpha, n_values - k_values + beta)) - len(n_values) * betaln(alpha, beta)))


def _fit_beta_binomial(n_values: np.ndarray, k_values: np.ndarray) -> dict[str, float]:
    """Fit positive beta-binomial shape parameters for one CMV class."""
    result = minimize(_neg_log_likelihood, x0=np.array([1.0, 1000.0]), args=(n_values, k_values), method="L-BFGS-B", bounds=((1e-5, None), (1e-5, None)))
    if not result.success:
        raise RuntimeError(f"beta-binomial optimisation failed: {result.message}")
    return {"alpha": float(result.x[0]), "beta": float(result.x[1]), "n_donors": int(len(n_values))}


def _posterior_probability(n_values: np.ndarray, k_values: np.ndarray, negative: dict[str, float], positive: dict[str, float]) -> np.ndarray:
    """Return posterior CMV-positive probabilities with Laplace-smoothed priors."""
    total = negative["n_donors"] + positive["n_donors"]
    log_negative = betaln(k_values + negative["alpha"], n_values - k_values + negative["beta"]) - betaln(negative["alpha"], negative["beta"]) + np.log(negative["n_donors"] + 1) - np.log(total + 2)
    log_positive = betaln(k_values + positive["alpha"], n_values - k_values + positive["beta"]) - betaln(positive["alpha"], positive["beta"]) + np.log(positive["n_donors"] + 1) - np.log(total + 2)
    return expit(log_positive - log_negative)


def _signature_counts(paths: Iterable[Path], labels: dict[str, int], diagnostic: set[str]) -> pd.DataFrame:
    """Evaluate a frozen diagnostic set on source repertoires."""
    rows = []
    for index, path in enumerate(paths, start=1):
        keys = _productive_keys(path)
        rows.append({"sample": path.stem, "cmv_label": labels[path.stem], "n_unique_productive": len(keys), "k_diagnostic_present": len(keys & diagnostic)})
        if index % 25 == 0:
            print(f"signature counts: {index} repertoires", flush=True)
    return pd.DataFrame(rows)


def _roc_figure(frame: pd.DataFrame, path: Path, title: str) -> float:
    """Draw an ROC curve and return AUROC for one labelled cohort."""
    y = frame["cmv_label"].to_numpy(dtype=int)
    score = frame["cmv_probability"].to_numpy(dtype=float)
    value = float(roc_auc_score(y, score))
    fpr, tpr, _ = roc_curve(y, score)
    fig, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.plot(fpr, tpr, linewidth=2, label=f"AUROC = {value:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    axis.legend(loc="lower right", frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return value


def run_exact_reproduction(source_dir: str | Path, tmp_dir: str | Path, results_dir: str | Path, *, p_threshold: float = 1e-4, bloom_bits: int = 1 << 33) -> dict[str, object]:
    """Derive the P classifier and evaluate its frozen parameters on Keck."""
    source, temporary, results = Path(source_dir), Path(tmp_dir), Path(results_dir)
    if temporary.exists() or results.exists():
        raise FileExistsError("temporary and results directories must be new to prevent accidental mixing of runs")
    if not 0 < p_threshold < 1:
        raise ValueError("p_threshold must be between zero and one")
    p_meta, keck_meta = _metadata(source, "P*.tsv"), _metadata(source, "Keck*.tsv")
    p_labels = {row.sample: int(row.cmv_status == "positive") for row in p_meta.itertuples() if row.cmv_status in {"positive", "negative"}}
    keck_labels = {row.sample: int(row.cmv_status == "positive") for row in keck_meta.itertuples() if row.cmv_status in {"positive", "negative"}}
    p_paths = [path for path in sorted(source.glob("P*.tsv")) if path.stem in p_labels]
    keck_paths = [path for path in sorted(source.glob("Keck*.tsv")) if path.stem in keck_labels]
    if len(p_paths) != len(p_labels) or len(keck_paths) != len(keck_labels):
        raise RuntimeError("metadata/file mismatch")
    temporary.mkdir(parents=True)
    results.mkdir(parents=True)
    public = _public_gate(p_paths, bloom_bits=bloom_bits)
    counts, class_sizes, _ = _count_candidates(p_paths, p_labels, public)
    selected = _fisher_statistics(counts, class_sizes, p_threshold=p_threshold)
    diagnostic = set(selected["clonotype_key"])
    if not diagnostic:
        raise RuntimeError("no diagnostic clonotypes passed the published threshold")
    p_counts = _signature_counts(p_paths, p_labels, diagnostic)
    negative = _fit_beta_binomial(p_counts.loc[p_counts["cmv_label"].eq(0), "n_unique_productive"].to_numpy(float), p_counts.loc[p_counts["cmv_label"].eq(0), "k_diagnostic_present"].to_numpy(float))
    positive = _fit_beta_binomial(p_counts.loc[p_counts["cmv_label"].eq(1), "n_unique_productive"].to_numpy(float), p_counts.loc[p_counts["cmv_label"].eq(1), "k_diagnostic_present"].to_numpy(float))
    for frame in (p_counts,):
        frame["cmv_probability"] = _posterior_probability(frame["n_unique_productive"].to_numpy(float), frame["k_diagnostic_present"].to_numpy(float), negative, positive)
    keck_counts = _signature_counts(keck_paths, keck_labels, diagnostic)
    keck_counts["cmv_probability"] = _posterior_probability(keck_counts["n_unique_productive"].to_numpy(float), keck_counts["k_diagnostic_present"].to_numpy(float), negative, positive)
    p_auc = _roc_figure(p_counts, results / "emerson_exact_p_apparent_roc.png", "Emerson exact classifier: P (apparent)")
    keck_auc = _roc_figure(keck_counts, results / "emerson_exact_keck_roc.png", "Emerson exact classifier: Keck external validation")
    selected.to_csv(results / "emerson_exact_diagnostic_clonotypes.tsv", sep="\t", index=False)
    p_counts.to_csv(results / "emerson_exact_p_scores.tsv", sep="\t", index=False)
    keck_counts.to_csv(results / "emerson_exact_keck_scores.tsv", sep="\t", index=False)
    manifest = {"source_dir": str(source), "identity": "raw_v_gene + raw_amino_acid + raw_j_gene", "productive_filter": "frame_type == In", "selection_test": "one-sided Fisher exact (CMV-positive enrichment)", "minimum_subject_incidence": 2, "p_threshold": p_threshold, "selection_multiple_testing_adjustment": "none (published threshold)", "scoring_model": "class-specific beta-binomial posterior with Laplace-smoothed class prior", "n_p_positive": class_sizes[1], "n_p_negative": class_sizes[0], "n_keck_positive": int(sum(keck_labels.values())), "n_keck_negative": int(len(keck_labels) - sum(keck_labels.values())), "n_public_tested": len(counts), "n_diagnostic": len(diagnostic), "p_apparent_auroc": p_auc, "keck_external_auroc": keck_auc, "beta_binomial_negative": negative, "beta_binomial_positive": positive}
    (results / "emerson_exact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (temporary / "emerson_exact_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emerson-tsv", required=True, help="directory containing native P*.tsv and Keck*.tsv files")
    parser.add_argument("--tmp", required=True, help="new rebuildable temporary output directory")
    parser.add_argument("--results", required=True, help="new compact human-facing results directory")
    parser.add_argument("--p-threshold", type=float, default=1e-4, help="one-sided Fisher threshold from the published procedure")
    parser.add_argument("--bloom-bits", type=int, default=1 << 33, help="Bloom-filter size in bits for the first source pass")
    args = parser.parse_args(argv)
    run_exact_reproduction(args.emerson_tsv, args.tmp, args.results, p_threshold=args.p_threshold, bloom_bits=args.bloom_bits)


if __name__ == "__main__":
    main()
