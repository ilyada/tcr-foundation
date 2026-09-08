"""
benchmark.py -- donor-centric repertoire benchmarking through the library (new code; does not touch
scripts/benchmark_ram.py). Given a set of RepertoireFeaturizers and a cohort of per-donor clonotype clouds,
fit each featurizer on the cohort, featurize every donor, and score donor-centric AUROC (leave-one-out and/or
reference-set-size sweep). Two tasks:

  donor_task(...)     one label per donor (e.g. CD4 vs CD8) -> {feature: {ref_size: auroc}}
  vaccine_delta(...)  before/after paired subjects -> AUROC of the DELTA direction (signal beyond baseline)

CLI:
  python -m tcr_foundation.benchmark --clouds <dir> --encoder ours --model <ckpt> [--task cd4cd8]

Reuses schema.read (ingestion), metrics (canonical AUROC), rep_data (cloud listing + vaccine pairing) and the
library featurizers/descriptors. Encoder-backed featurizers need a GPU; the model-free ones (V-usage, k-mer)
run on CPU.
"""
from __future__ import annotations

import os

import numpy as np

from repertoire import rep_data
from ..core import schema as S
from ..repertoire import metrics


def load_labeled_clouds(cloud_dir, label_fn, cap=2000, seed=0):
    """Read each *.parquet in cloud_dir -> canonical df (optionally capped), keep those with a non-None label.
    label_fn(stem) -> label or None. Returns (list[df], np.array[labels], list[stem])."""
    files = rep_data.cloud_files(cloud_dir)
    rng = np.random.RandomState(seed)
    dfs, labels, stems = [], [], []
    for stem, path in files.items():
        lab = label_fn(stem)
        if lab is None:
            continue
        df = S.read(path)
        if len(df) == 0:
            continue
        if len(df) > cap:
            df = df.iloc[rng.choice(len(df), cap, replace=False)].reset_index(drop=True)
        dfs.append(df); labels.append(lab); stems.append(stem)
    return dfs, np.asarray(labels), stems


def _fit_featurize(featurizers, dfs):
    """Fit each featurizer on the cohort, then featurize every df -> {name: V [n_donors, dim]}."""
    out = {}
    for name, f in featurizers.items():
        f.fit(dfs)
        out[name] = np.stack([f.featurize(d) for d in dfs])
    return out


def donor_task(featurizers, dfs, labels, ref_sizes=(1, 3, 10)):
    """One-label-per-donor donor-centric AUROC by feature. ref_size 1 == leave-one-out (max-cosine to a single
    reference). Returns {feature: {ref_size: auroc}}."""
    V = _fit_featurize(featurizers, dfs)
    return {name: metrics.ref_size_sweep_auroc(M, labels, list(ref_sizes)) for name, M in V.items()}


def vaccine_delta(featurizers, cloud_dir, meta_path, cap=2000, ref_size=10, seed=0):
    """Before/after paired-subject DELTA benchmark (signal beyond baseline): per subject, delta = after - before
    for each feature; AUROC of predicting the vaccine label from the delta direction. Returns ({feature: auroc},
    n_subjects)."""
    meta = rep_data.load_vaccine_meta(meta_path)
    paired = rep_data.paired_subjects(meta, cloud_dir)
    files = rep_data.cloud_files(cloud_dir)
    rng = np.random.RandomState(seed)

    # featurizers need a corpus fit -> pool all before/after clouds once
    stems = sorted({s for d in paired.values() for s in (d[0], d[1])})
    def _read(stem):
        df = S.read(files[stem])
        return df.iloc[rng.choice(len(df), cap, replace=False)].reset_index(drop=True) if len(df) > cap else df
    dfs_by_stem = {s: _read(s) for s in stems if s in files}
    for f in featurizers.values():
        f.fit(list(dfs_by_stem.values()))

    deltas = {name: [] for name in featurizers}
    y = []
    for subj, d in paired.items():
        if d[0] not in dfs_by_stem or d[1] not in dfs_by_stem:
            continue
        for name, f in featurizers.items():
            deltas[name].append(f.featurize(dfs_by_stem[d[1]]) - f.featurize(dfs_by_stem[d[0]]))
        y.append(d["vaccine"])
    if not y:
        return {}, 0
    y = np.asarray(y)
    return {name: metrics.ref_size_sweep_auroc(np.stack(D), y, [ref_size])[ref_size]
            for name, D in deltas.items()}, len(y)


def _build_featurizers(encoder_name, model_path, k=3, device=None):
    """Standard feature set: V-usage + CDR3 k-mer (model-free) + model mean+cov + within-V (encoder-backed)."""
    from ..repertoire import featurizers as F, descriptors as Dsc
    feats = {"V-usage": F.VUsage(), "CDR3 k-mer": F.Kmer(k=k)}
    if encoder_name and encoder_name != "none":
        from ..repertoire.encoders import NeuralEncoder, SceptrEncoder
        enc = SceptrEncoder() if encoder_name == "sceptr" else NeuralEncoder(model_path, device=device)
        feats["model mean+cov"] = Dsc.MeanCov(enc)
        feats["model within-V"] = Dsc.WithinV(enc)
    return feats


def main():
    import argparse
    ap = argparse.ArgumentParser(description="donor-centric repertoire benchmark via the library")
    ap.add_argument("--clouds", required=True, help="dir of per-donor *.parquet clouds")
    ap.add_argument("--task", default="cd4cd8", choices=["cd4cd8"])
    ap.add_argument("--encoder", default="ours", help="ours | sceptr | none (model-free only)")
    ap.add_argument("--model", default=None, help="checkpoint dir for --encoder ours")
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ref-sizes", default="1,3,10")
    args = ap.parse_args()

    feats = _build_featurizers(args.encoder, args.model, k=args.k)
    if args.task == "cd4cd8":
        dfs, y, _ = load_labeled_clouds(
            args.clouds, lambda s: 1 if "CD8" in s else (0 if "CD4" in s else None), cap=args.cap)
        print(f"CD4/CD8: {len(dfs)} donors (CD8={int(y.sum())}, CD4={int((y == 0).sum())})")
        res = donor_task(feats, dfs, y, ref_sizes=[int(x) for x in args.ref_sizes.split(",")])
        for name, sweep in res.items():
            print(f"  {name:<18} " + "  ".join(f"r{r}={a:.3f}" for r, a in sweep.items()))


if __name__ == "__main__":
    main()
