"""
build_descriptors.py — turn a folder of per-donor clonotype "clouds" into per-donor descriptor vectors.

Stage 2 of the repertoire-cloud pipeline. Input: a folder of parquet clouds from encode_repertoires.py
(one per donor: columns e0..e{H-1} + w_log + ...). Output: a parquet [donor, d0..d{D-1}] (one ~200-d
descriptor row per donor) plus the fitted whitened PCA frame (.npz) for reuse on other cohorts / new donors.

Frame policy (locked): the frame is fit ONCE on a pooled random sample of clonotype embeddings (the
model-intrinsic axes). For beta we fit it on FMBA_main and REUSE the same frame on other beta cohorts
(--frame-in) so every donor lands in one shared coordinate system. Separate frames per chain (beta/paired).

Run from scripts/ (sibling import of repertoire_cloud); pure numpy/pandas, no torch.

Usage (from scripts/):
    # fit the beta frame on FMBA_main AND write its descriptors
    python repertoire/build_descriptors.py \
        --clouds-dir ../data/processed/clouds/FMBA_main_TRB \
        --output     ../data/processed/descriptors/FMBA_main_TRB.parquet \
        --frame-out  ../data/processed/descriptors/frame_beta.npz

    # reuse that frame on the vaccine cohort
    python repertoire/build_descriptors.py \
        --clouds-dir ../data/processed/clouds/FMBA_vaccine_TRB \
        --output     ../data/processed/descriptors/FMBA_vaccine_TRB.parquet \
        --frame-in   ../data/processed/descriptors/frame_beta.npz
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import repertoire_cloud as rc   # sibling module (both in scripts/repertoire/)


def find_embedding_columns(parquet_path, prefix):
    """Return embedding column names (prefix + integer, e.g. e0..e127) sorted by index, read from the
    parquet schema without loading the data."""
    names = pq.ParquetFile(parquet_path).schema.names
    pat = re.compile(re.escape(prefix) + r"(\d+)$")
    cols = [c for c in names if pat.match(c)]
    return sorted(cols, key=lambda c: int(pat.match(c).group(1)))


def build_reference_pool(files, emb_cols, ref_sample, seed):
    """Pool a random sample of ~ref_sample clonotype embeddings across donors (roughly uniform per file).
    Reads only the embedding columns. Returns an [<=ref_sample x d] array for fitting the frame."""
    rng = np.random.default_rng(seed)
    per_file = max(1, ref_sample // max(len(files), 1))
    pool, total = [], 0
    for p in rng.permutation(files):                       # random file order
        arr = pd.read_parquet(p, columns=emb_cols).to_numpy()
        if len(arr) == 0:
            continue
        take = min(len(arr), per_file)
        idx = rng.choice(len(arr), size=take, replace=False)
        pool.append(arr[idx])
        total += take
        if total >= ref_sample:
            break
    ref = np.vstack(pool)
    if len(ref) > ref_sample:
        ref = ref[:ref_sample]
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clouds-dir", required=True, help="folder of per-donor cloud parquets")
    ap.add_argument("--output", required=True, help="output parquet [donor, d0..d{D-1}]")
    ap.add_argument("--frame-out", default=None, help="fit the frame on this cohort and save it here (.npz)")
    ap.add_argument("--frame-in", default=None, help="load a previously saved frame (.npz) and reuse it")
    ap.add_argument("--m1", type=int, default=None, help="order-1 mean axes (default: ALL axes = full mean)")
    ap.add_argument("--k", type=int, default=None, help="order-2 covariance axes (default: ALL axes = FULL covariance)")
    ap.add_argument("--ref-sample", type=int, default=500_000, help="clonotypes pooled to fit the frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-prefix", default="e", help="embedding column prefix (e -> e0..e{H-1})")
    args = ap.parse_args()

    if (args.frame_out is None) == (args.frame_in is None):
        ap.error("pass exactly one of --frame-out (fit a new frame) or --frame-in (reuse a saved one)")

    files = sorted(glob.glob(os.path.join(args.clouds_dir, "*.parquet")))
    if not files:
        ap.error(f"no parquet clouds found in {args.clouds_dir}")
    emb_cols = find_embedding_columns(files[0], args.emb_prefix)
    print(f"{len(files)} cloud files | embedding dim = {len(emb_cols)}")

    # ---- the whitened frame: fit-and-save, or load-and-reuse ----
    if args.frame_out:
        print(f"Fitting frame on a pooled sample (~{args.ref_sample:,} clonotypes) ...")
        ref = build_reference_pool(files, emb_cols, args.ref_sample, args.seed)
        frame = rc.fit_frame(ref, k_max=len(emb_cols))     # keep ALL axes (full covariance needs them)
        os.makedirs(os.path.dirname(os.path.abspath(args.frame_out)), exist_ok=True)
        np.savez(args.frame_out, mu_ref=frame["mu_ref"], V=frame["V"], eigvals=frame["eigvals"])
        print(f"  fit on {len(ref):,} embeddings -> saved frame to {args.frame_out}")
        print(f"  top eigenvalues: {np.round(frame['eigvals'][:5], 4)}")
    else:
        z = np.load(args.frame_in)
        frame = {"mu_ref": z["mu_ref"], "V": z["V"], "eigvals": z["eigvals"]}
        print(f"Loaded frame from {args.frame_in} (axes available: {frame['V'].shape[1]})")

    avail = frame["V"].shape[1]
    eff_m1 = avail if args.m1 is None else min(args.m1, avail)
    eff_k = avail if args.k is None else min(args.k, avail)
    print(f"descriptor: m1={eff_m1}, k={eff_k} -> dim {rc.descriptor_dim(eff_m1, eff_k)}"
          f"  ({'FULL covariance' if eff_k == avail else 'top-k covariance'})")

    # ---- per-donor descriptor ----
    donors, rows = [], []
    for i, p in enumerate(files, 1):
        df = pd.read_parquet(p, columns=emb_cols + ["w_log"])
        if len(df) == 0:
            print(f"[{i}/{len(files)}] {os.path.basename(p)}: empty — skipped")
            continue
        Z = df[emb_cols].to_numpy()
        w = df["w_log"].to_numpy()
        rows.append(rc.descriptor(Z, w, frame, m1=args.m1, k=args.k))
        donors.append(os.path.basename(p)[:-len(".parquet")])
        if i % 200 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] descriptors computed")

    D = len(rows[0])
    out = pd.DataFrame(np.vstack(rows), columns=[f"d{j}" for j in range(D)])
    out.insert(0, "donor", donors)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"\nDONE -> {args.output}  ({len(out)} donors x {D} features)")


if __name__ == "__main__":
    main()
