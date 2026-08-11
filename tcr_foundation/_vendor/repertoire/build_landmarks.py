"""
build_landmarks.py -- fit a fixed landmark codebook (k-means) on a pool of clonotype embeddings.

Companion to build_descriptors / frame_beta: the model-intrinsic "vocabulary" of TCR-embedding neighbourhoods.
A repertoire is then described by its landmark OCCUPANCY (how much clonotype mass sits near each landmark) --
a Nystrom kernel-mean / bag-of-TCR-words descriptor that captures cloud SHAPE (which regions are populated),
unlike the order-1/2 moments. Fit ONCE on a reference clouds dir (FMBA_main, the same pool as frame_beta) and
reuse on any beta cohort (the codebook is model-intrinsic, donor-independent).

Reuses build_descriptors.build_reference_pool / find_embedding_columns and repertoire_cloud.l2_normalize.
Run from scripts/ (sibling imports); needs numpy + pandas + pyarrow + scikit-learn (no torch).

Usage (from scripts/):
    python repertoire/build_landmarks.py \
        --clouds-dir ../data/processed/clouds/FMBA_main_TRB \
        --output     ../data/processed/descriptors/landmarks_beta.npz \
        --k 256
"""

import argparse
import glob
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import repertoire_cloud as rc
from build_descriptors import find_embedding_columns, build_reference_pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clouds-dir", required=True, help="folder of per-donor cloud parquets to pool from")
    ap.add_argument("--output", required=True, help="output codebook (.npz: centroids [K x H])")
    ap.add_argument("--k", type=int, default=256, help="number of landmarks (k-means clusters)")
    ap.add_argument("--ref-sample", type=int, default=500_000, help="clonotype embeddings pooled to fit on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-prefix", default="e", help="embedding column prefix (e -> e0..e{H-1})")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.clouds_dir, "*.parquet")))
    if not files:
        ap.error(f"no parquet clouds found in {args.clouds_dir}")
    emb_cols = find_embedding_columns(files[0], args.emb_prefix)
    print(f"{len(files)} clouds | embedding dim = {len(emb_cols)} | pooling ~{args.ref_sample:,} embeddings ...")

    ref = build_reference_pool(files, emb_cols, args.ref_sample, args.seed)
    Zn = rc.l2_normalize(ref)                       # spherical geometry, matches the descriptor / frame
    print(f"fitting MiniBatchKMeans(k={args.k}) on {len(Zn):,} embeddings ...")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=10_000, n_init=3, max_iter=200)
    lab = km.fit_predict(Zn)
    C = km.cluster_centers_
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)   # L2-normalize the landmarks

    sizes = np.bincount(lab, minlength=args.k)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, centroids=C, k=args.k, seed=args.seed)
    print(f"saved {C.shape} landmarks -> {args.output}")
    print(f"cluster sizes: min {int(sizes.min())} / median {int(np.median(sizes))} / max {int(sizes.max())} "
          f"| inertia {km.inertia_:.1f}")


if __name__ == "__main__":
    main()
