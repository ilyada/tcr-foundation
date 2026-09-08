"""
diagnostics.py -- measurements that decide whether a construction is sound, run BEFORE it is trained.

`frame_leak_report` is the first: it answers, in nats, how much of the difference between a donor's
productive and non-productive rearrangements is selection and how much is the frame test itself.

THE QUESTION
------------
The design fits two densities over recombination events, `P_pre` on the non-productive rearrangements and
`P_post` on the productive ones, and reads selection off their difference. But "productive" means the
coordinates sum to a multiple of three, so `P_post` is fitted on a CONDITIONED sample, and part of the gap
between the two could be that conditioning rather than biology.

Theory says the contamination is small: conditioning a sum of near-independent integer coordinates on its
residue class barely moves the individual marginals when the coordinates spread over many residues (the
perturbation decays like the second Fourier coefficient of each marginal, and deletions run 0-20, insertions
0-30). Theory is not a measurement, and the number has to be comparable with the bar it would corrupt --
0.21 nats, the donor-specific selection obtainable by pure counting.

THE DESIGN
----------
Three arms on one axis, all in nats, all summed over coordinates. Summing is not a convenience: under a
factorised model the KL of the product is the sum of the per-coordinate KLs, so this sum IS the quantity
that would enter log Q.

  main         out & residue 1   vs  out & residue 2   both out of frame, so no selection separates them;
                                                       any difference is the residue conditioning alone
  inframe_null out (both)        vs  stop              in-frame and unselected -- the direct question, but
                                                       biased: stop-carriers are longer because more
                                                       insertion means more chance of catching a stop
  frame_fixed  in                vs  stop              both at residue 0, so the arithmetic cancels and
                                                       what remains is selection -- the complement of
                                                       `main`, and the arm that says whether the headline
                                                       contrast survives once the frame is held fixed
  calibration  in                vs  out + stop        the full signal, selection plus frame, for scale

`main` and `frame_fixed` bracket the answer from opposite sides: the first fixes selection and varies the
residue, the second fixes the residue and varies selection. Neither alone is decisive -- `main` cannot see
the stop-codon half of the productivity rule, and `frame_fixed` leans on the 2% stop class, which is itself
skewed toward long insertions.

Each arm runs separately on D-bearing and D-less rows: without a D call the n1/n2 split does not exist and
only the total V-J insertion does, so the event has a different dimension and the two cannot be pooled.

v_gene and j_gene are carried as coordinates on purpose. They are not terms of the sum, so the residue
cannot move them; if they move, something other than the frame is separating the groups and the main arm
is not measuring what it claims.

THE CONTROL
-----------
A permutation null, stratified WITHIN DONOR. Unstratified shuffling would let donor composition differ
between the two groups and the statistic would then read donor variation, which is the very thing the
generator exists to model. The null gives a mean and a spread; the observed value is reported as z against
them, because a KL estimated from finite counts is positive even when the two distributions are identical.

READING IT
----------
main at the noise floor and calibration far above it -> the productive/non-productive gap is selection, the
two-density construction is clean. main comparable to 0.21 nats -> the frame leaks through the marginals
too, and the construction must change before anything is trained.

CLI
    python -m tcr_foundation.diagnostics --events DIR --out results/YYYY-MM-DD [--donors N]
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

#: Coordinates compared when a D segment was assigned, and when it was not.
COORDS_WITH_D = ("del_v", "del_d5", "del_d3", "del_j", "ins_n1", "ins_n2", "v_gene", "j_gene")
COORDS_NO_D = ("del_v", "del_j", "ins_vj", "v_gene", "j_gene")

ARMS = ("main", "inframe_null", "frame_fixed", "calibration")


def _codes(series: pd.Series) -> np.ndarray:
    """Dense non-negative integer codes for either an integer coordinate or a gene name."""
    if pd.api.types.is_numeric_dtype(series):
        v = series.to_numpy()
        return (v - v.min()).astype(np.int64)
    return pd.Categorical(series.astype(str)).codes.astype(np.int64)


def _kl_nats(a: np.ndarray, b: np.ndarray, smoothing: float) -> float:
    """KL(P_a || P_b) in nats between two count vectors over a shared support, add-alpha smoothed.

    Smoothing is required, not cosmetic: an unsmoothed zero in b makes the divergence infinite on a single
    unseen value, which for a long-tailed deletion count happens constantly."""
    pa = (a + smoothing) / (a.sum() + smoothing * len(a))
    pb = (b + smoothing) / (b.sum() + smoothing * len(b))
    return float(np.sum(pa * np.log(pa / pb)))


def _arm_kl(codes: dict, labels: np.ndarray, smoothing: float) -> dict:
    """Per-coordinate KL between the two groups defined by a boolean label array."""
    out = {}
    for name, v in codes.items():
        k = int(v.max()) + 1
        a = np.bincount(v[labels], minlength=k).astype(float)
        b = np.bincount(v[~labels], minlength=k).astype(float)
        out[name] = _kl_nats(a, b, smoothing)
    return out


def _permute_within(donor_codes: np.ndarray, labels: np.ndarray, rng) -> np.ndarray:
    """Shuffle group membership inside each donor, preserving that donor's two group sizes."""
    order = np.lexsort((rng.random(len(labels)), donor_codes))
    out = np.empty_like(labels)
    out[order] = labels[np.argsort(donor_codes, kind="stable")]
    return out


def _select(df: pd.DataFrame, arm: str):
    """(mask over df, boolean label array on the selected rows) for one arm."""
    ft = df["frame_type"].astype(str).to_numpy()
    mod = (df["cdr3_length"].to_numpy() % 3)
    if arm == "main":
        m = (ft == "out")
        return m, (mod[m] == 1)
    if arm == "inframe_null":
        m = (ft == "out") | (ft == "stop")
        return m, (ft[m] == "out")
    if arm == "frame_fixed":
        # Both groups sit at residue 0, so the arithmetic cancels by construction and what is left is
        # selection (plus the stop-codon condition itself). The complement of `main`: that one fixes
        # selection and varies the residue, this one fixes the residue and varies selection.
        m = (ft == "in") | (ft == "stop")
        return m, (ft[m] == "in")
    if arm == "calibration":
        m = np.ones(len(df), dtype=bool)
        return m, (ft == "in")
    raise ValueError(f"unknown arm {arm!r}")


def frame_leak_report(df: pd.DataFrame, n_perm: int = 200, seed: int = 0, smoothing: float = 0.5,
                      max_rows: int = 400_000) -> pd.DataFrame:
    """Run all three arms on D-bearing and D-less rows. Returns one row per (arm, d_present, coordinate)
    plus a `TOTAL` row per block, with the permutation null and z beside every value."""
    rng = np.random.default_rng(seed)
    rows = []
    for d_present, coords in ((True, COORDS_WITH_D), (False, COORDS_NO_D)):
        block = df[df["d_present"] == d_present]
        if len(block) < 1000:
            print(f"  d_present={d_present}: only {len(block)} rows, skipped")
            continue
        for arm in ARMS:
            mask, labels = _select(block, arm)
            sub = block[mask]
            if labels.sum() < 100 or (~labels).sum() < 100:
                print(f"  {arm} d_present={d_present}: group too small "
                      f"({int(labels.sum())} vs {int((~labels).sum())}), skipped")
                continue
            if len(sub) > max_rows:                    # stated, never silent
                take = rng.choice(len(sub), max_rows, replace=False)
                print(f"  {arm} d_present={d_present}: subsampled {len(sub)} -> {max_rows} rows")
                sub, labels = sub.iloc[take], labels[take]
            codes = {c: _codes(sub[c]) for c in coords if c in sub.columns}
            donor_codes = pd.Categorical(sub["donor"].astype(str)).codes.astype(np.int64)

            obs = _arm_kl(codes, labels, smoothing)
            null = {c: np.empty(n_perm) for c in obs}
            for p in range(n_perm):
                perm = _permute_within(donor_codes, labels, rng)
                for c, v in _arm_kl(codes, perm, smoothing).items():
                    null[c][p] = v

            tot_obs, tot_null = 0.0, np.zeros(n_perm)
            for c in obs:
                mu, sd = float(null[c].mean()), float(null[c].std(ddof=1))
                rows.append({"arm": arm, "d_present": d_present, "coordinate": c,
                             "kl_nats": obs[c], "null_mean": mu, "null_sd": sd,
                             "z": (obs[c] - mu) / sd if sd > 0 else np.nan,
                             "n_a": int(labels.sum()), "n_b": int((~labels).sum())})
                tot_obs += obs[c]
                tot_null += null[c]
            mu, sd = float(tot_null.mean()), float(tot_null.std(ddof=1))
            rows.append({"arm": arm, "d_present": d_present, "coordinate": "TOTAL",
                         "kl_nats": tot_obs, "null_mean": mu, "null_sd": sd,
                         "z": (tot_obs - mu) / sd if sd > 0 else np.nan,
                         "n_a": int(labels.sum()), "n_b": int((~labels).sum())})
    return pd.DataFrame(rows)


def write_report(report: pd.DataFrame, out_dir: str, tag: str = "frame_leak") -> str:
    """CSV first, figure from the CSV -- so the picture cannot drift away from the computation."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.csv")
    cols = ["arm", "d_present", "coordinate", "kl_nats", "null_mean", "null_sd", "z", "n_a", "n_b"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for _, r in report.iterrows():
            fh.write(f"{r['arm']},{r['d_present']},{r['coordinate']},{r['kl_nats']:.6f},"
                     f"{r['null_mean']:.6f},{r['null_sd']:.6f},{r['z']:.2f},{r['n_a']},{r['n_b']}\n")
    print("saved", path)
    return path


def plot_frame_leak(report: pd.DataFrame, out_dir: str, tag: str = "frame_leak",
                    counting_bar: float = 0.21) -> str:
    """Per-coordinate observed KL against its permutation floor, with the 0.21-nat counting bar for scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = [(a, d) for d in (True, False) for a in ARMS
              if len(report[(report.arm == a) & (report.d_present == d)])]
    if not blocks:
        raise ValueError("nothing to plot")
    fig, axes = plt.subplots(1, len(blocks), figsize=(4.2 * len(blocks), 4.4), squeeze=False)
    for ax, (arm, d) in zip(axes[0], blocks):
        r = report[(report.arm == arm) & (report.d_present == d) & (report.coordinate != "TOTAL")]
        x = np.arange(len(r))
        ax.bar(x - 0.2, r["kl_nats"], width=0.4, label="observed", color="#2a78d6")
        ax.bar(x + 0.2, r["null_mean"], width=0.4, yerr=r["null_sd"], label="permutation floor",
               color="#c9c8c4", ecolor="#52514e")
        ax.axhline(counting_bar, ls="--", lw=1, color="#eb6834", label=f"counting bar {counting_bar}")
        ax.set_xticks(x)
        ax.set_xticklabels(r["coordinate"], rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{arm}  (D {'present' if d else 'absent'})", fontsize=10)
        ax.set_ylabel("KL, nats" if ax is axes[0][0] else "")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.png")
    fig.savefig(path, dpi=150)
    print("saved", path)
    return path


def load_events(events_dir: str, donors: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Read per-donor event parquets. `donors` takes a seeded random subset -- the marginals of six
    one-dimensional distributions do not need the whole corpus, and reading it is the expensive part."""
    import glob
    files = sorted(glob.glob(os.path.join(events_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files in {events_dir}")
    if donors and donors < len(files):
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), donors, replace=False))]
    print(f"reading {len(files)} donor files from {events_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main():
    import argparse
    import datetime as dt
    p = argparse.ArgumentParser(description="Measure how much the frame test leaks into event marginals.")
    p.add_argument("--events", required=True, help="directory of per-donor event parquets")
    p.add_argument("--out", default=None, help="output directory (default results/<today>)")
    p.add_argument("--tag", default="frame_leak")
    p.add_argument("--donors", type=int, default=40)
    p.add_argument("--perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=400_000)
    a = p.parse_args()

    out = a.out or os.path.join("results", dt.date.today().isoformat())
    df = load_events(a.events, donors=a.donors, seed=a.seed)
    print(f"{len(df)} rows, {df['donor'].nunique()} donors")
    rep = frame_leak_report(df, n_perm=a.perm, seed=a.seed, max_rows=a.max_rows)
    write_report(rep, out, a.tag)
    plot_frame_leak(rep, out, a.tag)
    tot = rep[rep.coordinate == "TOTAL"]
    print("\n=== totals, nats ===")
    for _, r in tot.iterrows():
        print(f"  {r['arm']:<13} D{'+' if r['d_present'] else '-'}  "
              f"observed {r['kl_nats']:.4f}  floor {r['null_mean']:.4f} +- {r['null_sd']:.4f}  z {r['z']:.1f}")


if __name__ == "__main__":
    main()
