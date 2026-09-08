"""
metrics.py -- canonical donor-centric retrieval metrics (thin re-export of repertoire.rep_metrics).

ONE implementation of each metric so numbers are comparable everywhere (the library-wide rule from
rep_metrics). Exposed here so package users import from `tcr_foundation.metrics` without reaching into
`scripts/`:

  donor_centric_auroc(V, labels)                 leave-one-out macro AUROC (one number)
  ref_size_sweep_auroc(V, labels, ref_sizes, ..) macro AUROC by reference-SET size -> {r: auroc}

Both take L2-normalizable descriptor matrices V [n_donors, D] and per-donor labels.
"""
from __future__ import annotations

from repertoire.rep_metrics import donor_centric_auroc, ref_size_sweep_auroc  # canonical impls

__all__ = ["donor_centric_auroc", "ref_size_sweep_auroc"]
