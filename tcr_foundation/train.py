"""
train.py -- launch the joint TCR foundation pretraining THROUGH the library.

Usage:
    python -m tcr_foundation.train --config <cfg> [--epochs N --batch-size B --smoke --no-comet ...]
    tcr-foundation-train --config <cfg> ...        # console script, after `pip install -e .`

Phase-1 REVERSIBLE wrapper: no training logic lives here. It imports the UNMODIFIED entrypoint
`scripts/foundation/tcr_foundation_pretrain_joint.py` (reachable via the package's sys.path bootstrap in
`tcr_foundation/__init__.py`) and forwards every CLI flag straight through -- so epochs / data sources / batch /
smoke all stay customizable via the existing flags. Moving the training code INTO the package is the parked
Phase 2. Config paths are resolved relative to the current working directory (the script's convention), so run
this from `scripts/` (or point --config / data flags at absolute paths).
"""
from __future__ import annotations

import sys

import tcr_foundation  # noqa: F401  -- import side effect: puts scripts/ on sys.path (bootstrap)


def run_pretrain(argv=None) -> None:
    """Run joint pretraining with CLI args (list of str), e.g. ["--config", cfg, "--epochs", "3"].

    Imports the training entrypoint lazily so that comet_ml imports before torch (the script's top-level
    ordering, required for Comet auto-logging) -- nothing here imports torch first."""
    if argv is None:
        argv = sys.argv[1:]
    from foundation.tcr_foundation_pretrain_joint import main as _main  # lazy: comet_ml-before-torch order
    saved = sys.argv
    sys.argv = ["tcr-foundation-train", *list(argv)]   # the script parses sys.argv via its own argparse
    try:
        _main()
    finally:
        sys.argv = saved


def cli() -> None:
    """Console-script / `python -m` entry: forward this process's args to the pretraining."""
    run_pretrain(sys.argv[1:])


if __name__ == "__main__":
    cli()
