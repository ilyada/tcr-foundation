"""
train.py -- launch the joint TCR foundation pretraining THROUGH the library.

Usage:
    python -m tcr_foundation.train --config <cfg> [--epochs N --batch-size B --smoke --no-comet ...]
    tcr-foundation-train --config <cfg> ...        # console script, after `pip install -e .`

Phase-1 REVERSIBLE wrapper: no training logic lives here. It imports the UNMODIFIED entrypoint
`scripts/foundation/tcr_foundation_pretrain_joint.py` (reachable via the sys.path bootstrap in
`tcr_foundation/__init__.py`) and forwards every CLI flag straight through -- so epochs / data sources /
batch / smoke all stay customizable via the existing flags.

Tokenizer "подсос" (library-side): if the --config declares `tokenizer_hf: <repo-id>`, the LIBRARY resolves
it to a local dir (HF snapshot_download, or a local dir as-is) and hands the underlying script a patched temp
config whose `tokenizer_path`/`vgene_map_path` point at that local dir. The script keeps its OWN mechanism
(load from a local path); it never learns about HF. No edits to scripts/. See hf.resolve_tokenizer.

Config paths are resolved relative to the current working directory (the script's convention), so run this
from `scripts/` (or point --config / data flags at absolute paths).
"""
from __future__ import annotations

import os
import sys

import tcr_foundation  # noqa: F401  -- import side effect: puts scripts/ on sys.path (bootstrap)


def _maybe_resolve_hf_tokenizer(argv):
    """If the --config declares `tokenizer_hf`, resolve it to a local dir and return (patched_argv, tmp_path)
    where --config points at a temp copy of the config with tokenizer_path/vgene_map_path overridden to that
    local dir. No-op (returns argv unchanged, tmp_path None) when there is no `tokenizer_hf`."""
    import tempfile

    import yaml

    argv = list(argv)
    # locate --config value (supports "--config X" and "--config=X")
    cfg_idx = None
    cfg_path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg_idx, cfg_path = i + 1, argv[i + 1]
            break
        if a.startswith("--config="):
            cfg_idx, cfg_path = i, a.split("=", 1)[1]
            break
    if cfg_path is None or not os.path.isfile(cfg_path):
        return argv, None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("input_format", "cdr123") != "cdr123":
        raise ValueError("only the maintained cdr123 training format is supported")
    ref = cfg.get("tokenizer_hf")
    if not ref:
        return argv, None                                   # script's local-path mechanism, untouched

    from ..integrations.hf import resolve_tokenizer
    tok_dir = resolve_tokenizer(ref, cfg.get("tokenizer_hf_revision"))
    cfg["tokenizer_path"] = tok_dir
    cfg["vgene_map_path"] = os.path.join(tok_dir, "vgene_map.json")

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    argv[cfg_idx] = (f"--config={tmp.name}" if argv[cfg_idx].startswith("--config=") else tmp.name)
    print(f"[tcr_foundation] tokenizer_hf='{ref}' -> {tok_dir}  (patched config -> {tmp.name})")
    return argv, tmp.name


def run_pretrain(argv=None) -> None:
    """Run joint pretraining with CLI args (list of str), e.g. ["--config", cfg, "--epochs", "3"].

    Resolves a library-side HF tokenizer if the config asks for one, then imports the training entrypoint
    lazily (so comet_ml imports before torch -- the script's required order; nothing here imports torch first)."""
    if argv is None:
        argv = sys.argv[1:]
    argv, tmp_cfg = _maybe_resolve_hf_tokenizer(argv)
    from foundation.tcr_foundation_pretrain_joint import main as _main   # lazy: comet_ml-before-torch order
    saved = sys.argv
    sys.argv = ["tcr-foundation-train", *list(argv)]        # the script parses sys.argv via its own argparse
    try:
        _main()
    finally:
        sys.argv = saved
        if tmp_cfg and os.path.exists(tmp_cfg):
            try:
                os.remove(tmp_cfg)
            except OSError:
                pass


def cli() -> None:
    """Console-script / `python -m` entry: forward this process's args to the pretraining."""
    run_pretrain(sys.argv[1:])


if __name__ == "__main__":
    cli()
