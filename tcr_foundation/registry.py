"""
registry.py -- backend-agnostic model registry: a friendly name -> where the weights live -> a loaded encoder.

An entry resolves to a local checkpoint directory via one of two backends:
  {"fs": "<subpath>"}          -> <models_root>/<subpath>   (models_root = $TCR_FOUNDATION_MODELS or <repo>/models)
  {"hf": ("repo", "revision")} -> snapshot_download(repo, revision)   [needs the `hf` extra; added last]

`load(name)` returns a NeuralEncoder on the resolved directory. The FS backend is the on-cluster fast path
(weights already on disk, no download); the HF backend is the portable primary store, wired in the LAST phase
once an HF account exists. Swapping a model's storage = editing one registry entry; call sites don't change.
"""
from __future__ import annotations

import os

from . import REPO_ROOT

# name -> {backend: locator}. FS subpaths are relative to models_root.
REGISTRY = {
    "joint-tiny":        {"fs": "foundation/tcr-foundation-joint-tiny",             # local fast path
                          "hf": ("argentel/tcr-foundation-joint-tiny", None)},       # public HF fallback (repo, revision)
    "joint-vtoken-tiny": {"fs": "foundation/tcr-foundation-joint-vtoken-tiny"},      # HF weights not uploaded yet
}


def models_root() -> str:
    """Root dir holding checkpoints. Env override (cluster shared FS) or <repo>/models."""
    return os.environ.get("TCR_FOUNDATION_MODELS", os.path.join(REPO_ROOT, "models"))


def resolve(name: str) -> str:
    """Resolve a registry name to a local checkpoint directory (downloading from HF if that backend)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(REGISTRY)}")
    entry = REGISTRY[name]
    if "fs" in entry:                                          # 1) local fast path (cluster / your machine)
        path = os.path.join(models_root(), entry["fs"])
        if os.path.isdir(path):                               # only if actually present -> else fall through to HF
            return path
    if "hf" in entry:                                         # 2) portable: pull from HF into a local cache
        from huggingface_hub import snapshot_download          # lazy; needs the `hf` extra
        repo, revision = entry["hf"]                           # (repo_id, revision) from the registry entry
        cache = os.path.join(os.path.expanduser("~"), ".cache", "tcr_foundation", "models",
                             str(repo).replace("/", "__"))     # per-repo local dir under ~/.cache
        return snapshot_download(repo_id=repo, revision=revision, local_dir=cache)  # real files, no symlink cache (Windows-safe)
    raise FileNotFoundError(f"'{name}' not found locally and no HF backend. entry={entry}")   # 3) nothing worked


def load(name: str, **encoder_kwargs):
    """Resolve `name` and return a NeuralEncoder on it."""
    from .encoders import NeuralEncoder
    return NeuralEncoder(resolve(name), **encoder_kwargs)
