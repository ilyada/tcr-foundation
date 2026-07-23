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
    "joint-tiny":        {"fs": "foundation/tcr-foundation-joint-tiny"},
    "joint-vtoken-tiny": {"fs": "foundation/tcr-foundation-joint-vtoken-tiny"},
}


def models_root() -> str:
    """Root dir holding checkpoints. Env override (cluster shared FS) or <repo>/models."""
    return os.environ.get("TCR_FOUNDATION_MODELS", os.path.join(REPO_ROOT, "models"))


def resolve(name: str) -> str:
    """Resolve a registry name to a local checkpoint directory (downloading from HF if that backend)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(REGISTRY)}")
    entry = REGISTRY[name]
    if "fs" in entry:
        path = os.path.join(models_root(), entry["fs"])
        if not os.path.isdir(path):
            raise FileNotFoundError(f"'{name}' -> {path} not found (set TCR_FOUNDATION_MODELS or add the HF backend)")
        return path
    if "hf" in entry:
        from huggingface_hub import snapshot_download   # lazy; needs the `hf` extra
        repo, revision = entry["hf"]
        return snapshot_download(repo_id=repo, revision=revision)
    raise ValueError(f"registry entry for '{name}' has no known backend: {entry}")


def load(name: str, **encoder_kwargs):
    """Resolve `name` and return a NeuralEncoder on it."""
    from .encoders import NeuralEncoder
    return NeuralEncoder(resolve(name), **encoder_kwargs)
