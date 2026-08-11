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

# name -> {backend: locator, "note": what this checkpoint IS}. FS subpaths are relative to models_root.
#
# The registry is also the CATALOGUE: models/ is gitignored, so the directory names on disk are the only
# record of what exists -- and they do not say which is the fully-trained one. These notes are versioned,
# so the index survives even when the weights do not. Print it with `python -m tcr_foundation.registry`.
# Unknown keys (like "note") are ignored by resolve(); only "fs" and "hf" are backends.
REGISTRY = {
    "joint-tiny":         {"fs": "foundation/tcr-foundation-joint-tiny",              # local fast path
                           "hf": ("argentel/tcr-foundation-joint-tiny", None),        # public HF fallback (repo, revision)
                           "note": "BASELINE backbone. Joint single-stage, cdr123 input, 128/4, 5 epochs, "
                                   "batch 512, all paired sources + Emerson. Publicness rho^2 0.276; "
                                   "CD4/CD8 mean+cov 0.949, vaccine-delta 0.614 / within-V 0.609."},

    "joint-vtoken":       {"fs": "foundation/tcr-foundation-joint-vtoken",
                           "hf": ("argentel/tcr-foundation-joint-vtoken", None),       # PRIVATE repo -> needs a token
                           "private": True,                                            # catalogue hint only
                           "note": "V-token + alpha-detach, FULLY TRAINED on the cluster (job 1418983, "
                                   "2026-07-28): 128/4, 5 epochs, batch 512, all sources, real Emerson. "
                                   "Atomic V token replaces AA CDR1/CDR2; lambda_drop_alpha=0 with alpha "
                                   "SimCSE. The apples-to-apples counterpart of joint-tiny. HF repo is "
                                   "PRIVATE: `hf auth login` or export HF_TOKEN before loading."},

    "joint-vtoken-tiny":  {"fs": "foundation/tcr-foundation-joint-vtoken-tiny",
                           "note": "UNDERTRAINED probe, not for conclusions: same V-token design but 1 epoch, "
                                   "batch 128, Tanno_2020 only, trained on a local 8GB GPU (2026-07-22)."},

    "joint-tiny-fullcov": {"fs": "foundation/tcr-foundation-joint-tiny-fullcov",
                           "note": "CONTROL for the tiny-vs-light confound (2026-06-19): same arch as "
                                   "joint-tiny but retrained on the new full-coverage Emerson rotation. "
                                   "Held publicness (rho^2 0.264 vs 0.276), which exonerated the code."},

    "joint-light":        {"fs": "foundation/tcr-foundation-joint-light",
                           "note": "Larger joint model, 256/6, 5 epochs, batch 256. Publicness COLLAPSED "
                                   "(rho^2 0.008) -- shown to be architectural, not a data/code artifact. "
                                   "Kept as evidence; do not use as a backbone."},

    "stage1-light":       {"fs": "foundation/tcr-foundation-light",
                           "note": "LEGACY two-stage design: Stage 1 beta-only backbone (light). Superseded "
                                   "by the joint single-stage models above."},
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
        try:
            # real files via local_dir, not the symlink cache -> works on Windows without Developer Mode
            return snapshot_download(repo_id=repo, revision=revision, local_dir=cache)
        except Exception as exc:
            # A private repo pulled without credentials surfaces as 401/403/RepositoryNotFound -- none of which
            # says "log in". Re-raise with the fix, keeping the original exception chained for debugging.
            if entry.get("private") or _is_auth_error(exc):
                raise PermissionError(
                    f"'{name}' lives in the PRIVATE HF repo '{repo}' and could not be downloaded.\n"
                    f"  1) get access to the repo, then\n"
                    f"  2) authenticate: `hf auth login`  (or export HF_TOKEN=hf_...)\n"
                    f"Local fallback: put the checkpoint at {os.path.join(models_root(), entry.get('fs', ''))} "
                    f"or point $TCR_FOUNDATION_MODELS at the directory holding it.\n"
                    f"original error: {type(exc).__name__}: {exc}") from exc
            raise
    raise FileNotFoundError(f"'{name}' not found locally and no HF backend. entry={entry}")   # 3) nothing worked


def _is_auth_error(exc: Exception) -> bool:
    """True if this HF exception is an authentication/permission problem rather than a network hiccup.

    Matches by CLASS NAME, not isinstance: the concrete error types moved between huggingface_hub versions,
    and a wrong import here would break loading for everyone."""
    names = {type(exc).__name__ for exc in (exc, getattr(exc, "__cause__", None)) if exc is not None}
    if names & {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError", "LocalEntryNotFoundError"}:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def load(name: str, **encoder_kwargs):
    """Resolve `name` and return a NeuralEncoder on it."""
    from .encoders import NeuralEncoder
    return NeuralEncoder(resolve(name), **encoder_kwargs)


def catalog() -> list[dict]:
    """One row per registered model: name, whether the local dir is present, backends, and the note.

    Availability is checked WITHOUT downloading: a missing local dir with an `hf` backend is reported as
    "hf" (fetchable), not as absent."""
    rows = []
    for name, entry in REGISTRY.items():
        local = os.path.join(models_root(), entry["fs"]) if "fs" in entry else None
        if local and os.path.isdir(local):
            avail = "local"
        elif "hf" in entry:                                    # fetchable; mark the ones that need credentials
            avail = "hf-private" if entry.get("private") else "hf"
        else:
            avail = "MISSING"
        rows.append({
            "name": name,
            "available": avail,
            "path": local,
            "hf_repo": entry["hf"][0] if "hf" in entry else None,
            "note": entry.get("note", ""),
        })
    return rows


def main() -> None:
    """`python -m tcr_foundation.registry` -- print the catalogue instead of browsing models/ by hand."""
    print(f"models_root: {models_root()}")
    print("availability: local = on disk | hf = downloadable | hf-private = downloadable after `hf auth login` "
          "| MISSING = weights only exist on the machine that trained them\n")
    for r in catalog():
        print(f"[{r['available']:<10}] {r['name']}" + (f"   <- {r['hf_repo']}" if r["hf_repo"] else ""))
        if r["note"]:
            for line in _wrap(r["note"], 96):
                print(f"          {line}")
        print()


def _wrap(text: str, width: int) -> list[str]:
    """Minimal word wrap (no textwrap import needed for one call site)."""
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
