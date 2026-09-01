"""
tcr_foundation -- TCR repertoire library (first cut, isolated).

Layers (protocol-based, swappable):
  schema       -- canonical clonotype table + ingestion (column auto-detect) + tidytcells germline resolver
  encoders     -- ClonotypeEncoder protocol -> per-clonotype embedding Z [N, dim]  (neural, later tcrdist)
  featurizers  -- RepertoireFeaturizer protocol -> one vector per donor  (V-usage, k-mer, mean+cov, whitened-SPD)
  metrics      -- donor-centric retrieval AUROC (canonical, one implementation)
  registry     -- name -> (FS path | HF repo) model loader  (HF backend added last)

SELF-CONTAINED: the pipeline (repertoire.*, utils.*, foundation.*) is VENDORED into `_vendor/` and put on
sys.path here -- the package needs nothing outside this folder, so it can be shared and used (including
training from scratch) without the rest of the repo. It is still not installed by default, and deleting
this folder reverts the project: nothing outside it was modified. Parked (needs separate approval): moving
the core in and deleting the originals in scripts/, in-house TCRdist, training primitives, private weights.
"""
import os
import sys

__version__ = "0.1.0"   # keep in sync with pyproject.toml [project] version

# --- bootstrap: SELF-CONTAINED. Resolve the training/analysis pipeline (utils.*, repertoire.*, foundation.*)
#     from our OWN VENDORED copy in _vendor/, NOT the repo's scripts/. So the package needs nothing outside this
#     folder. _vendor/ lives INSIDE the package (next to this file) so that a plain `pip install .` ships it in
#     the wheel; when it sat one level up, wheels silently dropped it and every _vendor-backed submodule
#     (metrics, featurizers, descriptors, encoders, benchmark, train) raised ModuleNotFoundError: 'repertoire'.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))   # source checkout only; installed copies use $TCR_FOUNDATION_MODELS
_VENDOR = os.path.join(_HERE, "_vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

REPO_ROOT = _REPO_ROOT
VENDOR_DIR = _VENDOR

# --- lazy public API (PEP 562): submodules load on first access, so `import tcr_foundation` stays LIGHT
#     (no torch pulled) until you touch a layer that needs it. ---
_SUBMODULES = {
    "protocols", "schema", "events", "encoders", "featurizers", "descriptors",
    "metrics", "diagnostics", "generator", "oar", "oar_clouds", "registry", "benchmark", "train", "hf",
}
__all__ = sorted(_SUBMODULES) + ["load", "REPO_ROOT", "VENDOR_DIR", "__version__"]


def __getattr__(name):
    """Import a submodule (or the `load` convenience) on first access."""
    import importlib
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "load":                                  # tcr_foundation.load("joint-tiny")
        load = importlib.import_module(".registry", __name__).load
        globals()["load"] = load
        return load
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES) + ["load"])
