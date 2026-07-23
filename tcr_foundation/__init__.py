"""
tcr_foundation -- TCR repertoire library (first cut, isolated).

Layers (protocol-based, swappable):
  schema       -- canonical clonotype table + ingestion (column auto-detect) + tidytcells germline resolver
  encoders     -- ClonotypeEncoder protocol -> per-clonotype embedding Z [N, dim]  (neural, later tcrdist)
  featurizers  -- RepertoireFeaturizer protocol -> one vector per donor  (V-usage, k-mer, mean+cov, whitened-SPD)
  metrics      -- donor-centric retrieval AUROC (canonical, one implementation)
  registry     -- name -> (FS path | HF repo) model loader  (HF backend added last)

ISOLATION (first cut): this package is NOT installed. It reuses the existing, unmodified implementations in
`scripts/` (repertoire.*, utils.*) by putting `scripts/` on sys.path here -- so deleting this whole folder
reverts everything, and no code outside the folder is touched. The real migration (vendoring those helpers,
severing the encode_repertoires->utils edge, deleting the originals) is a later, separately-approved step.
"""
import os
import sys

__version__ = "0.0.1"

# --- bootstrap: make the existing scripts/ tree importable (repertoire.*, utils.*) without installing ---
# Layout: <repo>/tcr_foundation/tcr_foundation/__init__.py  ->  parents[2] == <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if os.path.isdir(_SCRIPTS) and _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

REPO_ROOT = _REPO_ROOT
SCRIPTS_DIR = _SCRIPTS

# --- lazy public API (PEP 562): submodules load on first access, so `import tcr_foundation` stays LIGHT
#     (no torch pulled) until you touch a layer that needs it. ---
_SUBMODULES = {
    "protocols", "schema", "encoders", "featurizers", "descriptors",
    "metrics", "registry", "benchmark", "train", "hf",
}
__all__ = sorted(_SUBMODULES) + ["load", "REPO_ROOT", "SCRIPTS_DIR", "__version__"]


def __getattr__(name):
    """Import a submodule (or the `load` convenience) on first access."""
    import importlib
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "load":                                  # tcr_foundation.load("joint-vtoken-tiny")
        load = importlib.import_module(".registry", __name__).load
        globals()["load"] = load
        return load
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES) + ["load"])
