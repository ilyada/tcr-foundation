"""
hf.py -- the LIBRARY's own tokenizer fetch: resolve a tokenizer reference to a LOCAL directory.

A reference is either an existing local directory (returned as-is) or a Hugging Face repo id (downloaded
via snapshot_download to the HF cache, and that local dir returned). This is how the library keeps its OWN
tokenizer "подсос" separate from the training script's: a library config can point tokenizer storage at HF,
but the underlying (UNMODIFIED) pretrain script still receives a plain LOCAL path (its own mechanism).

The resolved directory is expected to contain BOTH the tokenizer files (for AutoTokenizer.from_pretrained)
AND vgene_map.json (the V-gene -> token map). Needs the `hf` extra (huggingface_hub) only for the HF branch.
"""
from __future__ import annotations

import os


def resolve_tokenizer(ref: str, revision: str = None) -> str:
    """Return a LOCAL directory holding the tokenizer (+ vgene_map.json).

    - existing local dir  -> returned unchanged (no network, no dep);
    - otherwise treated as an HF repo id -> snapshot_download into a plain local dir -> that dir.

    Downloads into `~/.cache/tcr_foundation/tokenizers/<repo>` via `local_dir` (REAL files, not cache
    symlinks) so it works on Windows without Developer Mode too (the default cache symlinks raise WinError
    1314). Re-calls reuse the dir (only changed files re-download)."""
    if ref and os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download   # lazy: only the HF branch needs the dep
    cache = os.path.join(os.path.expanduser("~"), ".cache", "tcr_foundation", "tokenizers",
                         str(ref).replace("/", "__"))
    return snapshot_download(repo_id=ref, revision=revision, local_dir=cache)
