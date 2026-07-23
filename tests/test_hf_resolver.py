"""
Test the library-side tokenizer resolver + config patch (no network): resolve_tokenizer passes a local dir
through unchanged, and train._maybe_resolve_hf_tokenizer rewrites --config to a temp copy whose
tokenizer_path / vgene_map_path point at the resolved dir. The HF-download branch is exercised only when a
real repo exists (verified at upload time); here we drive the local branch by using a local dir as the ref.

Run: python tcr_foundation/tests/test_hf_resolver.py
"""
import os
import sys
import tempfile

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PKG_DIR)
sys.path.insert(0, _PKG_DIR)

from tcr_foundation import hf                      # noqa: E402
from tcr_foundation import train                   # noqa: E402

fails = []

# a real local dir to stand in for "the resolved tokenizer" (exists -> resolve_tokenizer passthrough)
TOK_DIR = os.path.join(_REPO, "models", "tokenizers", "tcr-vtoken")
assert os.path.isdir(TOK_DIR), f"expected local tokenizer at {TOK_DIR}"

# (1) resolve_tokenizer: local dir -> passthrough (no network, no hf dep)
r = hf.resolve_tokenizer(TOK_DIR)
print(f"resolve_tokenizer(local) -> {r}")
if r != TOK_DIR:
    fails.append("resolve_tokenizer local passthrough")

# (2) _maybe_resolve_hf_tokenizer: a config with tokenizer_hf -> patched temp config with local paths
base_cfg = {"input_format": "vtoken", "tokenizer_hf": TOK_DIR,      # local dir acts as the 'repo' ref
            "tokenizer_path": "SHOULD_BE_OVERRIDDEN", "vgene_map_path": "SHOULD_BE_OVERRIDDEN",
            "batch_size": 128, "epochs": 3}
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    yaml.safe_dump(base_cfg, f)
    cfg_path = f.name

argv = ["--config", cfg_path, "--epochs", "1"]
new_argv, tmp = train._maybe_resolve_hf_tokenizer(argv)
print(f"argv: {argv}\n ->  {new_argv}\n tmp: {tmp}")
patched = yaml.safe_load(open(new_argv[1]))
print(f"patched tokenizer_path = {patched['tokenizer_path']}")
print(f"patched vgene_map_path = {patched['vgene_map_path']}")
ok2 = (tmp is not None and new_argv[1] == tmp and new_argv[0] == "--config"
       and patched["tokenizer_path"] == TOK_DIR
       and patched["vgene_map_path"] == os.path.join(TOK_DIR, "vgene_map.json")
       and patched["epochs"] == 3 and patched["batch_size"] == 128)   # other keys preserved
if not ok2:
    fails.append("config patch (tokenizer_hf -> local paths, rest preserved)")

# (3) no tokenizer_hf -> passthrough, no temp config (script's own local mechanism untouched)
base_cfg.pop("tokenizer_hf")
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    yaml.safe_dump(base_cfg, f); cfg2 = f.name
argv2 = ["--config", cfg2]
na2, tmp2 = train._maybe_resolve_hf_tokenizer(argv2)
print(f"no tokenizer_hf -> argv unchanged={na2 == argv2}, tmp={tmp2}")
if not (na2 == argv2 and tmp2 is None):
    fails.append("no-tokenizer_hf passthrough")

# cleanup temp files
for p in (cfg_path, cfg2, tmp):
    if p and os.path.exists(p):
        os.remove(p)

print("\nHF_RESOLVER_OK" if not fails else f"HF_RESOLVER_FAIL: {fails}")
sys.exit(0 if not fails else 1)
