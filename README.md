# tcr_foundation (first cut — isolated)

TCR repertoire library: canonical clonotype IO, swappable clonotype **encoders**, repertoire
**featurizers/descriptors**, donor-centric **metrics**, and a backend-agnostic model **registry**.

## Isolation / rollback
This whole directory is self-contained and **not installed**. It reuses the existing, unmodified code in
`../scripts/` (`repertoire.*`, `utils.*`) by putting `scripts/` on `sys.path` in `tcr_foundation/__init__.py`.
**Deleting `tcr_foundation/` at the repo root fully reverts the project** — nothing outside this folder is
touched and there is no `pip install` (so no stray `.pth`/`.egg-link` in any environment).

Import (no install) by adding this folder to the path:

```python
import sys; sys.path.insert(0, "<repo>/tcr_foundation")
import tcr_foundation
from tcr_foundation import schema, featurizers, descriptors, metrics, registry
from tcr_foundation.encoders import NeuralEncoder, SceptrEncoder
```

## Layers (protocol-based, swappable — see `protocols.py`)
- `schema` — canonical clonotype table (`v_gene`, `cdr3`, `+ j_gene/count/cdr1/cdr2`), `ingest`/`read`
  (auto-detect columns: VDJtools/MiXCR/AIRR/our clouds), `resolve_germline` (tidytcells V→CDR1/2, lazy).
- `encoders` — `ClonotypeEncoder`: `encode(df) -> (Z [N,dim], keep)`. `NeuralEncoder` (our foundation model,
  auto-detects `vtoken` vs `cdr123` from the checkpoint), `SceptrEncoder` (dim 64). Heavy deps lazy.
- `featurizers` — model-free `RepertoireFeaturizer`s: `VUsage`, `Kmer` (fit vocab on a corpus, then featurize).
- `descriptors` — encoder-backed featurizers: `MeanCov`, `WithinV` (moments of an encoder's cloud).
- `metrics` — `donor_centric_auroc`, `ref_size_sweep_auroc` (canonical, re-exported).
- `registry` — `load("joint-vtoken-tiny")` → NeuralEncoder; `name → (FS path | HF repo)`, FS fast-path via
  `TCR_FOUNDATION_MODELS`; HF backend added last.

## Training via the library (Phase 1 — reversible)
The library can launch the joint V-token pretraining. `tcr_foundation/train.py` is a thin wrapper that forwards
CLI flags to the UNMODIFIED `scripts/foundation/tcr_foundation_pretrain_joint.py` (reached via the sys.path
bootstrap) — so epochs / data sources / batch stay customizable through the existing flags.

```bash
# local smoke (from scripts/, no install): a few steps to prove the entrypoint launches
cd scripts
PYTHONPATH=<repo>/tcr_foundation python -m tcr_foundation.train \
    --config config/foundation/foundation_joint_vtoken_tiny.yaml \
    --emerson-data-path <a local Emerson parquet> --smoke --no-comet
```

### Cluster full run (via the library)
1. `git pull` on the cluster; ensure `models/tokenizers/tcr-vtoken` (+ `vgene_map.json`) is present
   (rsync from local or rebuild via `scripts/utils/data_build/build_vtoken_tokenizer.py` — `models/` is gitignored).
2. `pip install -e .` inside `tcr_foundation/` (editable → keeps the package in-repo so the bootstrap finds
   `scripts/`; also gives the `tcr-foundation-train` console command). The slurm script also sets `PYTHONPATH`, so
   installing is optional.
3. `sbatch tcr_foundation/slurm/vtoken_full.sh` — runs `python -m tcr_foundation.train` with
   `configs/vtoken_full.yaml` (ALL sources, real Emerson, batch 512). Override per job:
   `sbatch --export=ALL,EPOCHS=5,BATCH_SIZE=512 tcr_foundation/slurm/vtoken_full.sh`.

Config: `tcr_foundation/configs/vtoken_full.yaml` (in this folder — deletable with it). `epochs` there is a default,
overridable with `--epochs` / `EPOCHS=`. Saves to `models/foundation/tcr-foundation-joint-vtoken` (distinct from
the local 1-epoch `-vtoken-tiny` checkpoint).

### Tokenizer via HF (library-side "подсос")
The library and the training script fetch the tokenizer independently. The **script** always loads it from a
LOCAL path (`tokenizer_path` + `vgene_map_path`). The **library** can instead pull it from Hugging Face: set
`tokenizer_hf: <org>/<repo>` in the config, and `python -m tcr_foundation.train` will `snapshot_download` that repo
(tokenizer files + `vgene_map.json`) and hand the script a patched temp config pointing at the local cache — so
the script stays unmodified and HF-agnostic (`hf.resolve_tokenizer` + `train._maybe_resolve_hf_tokenizer`). A local
dir given to `tokenizer_hf` is passed through as-is. Upload once: `huggingface-cli upload <org>/<repo>
models/tokenizers/tcr-vtoken` (a PUBLIC repo needs no token to pull); then uncomment `tokenizer_hf` in
`vtoken_full.yaml` and the "rsync the tokenizer" prereq goes away.

## Status
First cut, isolated. Real migration (vendor the utils model-helpers to sever the `encode_repertoires → utils`
edge, move the core in + delete originals, in-house TCRdist, training primitives, HF weight hosting) is a
later, separately-approved step. Tests: `tests/test_smoke_cpu.py` (CPU layers on real clouds),
`tests/test_dogfood_neural.py` (GPU NeuralEncoder == benchmark_ram).
