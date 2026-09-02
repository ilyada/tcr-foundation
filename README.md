# tcr_foundation

TCR repertoire toolkit: canonical clonotype IO, swappable clonotype **encoders**, repertoire
**featurizers/descriptors**, donor-centric **metrics**, a backend-agnostic model **registry**, a
**benchmark** harness, and a wrapper to **train** the foundation model from scratch.

## Install

```bash
pip install "tcr-foundation[neural,hf] @ git+https://github.com/ilyada/tcr-foundation"
```

Extras: `neural` (our encoder: torch + transformers + tidytcells) | `sceptr` | `tcrdist-ref` | `hf` (weights
from the Hub) | `train`. Without extras the install stays light and gives you the model-free half — schema,
V-usage/k-mer featurizers, metrics.

Weights are **not** in the package; they come from the Hub on first use (see [Model registry](#model-registry)):

```python
import tcr_foundation as tf
enc = tf.load("joint-tiny")       # downloads the checkpoint, returns a ready encoder
Z, keep = enc.encode(df)          # df: v_gene + cdr3  ->  Z [N, 128]
```

`import tcr_foundation` is **light** — submodules load lazily (PEP 562), so nothing pulls torch until you
touch a layer that needs it.

### Self-contained

The package carries its own copy of the training/analysis pipeline in `tcr_foundation/_vendor/`
(`utils.*`, `repertoire.*`, `foundation.*`), put on `sys.path` by `__init__.py` — so it needs nothing
outside itself, training from scratch included. `_vendor/` lives **inside** the package directory precisely
so wheels ship it; while it sat one level up, `pip install .` produced an installable package whose every
`_vendor`-backed submodule raised `ModuleNotFoundError: No module named 'repertoire'`.

For development, an editable install from a checkout works the same way:

```bash
pip install -e ".[neural,hf]"     # also adds the `tcr-foundation-train` console script
```

## Quickstart: repertoire → encoder → descriptor → AUROC

```python
import numpy as np, glob
import tcr_foundation as tf
from tcr_foundation import schema, descriptors, featurizers, metrics

# 1. read per-donor clonotype tables (columns auto-detected: VDJtools / MiXCR / AIRR / our clouds)
paths  = sorted(glob.glob("clouds/*.parquet"))
dfs    = [schema.read(p) for p in paths]                      # canonical: v_gene, cdr3 (+ j_gene, count)
labels = np.array([1 if "CD8" in p else 0 for p in paths])

# 2. an encoder: registry name -> local checkpoint, else pulled from HF
enc = tf.load("joint-tiny")                                   # == registry.load(...) -> NeuralEncoder

# 3. a descriptor over the encoder's cloud (one vector per donor)
desc = descriptors.MeanCov(enc).fit(dfs)                      # also: WithinV, Occupancy, WhitenedSPD
V    = np.stack([desc.featurize(d) for d in dfs])

# 4. donor-centric retrieval AUROC (ref_size 1 == leave-one-out)
print(metrics.ref_size_sweep_auroc(V, labels, [1, 3, 10]))

# model-free baselines use the SAME interface -- the must-beat benchmark is V-usage
vu = featurizers.VUsage().fit(dfs)                            # also: Kmer(k=3)
print(metrics.ref_size_sweep_auroc(np.stack([vu.featurize(d) for d in dfs]), labels, [1, 3, 10]))
```

The same comparison in three lines through the benchmark layer:

```python
from tcr_foundation import benchmark as B
feats = {"V-usage": featurizers.VUsage(), "model mean+cov": descriptors.MeanCov(enc)}
print(B.donor_task(feats, dfs, labels, ref_sizes=(1, 3, 10)))   # {feature: {ref_size: auroc}}
```

## Layers (protocol-based, swappable — see `protocols.py`)

| Layer | What it gives you |
|---|---|
| `schema` | canonical clonotype table (`v_gene`, `cdr3` + `j_gene`/`count`/`cdr1`/`cdr2`), `ingest`/`read` with column auto-detect, `resolve_germline` (tidytcells V → CDR1/2, lazy) |
| `encoders` | `ClonotypeEncoder.encode(df) -> (Z [N, dim], keep)`. `NeuralEncoder` (the maintained cdr123 model), `SceptrEncoder` (dim 64) |
| `featurizers` | model-free, one vector per donor: `VUsage`, `Kmer(k)` — `fit(dfs)` on a corpus, then `featurize(df)` |
| `descriptors` | encoder-backed, same interface: `MeanCov` (mean + covariance), `WithinV` (germline-V removed), `Occupancy` (landmark / bag-of-TCR-words), `WhitenedSPD` (mean + logm(cov) in a whitened PCA frame) |
| `metrics` | `donor_centric_auroc`, `ref_size_sweep_auroc` (one canonical implementation) |
| `registry` | `load(name)` → encoder; name → FS path or HF repo |
| `benchmark` | `donor_task`, `vaccine_delta`, plus a CLI |
| `train` | launch the joint pretraining (see below) |
| `hf` | `resolve_tokenizer(ref)` — HF repo id or local dir → local directory |

## Model registry

```python
tcr_foundation.load("joint-tiny")          # backbone + pooler + tokenizer, ready to encode
```

Resolution order per entry: **local filesystem first** (`$TCR_FOUNDATION_MODELS` or `<repo>/models`, the
on-cluster fast path), then **Hugging Face** if the local dir is absent. Print the catalogue — including
which entries are fetchable — with `python -m tcr_foundation.registry`:

| Name | HF | What it is |
|---|---|---|
| `joint-tiny` | `argentel/tcr-foundation-joint-tiny` (public) | Maintained cdr123-input foundation model |
| `joint-tiny-fullcov`, `joint-light`, `stage1-light` | — | History and controls; weights exist only where they were trained, so these report `MISSING` |

Changing where a model lives means editing one registry entry — call sites do not change. HF downloads go
to `~/.cache/tcr_foundation/models/<repo>` as **real files** (`local_dir`), not the symlink cache, which is
what makes it work on Windows without Developer Mode (the symlink cache raises WinError 1314).

## Benchmarking

```bash
python -m tcr_foundation.benchmark --clouds <dir> --encoder ours --model <ckpt> --task cd4cd8
```

- `donor_task(featurizers, dfs, labels, ref_sizes)` — one label per donor → `{feature: {ref_size: auroc}}`
- `vaccine_delta(featurizers, cloud_dir, meta_path, ref_size=10)` — before/after paired subjects; AUROC of
  the **delta** direction, i.e. signal beyond each subject's baseline
- `--encoder ours | sceptr | none`; model-free features (V-usage, k-mer) run on CPU, encoder-backed ones
  want a GPU

## Non-productive nucleotide generator

The current preparatory nucleotide generator fits separate autoregressive densities: `P_pre` on out-of-frame rearrangements and `P_post` on productive rearrangements. Its fixed-length `log Q` removes the explicit length term from both densities. This nucleotide `P_post` is not the target post-selection density in the IGoR proposal: selection acts on amino-acid receptors, for which the future target is `P_post(a)`.

Build the event corpus with an identity map before any donor-level or paired analysis. The map is a TSV or parquet file with one row per sample and the required columns `sample` and `donor`; `library` and `timepoint` remain available for longitudinal analyses. The corpus preserves these identifiers per event, fits library-depth bins on training donors only, and writes `build_manifest.json` with every source input that was not rebuilt.

```bash
python -m tcr_foundation.events --input <raw-dir> --out <event-build-dir> --platform adaptive --identity-map <identity.tsv>
python -m tcr_foundation.germline --events <event-build-dir/events> --out <reference.json>
python -m tcr_foundation.generator --events <event-build-dir/events> --samples <event-build-dir/samples.parquet> --reference <reference.json> --donor-representations none --train --out <generator-out-dir>
```

The donor representation specifies what information about a donor enters the generator. The current baseline is `none`, which supplies no donor representation. The model contains a placeholder method for a future IGoR vector, but no donor-ID table or unknown-donor embedding remains. Until the versioned IGoR cache and its conditioner are implemented, any value other than `none` is rejected. Training also rejects missing cohort metadata, an empty holdout cohort, and an empty train/dev/holdout partition. The output CSV records best dev likelihood and held-out-donor likelihood for each class and donor representation.

### OAR correction

`tcr_foundation.oar` accepts one patient-level canonical parquet, Adaptive TSV, or MiXCR TSV. It estimates V- and J-gene over-amplification rates from that patient's non-productive (`out` plus `stop`) events independently for each `sample` and chain, then divides productive template counts by the product of their V/J factors. The corrected productive table retains raw counts and OAR provenance. It never overwrites the input.

```bash
python -m tcr_foundation.oar --input PATIENT.tsv --out PRODUCTIVE_OAR.parquet --chain TRB
```

With `--min-unique-clonotypes 15` (the default), a V/J segment represented by fewer than 15 unique non-productive clonotypes, including a segment absent from the calibration set, receives a neutral `OAR = 1`. The output records whether each factor was calibrated, sparse, or absent. This is the documented `min_outframe` behavior of iROAR.

The maintained repertoire-cloud builder has one raw/OAR path. Its `--oar` tag applies patient-specific correction before log weighting; without it, the same Adaptive, MiXCR, or canonical-Parquet reader, CDR resolution, tokenisation, and encoder use raw template counts. The historical VDJtools reader remains available for raw-only inputs, which cannot support OAR because they lack non-productive frame calls. Every cloud records the Boolean `oar` tag, raw and effective counts, normalized `w_log`, and OAR provenance. For an OAR run with a local checkpoint:

```bash
python -m tcr_foundation._vendor.repertoire.encode_repertoires --input-dir PATIENT_DIRECTORY --glob 'P*.tsv' --chain beta --model CHECKPOINT_DIRECTORY --output-dir OAR_CLOUD_DIRECTORY --oar
```

`python -m tcr_foundation.oar_clouds --input PATIENT_DIRECTORY --out NEW_RUN_DIRECTORY` remains a convenience wrapper: it downloads `argentel/tcr-foundation-joint-tiny` once into `NEW_RUN_DIRECTORY/model/`, calls the same `--oar` builder, and writes one cloud per patient plus shared `oar_factors.parquet` and `oar_summary.parquet` audit tables.

## Training the model

`train.py` forwards CLI flags to the vendored `foundation/tcr_foundation_pretrain_joint.py`, so epochs,
data sources and batch size stay controllable through the existing flags (and `comet_ml` still imports
before torch, as that script requires).

```bash
# local smoke from a checkout: a few steps to prove the entrypoint launches
python -m tcr_foundation.train --config tcr_foundation/configs/joint_full.yaml --emerson-data-path <a local Emerson parquet> --smoke --no-comet
```

Training needs data you supply: the config's paths are relative to the working directory.

### Cluster run

`slurm/joint_full.sh` is a **generic template**, not a site config: the code stays universal and the
launcher adapts one cluster to it. It carries no host name, home directory or credential — `REPO` comes
from `SLURM_SUBMIT_DIR` (so submit from the repo root), and everything site-specific is an env override.
Your cluster's own working launchers belong on the cluster, not in this repository.

```bash
cd <repo root>
sbatch --export=ALL,CONDA_ENV=<environment>,EPOCHS=5 tcr_foundation/slurm/joint_full.sh
sbatch --partition=short --time=00:30:00 --export=ALL,CONDA_ENV=<environment>,SMOKE=1,SAVE_PATH=/tmp/joint_smoke tcr_foundation/slurm/joint_full.sh
```

| Override | Meaning |
|---|---|
| `REPO`, `WORKDIR` | repo root (default `SLURM_SUBMIT_DIR`) and cwd (default `$REPO/scripts`, so the config's `../data`, `../models` resolve) |
| `CONDA_ENV` | environment to activate (required) |
| `CONFIG`, `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `SAVE_PATH` | forwarded to the trainer |
| `SMOKE`, `RESUME` | wiring check / resume from `save_path/joint_config.json` |
| `COMET_PROJECT`, `EXPERIMENT_NAME` | Comet routing |

The `#SBATCH` block (partition, constraint, gres, mem, cpus) is the other site-specific part — override it
with `sbatch` flags at submit time.

**Credentials:** the Comet key is read from `COMET_API_KEY` if set, otherwise from `~/.comet.config`
(chmod 600). Never put a key in a script.

`PYTHONPATH` is set by the launcher, so `pip install -e .` is optional.

`tcr_foundation/configs/joint_full.yaml` is the full-run config and is included in package data: all paired sources, the real Emerson corpus, batch 512,
and the maintained cdr123 input format. Checkpoints go to `models/foundation/tcr-foundation-joint-tiny`.

**Smoke caveat:** `--smoke` writes a four-step checkpoint into whatever `save_path` is active — always
pass `SAVE_PATH=` for smokes so it cannot land on a real run's checkpoint.

## Tests

| File | Needs | What it proves |
|---|---|---|
| `tests/test_synthetic.py` | nothing | Generates its own donors, so **anyone can run it**: ingest → V-usage → k-mer → descriptors → metrics → registry. The class signal is planted in V-gene usage, so V-usage must separate the groups (1.000) while the 3-mer comparison, whose CDR3s come from the same random process in both classes, must stay at chance (0.500) — a featurizer that finds signal there is broken. `TCR_FOUNDATION_TEST_MODEL=joint-tiny` adds the encoder option. |
| `tests/test_smoke_cpu.py` | closed cohort data | CPU layers on real clouds reproduce known signal (V-usage CD4/CD8 0.920, k-mer 0.850) |
| `tests/test_dogfood_neural.py` | closed cohort data + GPU | the package `NeuralEncoder` reproduces `benchmark_ram`'s CD4/CD8 mean+cov |
| `tests/test_descriptors.py` | closed cohort data | `Occupancy` (dim 32, sums to 1) and `WhitenedSPD` (dim 152) |
| `tests/test_hf_resolver.py` | network | tokenizer resolution: HF repo id and local dir passthrough |
| `tests_pytest/` | `pip install .[test]` | automated guards for identity mapping, donor-disjoint split, mandatory holdout, and donor-representation validation |

The last four read `<repo>/data/processed/clouds/cd4cd8_sorted_TRB_seq`, which is not distributable — outside
the lab, `test_synthetic.py` is the test that runs.

Run the automated guard suite with `pytest` from the outer `tcr_foundation/` directory after installing the `test` extra.

## Status

Shareable. A plain `pip install` produces a working package (all 10 submodules import from the wheel, and
the 23 vendored files ship with it), weights come from the Hub, and `tests/test_synthetic.py` runs with no
data of ours. Licence: MIT.

Still parked (needs separate approval): moving the core in and deleting the originals in `scripts/`, an
in-house TCRdist (pluggable distance matrix validated against tcrdist3), and training primitives.
