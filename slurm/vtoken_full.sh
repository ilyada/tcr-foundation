#!/bin/bash -l

# ---- SITE-SPECIFIC resource request: these five lines describe a CLUSTER, not the code. Override them at
#      submit time (sbatch --partition=... --time=... --gres=...) or edit your own copy on the cluster. ----
#SBATCH --partition=long
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --time=6-00:00:00
#SBATCH --job-name=vtoken_full
#SBATCH --ntasks=1
#SBATCH --output=vtoken_full-%j.log

# TCRFoundation V-token run launched THROUGH the library: `python -m tcr_foundation.train` wraps the vendored
# joint pretraining entry point (tcr_foundation/_vendor/foundation/tcr_foundation_pretrain_joint.py).
#
# This file is a TEMPLATE kept under version control. Working launchers live on the cluster: the code stays
# universal, the launcher adapts one site (paths, conda env, partitions, credentials) to it. Nothing here may
# contain a hostname, a home directory or a secret.
#
# SUBMIT FROM THE REPO ROOT (REPO defaults to SLURM_SUBMIT_DIR), or pass REPO= explicitly:
#   sbatch --export=ALL,EPOCHS=5 tcr_foundation/slurm/vtoken_full.sh
#   sbatch --partition=short --time=00:30:00 --export=ALL,SMOKE=1,SAVE_PATH=/tmp/vtoken_smoke \
#          tcr_foundation/slurm/vtoken_full.sh
#
# Credentials: the Comet key is taken from the environment if set, otherwise comet_ml falls back to
# ~/.comet.config (chmod 600) on the machine. Never write a key into this file.
#
# Tokenizer: pulled by the library from the public HF repo declared as `tokenizer_hf` in the config
# (no token, no rsync). A local directory in that field is used as-is.
#
# Parameters (override via sbatch --export=ALL,KEY=VAL,...):
#   REPO             - repo root (default: SLURM_SUBMIT_DIR, i.e. where you ran sbatch)
#   WORKDIR          - cwd for the run (default: $REPO/scripts -- makes the config's ../data, ../models resolve)
#   CONDA_ENV        - conda environment to activate (default: tcr-ml-env)
#   CONFIG           - YAML config (default: ../tcr_foundation/configs/vtoken_full.yaml, relative to WORKDIR)
#   EPOCHS           - epochs (default: from config)
#   BATCH_SIZE       - per-modality batch (default: from config = 512)
#   NUM_WORKERS      - DataLoader workers PER loader (default 16; two loaders -> 32 procs = cpus-per-task)
#   SAVE_PATH        - checkpoint dir (default: from config). USE IT FOR SMOKES: --smoke otherwise writes a
#                      four-step checkpoint straight into the real run's save_path.
#   SMOKE            - 1 = wiring check (1 epoch, batch 16, ~4 steps, no comet); submit it to a short partition
#   RESUME           - 1 = resume from save_path/joint_config.json
#   COMET_PROJECT    - Comet project (default: tcr_foundation)
#   EXPERIMENT_NAME  - Comet experiment name

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
WORKDIR=${WORKDIR:-$REPO/scripts}
CONDA_ENV=${CONDA_ENV:-tcr-ml-env}
CONFIG=${CONFIG:-../tcr_foundation/configs/vtoken_full.yaml}
NUM_WORKERS=${NUM_WORKERS:-16}
COMET_PROJECT=${COMET_PROJECT:-tcr_foundation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"TCR-Foundation | vtoken + alpha-detach + SimCSE | OTS+pan+Tanno+Emerson | tiny bf16 | FULL"}

conda activate "$CONDA_ENV"
cd "$WORKDIR" || { echo "ERROR: WORKDIR '$WORKDIR' does not exist (set REPO= or WORKDIR=)"; exit 1; }
export PYTHONPATH=$REPO/tcr_foundation:$PYTHONPATH   # works without `pip install -e .`
# Comet: use the environment key if one is set; otherwise comet_ml reads ~/.comet.config. No key lives here.
[ -n "$COMET_API_KEY" ] && export COMET_API_KEY

echo "=========================================="
echo "TCRFoundation V-token -- via library (python -m tcr_foundation.train)"
echo "  Repo root:       $REPO"
echo "  Workdir:         $WORKDIR"
echo "  Conda env:       $CONDA_ENV"
echo "  Config:          $CONFIG"
echo "  Comet project:   $COMET_PROJECT"
echo "  Experiment:      $EXPERIMENT_NAME"
echo "  num_workers/ldr: $NUM_WORKERS"
echo "  env overrides:   EPOCHS=${EPOCHS:-cfg} BATCH_SIZE=${BATCH_SIZE:-cfg} SAVE_PATH=${SAVE_PATH:-cfg} SMOKE=${SMOKE:-0} RESUME=${RESUME:-0}"
echo "  -> exact resolved values are printed by the Python script (RESOLVED JOINT CONFIG)"
echo "=========================================="

python -m tcr_foundation.train \
  --config $CONFIG \
  --num-workers $NUM_WORKERS \
  ${EPOCHS:+--epochs $EPOCHS} \
  ${BATCH_SIZE:+--batch-size $BATCH_SIZE} \
  ${SAVE_PATH:+--save-path $SAVE_PATH} \
  --comet-project $COMET_PROJECT \
  --experiment-name "$EXPERIMENT_NAME" \
  ${SMOKE:+--smoke} \
  ${RESUME:+--resume}

if [ $? -ne 0 ]; then
    echo "ERROR: V-token training failed"
    exit 1
fi

echo "V-token training complete. Foundation saved (see RESOLVED JOINT CONFIG save_path)."
