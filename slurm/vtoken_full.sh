#!/bin/bash -l

#SBATCH --job-name=vtoken_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --time=6-00:00:00
#SBATCH --output=vtoken_full-%j.log
#SBATCH --partition=long
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

# TCRFoundation V-token FULL run, launched THROUGH the library (python -m tcr_foundation.train).
# Same tiny backbone + V-token/alpha-detach, but ALL paired sources (OTS+pan+Tanno) + real Emerson, batch 512.
# The library wraps the unmodified scripts/foundation/tcr_foundation_pretrain_joint.py (Phase-1 reversible).
#
# PREREQ: the V-token tokenizer models/tokenizers/tcr-vtoken (+ vgene_map.json) must be on the cluster (rsync
#   from local, or rebuild via scripts/utils/data_build/build_vtoken_tokenizer.py) -- models/ is gitignored.
# INSTALL (once): from the repo root, `pip install -e .` inside tcr_foundation/ (editable, keeps the package in
#   the repo so its sys.path bootstrap still finds scripts/). This script also sets PYTHONPATH so it runs even
#   without installing.
#
# Parameters (override via sbatch --export=ALL,KEY=VAL,...):
#   CONFIG           - YAML (default: ../tcr_foundation/configs/vtoken_full.yaml)
#   EPOCHS           - epochs (default: from config = 3)   <- customizable per job
#   BATCH_SIZE       - default: from config = 512
#   NUM_WORKERS      - DataLoader workers PER loader (default 8; two loaders -> 16 procs)
#   EXPERIMENT_NAME  - Comet experiment name
#   COMET_PROJECT    - Comet project (default: tcr_foundation)
#   RESUME           - set to 1 to resume from save_path/joint_config.json

REPO=/home/idyugay/ADSCC
CONFIG=${CONFIG:-../tcr_foundation/configs/vtoken_full.yaml}
NUM_WORKERS=${NUM_WORKERS:-8}
COMET_PROJECT=${COMET_PROJECT:-tcr_foundation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"TCR-Foundation | vtoken + alpha-detach + SimCSE | OTS+pan+Tanno+Emerson | tiny bf16 | FULL"}

conda activate tcr-ml-env
cd $REPO/scripts                          # cwd = scripts/ so the config's ../data, ../models paths resolve
export PYTHONPATH=$REPO/tcr_foundation:$PYTHONPATH   # find the package even without `pip install -e .`
export COMET_API_KEY=${COMET_API_KEY:-NXKUKctBTmUQxgjuOWanHolhD}

echo "=========================================="
echo "TCRFoundation V-token FULL -- via library (python -m tcr_foundation.train)"
echo "  Config:          $CONFIG"
echo "  Comet project:   $COMET_PROJECT"
echo "  Experiment:      $EXPERIMENT_NAME"
echo "  num_workers/ldr: $NUM_WORKERS"
echo "  env overrides:   EPOCHS=${EPOCHS:-cfg3} BATCH_SIZE=${BATCH_SIZE:-cfg512} RESUME=${RESUME:-0}"
echo "  -> exact resolved values are printed by the Python script (RESOLVED JOINT CONFIG)"
echo "=========================================="

python -m tcr_foundation.train \
  --config $CONFIG \
  --num-workers $NUM_WORKERS \
  ${EPOCHS:+--epochs $EPOCHS} \
  ${BATCH_SIZE:+--batch-size $BATCH_SIZE} \
  --comet-project $COMET_PROJECT \
  --experiment-name "$EXPERIMENT_NAME" \
  ${RESUME:+--resume}

if [ $? -ne 0 ]; then
    echo "ERROR: V-token full training failed"
    exit 1
fi

echo "V-token full training complete. Foundation saved (see RESOLVED JOINT CONFIG save_path)."
