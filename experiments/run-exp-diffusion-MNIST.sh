#!/bin/bash
#SBATCH --job-name=sim-diff-mnist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128g
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-3

set -euo pipefail

# Optional conda import
source /miniconda/bin/activate
conda activate recursive

# -----------------------------
# Params that mirror your local command
# -----------------------------
BASE_SEED=12345

T_STEPS=200
TOTAL_PER_STEP=300
EPOCHS_PER_STEP=150    # epochs per iteration 
BATCH_SIZE=50
LR=0.0002

EVAL_TRUE=1024
EVAL_GEN=1024

TDIFF=200
SCHEDULE="cosine"
BETA_START=1e-4
BETA_END=2e-2

BASE_CH=256 # was 128
SWD_PROJECTIONS=2
SWD_MAX_SAMPLES=1024
SWD_SEED_OFFSET=999

N_REPS=3

SNAPSHOT_INTERVAL=1
SNAPSHOT_N=64

# Alpha grid (array index picks one alpha)
ALPHAS=(0.25 0.50 0.75 1.00)
alpha_real=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
ALPHAS_STR="0.25, 0.50, 0.75, 1.00"

# Save MNIST
DATA_ROOT="datasets/MNIST/data"

# -----------------------------
# Logging dirs (independent of python's internal logging)
# -----------------------------
ROOT_DIR="logs/mnist_4grid_fixed_seed=${BASE_SEED}_T=${T_STEPS}_total=${TOTAL_PER_STEP}_epochs=${EPOCHS_PER_STEP}_reps=${N_REPS}_Tdiff=${TDIFF}_${SCHEDULE}_bs=${BATCH_SIZE}_lr=${LR}_ch=${BASE_CH}"
mkdir -p "${ROOT_DIR}"

TASK_DIR="${ROOT_DIR}/alpha_real=$(printf '%.2f' "${alpha_real}")"
mkdir -p "${TASK_DIR}"

exec > >(tee -a "${TASK_DIR}/slurm.out") 2> >(tee -a "${TASK_DIR}/slurm.err" >&2)

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "alpha_real=${alpha_real}"
echo "DATA_ROOT=${DATA_ROOT}"
echo

python -u ./exp-diffusion-MNIST.py \
  --base-seed "${BASE_SEED}" \
  --T-steps "${T_STEPS}" \
  --total-per-step "${TOTAL_PER_STEP}" \
  --epochs-per-step "${EPOCHS_PER_STEP}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --eval-true "${EVAL_TRUE}" \
  --eval-gen "${EVAL_GEN}" \
  --Tdiff "${TDIFF}" \
  --schedule "${SCHEDULE}" \
  --beta-start "${BETA_START}" \
  --beta-end "${BETA_END}" \
  --base-ch "${BASE_CH}" \
  --swd-projections "${SWD_PROJECTIONS}" \
  --swd-max-samples "${SWD_MAX_SAMPLES}" \
  --swd-seed-offset "${SWD_SEED_OFFSET}" \
  --alphas-real "${ALPHAS_STR}" \
  --n-reps "${N_REPS}" \
  --alpha-real "${alpha_real}" \
  --snapshot-interval "${SNAPSHOT_INTERVAL}" \
  --snapshot-n "${SNAPSHOT_N}" \
  --data-root "${DATA_ROOT}"
