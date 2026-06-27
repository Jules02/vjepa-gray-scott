#!/bin/bash
# =============================================================================
# EXAMPLE EXPERIMENT — annotated template for running on a SLURM cluster.
#
# This is a worked example, not one of the real launchers. Copy it and adapt the
# #SBATCH directives + the body to your own run. It walks through a full
# experiment: train a Gray-Scott temporal V-JEPA, then evaluate it with VRMSE.
#
# HOW TO SUBMIT (always from the repo root, so $SLURM_SUBMIT_DIR == repo root):
#
#     sbatch scripts/example_experiment.sh
#
# Or run it directly on a machine with a GPU (no SLURM):
#
#     EBJEPA_REPO=$(pwd) bash scripts/example_experiment.sh
# =============================================================================

# ---- SLURM resource request -------------------------------------------------
# Adjust these to your cluster. Lines starting with "#SBATCH" are read by SLURM
# *before* the script runs; everything after is normal bash.
#SBATCH --job-name=gs_example
#SBATCH --partition=defq          # your cluster's GPU partition / queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --time=04:00:00          # walltime HH:MM:SS — raise for the large config
#SBATCH --output=gs_example_%j.out   # stdout -> repo root, %j = job id
#SBATCH --error=gs_example_%j.err    # stderr
# Some clusters also need an account / reservation, e.g.:
#   #SBATCH --account=<your-slurm-account>
#   #SBATCH --reservation=<event-reservation>

set -e   # stop on the first error

# ---- Environment ------------------------------------------------------------
# REPO is the repo root. Under SLURM it is the directory you ran `sbatch` from
# ($SLURM_SUBMIT_DIR); outside SLURM, export EBJEPA_REPO yourself (see header).
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"

# env.sh derives all work/cache/checkpoint paths from your cluster allocation and
# exports EBJEPA_CKPTS, EBJEPA_DSETS, etc. Override any of them by exporting the
# variable BEFORE this line (e.g. `export EBJEPA_CKPTS=/my/ckpts`).
source "$REPO/env.sh"

# uv-managed Python (matches .python-version = 3.12). Drop/adapt if your cluster
# provides Python differently (module load, conda, a plain venv, ...).
module load python312 2>/dev/null || true

# Where this run's checkpoints land. EBJEPA_CKPTS comes from env.sh.
RUN_NAME="example_$(date +%Y%m%d_%H%M%S)"
echo "=== Experiment '$RUN_NAME' | repo=$REPO | ckpts=$EBJEPA_CKPTS ==="

# ---- 1. Train ---------------------------------------------------------------
# Trains the temporal-JEPA on the self-supervised objective only. We deliberately
# do NOT monitor VRMSE during training (selecting on the reported metric would be
# cheating); epoch selection is done afterwards on the latent validation loss.
# Configs live in gray_scott/cfgs/ — swap in train_large.yaml for the big model.
uv run --project "$REPO" python -m gray_scott.main \
    --fname gray_scott/cfgs/train.yaml

# ---- 2. Evaluate ------------------------------------------------------------
# Score the trained checkpoint with VRMSE on the held-out TEST split, once, after
# training. Point CKPT at the epoch you selected (here: the run's latest).
CKPT="$EBJEPA_CKPTS/gray_scott/${RUN_NAME}/latest.pth.tar"
uv run --project "$REPO" python -m gray_scott.eval \
    --ckpt "$CKPT" \
    --H 30 \
    --split test

echo "=== DONE: $RUN_NAME ==="
