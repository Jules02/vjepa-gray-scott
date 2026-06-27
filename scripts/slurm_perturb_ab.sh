#!/bin/bash
#SBATCH --job-name=gs_perturb_ab
#SBATCH --output=slurm_perturb_ab_%j.out
#SBATCH --time=20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

CKPT="${CKPT:-$EBJEPA_CKPTS/gray_scott/dev/epoch_19.pth.tar}"

uv run python -m gray_scott.archive.perturb_ab_gif \
    --ckpt "$CKPT" \
    --split valid \
    --H 40 \
    --time-stride 4 \
    --delta 1.0 \
    --fps 8 \
    --outdir gray_scott/viz \
    --tag ep19

echo "=== DONE ==="
