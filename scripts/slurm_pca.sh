#!/bin/bash
#SBATCH --job-name=gs_pca
#SBATCH --output=slurm_pca_%j.out
#SBATCH --time=20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

CKPT="${CKPT:-$EBJEPA_CKPTS/gray_scott/dev/epoch_19.pth.tar}"

echo "=== PCA of JEPA-small latents ==="

uv run python -m gray_scott.archive.pca \
    --ckpt "$CKPT" \
    --split valid \
    --n-clips 300 \
    --n-frames 4 \
    --time-stride 4 \
    --outdir gray_scott/viz \
    --tag small_ep19_s4

echo "=== DONE ==="
