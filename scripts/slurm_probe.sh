#!/bin/bash
#SBATCH --job-name=gs_probe
#SBATCH --output=slurm_probe_%j.out
#SBATCH --time=30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

CKPT="${CKPT:-$EBJEPA_CKPTS/gray_scott/dev/epoch_19.pth.tar}"

echo "=== F/k probe — JEPA small D=16 stride=4 epoch=19 ==="

uv run python -m gray_scott.probe \
    --ckpt "$CKPT" \
    --time-stride 4 \
    --steps 2000 \
    --batch-size 64 \
    --epoch-size 4000

echo "=== DONE ==="
