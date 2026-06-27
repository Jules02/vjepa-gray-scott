#!/bin/bash
#SBATCH --job-name=gs_viz_s4b
#SBATCH --output=slurm_viz_s4baselines_%j.out
#SBATCH --time=60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

CKPT_SMALL="${CKPT_SMALL:-$EBJEPA_CKPTS/gray_scott/dev/epoch_19.pth.tar}"
CKPT_V2="${CKPT_V2:-$EBJEPA_CKPTS/gray_scott/vjepa_v2/epoch_10.pth.tar}"

echo "=== GIFs: Truth | JEPA-small | JEPA-v2 (ep10) | ResUNet-s4 | FNO-s4 ==="

uv run python -m gray_scott.archive.visualize \
    --ckpt "$CKPT_SMALL" \
    --ckpt2 "$CKPT_V2" \
    --label2 "JEPA-v2 (ep10)" \
    --H 60 \
    --fps 10 \
    --n 4 \
    --time-stride 4 \
    --outdir gray_scott/viz \
    --seed 42 \
    --baselines-s4 \
    --tag s4baselines

echo "=== DONE ==="
