#!/bin/bash
#SBATCH --job-name=gs_viz_v2
#SBATCH --output=slurm_viz_withv2_%j.out
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

echo "=== GIFs: Truth | JEPA-small | JEPA-v2 (abenmanso ep10) | UNetClassic | CNextU-Net ==="

# To use a local clone of The Well instead of the pip package, set THE_WELL_REPO.
[ -n "${THE_WELL_REPO:-}" ] && export PYTHONPATH="$THE_WELL_REPO:${PYTHONPATH:-}"

uv run --with "neuraloperator==0.3.0" --with torch-harmonics --with timm --with einops \
    python -m gray_scott.archive.visualize \
        --ckpt "$CKPT_SMALL" \
        --ckpt2 "$CKPT_V2" \
        --H 60 \
        --fps 10 \
        --n 4 \
        --time-stride 4 \
        --outdir gray_scott/viz \
        --seed 42 \
        --baselines \
        --tag withv2

echo "=== DONE ==="
