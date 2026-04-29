#!/usr/bin/env bash
set -euo pipefail

# 1) Set this to your wandb API key (optional; set wandb_enabled=false in the config to skip).
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# 2) LeRobot repo_id used for the converted G2 dataset (must match config / norm stats).
export G2_LEROBOT_REPO_ID="${G2_LEROBOT_REPO_ID:-local/icra_g2_dataset}"

# Step 1: Convert raw G2 data into a LeRobot dataset (24-D action / 26-D state).
uv run examples/g2/convert_g2_data_to_lerobot.py \
    --data-dir <your_g2_icra_data_dir> \
    --num-workers 32 \
    --queue-size 32

# Step 2: Compute normalization statistics.
#   Fast path (recommended): reads parquet directly, equivalent to compute_norm_stats.py.
uv run scripts/compute_norm_stats_fast.py --config-name pi05_g2_finetune
#   Reference path (slower):
# uv run scripts/compute_norm_stats.py --config-name pi05_g2_finetune

# Step 3: Fine-tune pi0.5 on G2 data.
uv run scripts/train.py pi05_g2_finetune --exp-name g2_finetune_run_1
