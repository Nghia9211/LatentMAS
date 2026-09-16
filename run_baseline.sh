#!/bin/bash

# Source môi trường
source .env
# Mode: baseline | text_mas | latent_mas
CONFIG="configs/latent_mas.yaml"

python3 run.py \
    --config "$CONFIG"

# --- Override ví dụ (bỏ comment để dùng) ---
# python3 run.py --config "$CONFIG" --task aime2024 --max_samples 30
