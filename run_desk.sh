#!/usr/bin/env bash
set -euo pipefail

python eval_captured.py \
    --scene-config ./data/captured/processed/desk.yml \
    --mde-config ./configs/captured/dorn.yml \
    "$@"

python eval_captured.py \
    --scene-config ./data/captured/processed/desk.yml \
    --mde-config ./configs/captured/densedepth.yml \
    "$@"

python eval_captured.py \
    --scene-config ./data/captured/processed/desk.yml \
    --mde-config ./configs/captured/midas.yml \
    "$@"
