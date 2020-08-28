#!/usr/bin/env bash
set -euo pipefail

python eval_nyuv2.py -c configs/dorn/median.yml "$@"
python eval_nyuv2.py -c configs/densedepth/median.yml "$@"
