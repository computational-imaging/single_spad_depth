#!/usr/bin/env bash
set -euo pipefail

python eval_nyuv2.py -c configs/dorn/gt_hist.yml "$@"
python eval_nyuv2.py -c configs/densedepth/gt_hist.yml "$@"
python eval_nyuv2.py -c configs/midas/gt_hist.yml "$@"
