#!/usr/bin/env bash
set -euo pipefail

python eval_nyuv2.py -c configs/dorn/transient.yml "$@"
python eval_nyuv2.py -c configs/densedepth/transient.yml "$@"
python eval_nyuv2.py -c configs/midas/transient.yml "$@"
