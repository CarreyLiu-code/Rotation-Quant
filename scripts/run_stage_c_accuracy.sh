#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_c_accuracy.py \
  --output-dir outputs/stage_c \
  --benchmark piqa
