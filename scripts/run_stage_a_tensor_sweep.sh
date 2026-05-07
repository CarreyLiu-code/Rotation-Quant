#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_a_weight_only.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage_a \
  --bits 4 3 2 \
  --methods direct_absmax hadamard_absmax hadamard_lm \
  --block-size 128
