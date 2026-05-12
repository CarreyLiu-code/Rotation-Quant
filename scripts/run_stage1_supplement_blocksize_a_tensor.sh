#!/usr/bin/env bash
set -euo pipefail

for block_size in 32 64 128; do
  PYTHONPATH=src conda run -n rotationquant python experiments/stage_a_weight_only.py \
    --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --output-dir outputs/stage1_supplement \
    --bits 4 3 \
    --methods mxfp4 hadamard_mxfp4 hadamard_lm \
    --block-size "${block_size}" \
    --mxfp4-group-size 32 \
    --rotation-seed 11
done
