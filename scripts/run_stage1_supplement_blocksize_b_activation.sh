#!/usr/bin/env bash
set -euo pipefail

for block_size in 32 64 128; do
  PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_activation.py \
    --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --output-dir outputs/stage1_supplement \
    --methods mxfp4 rot_mxfp4 rot_lm \
    --bits 4 3 \
    --block-size "${block_size}" \
    --mxfp4-group-size 32 \
    --rotation-seed 11 \
    --dtype float16 \
    --device mps \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split test \
    --max-samples 32 \
    --sequence-length 512
done
