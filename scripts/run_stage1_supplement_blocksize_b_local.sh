#!/usr/bin/env bash
set -euo pipefail

for block_size in 32 64 128; do
  PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_local.py \
    --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --output-dir outputs/stage1_supplement \
    --linear-methods mxfp4_w4a4 rot_mxfp4_w4a4 rot_lm_w4a4 rot_lm_w3a4 \
    --ffn-methods ffn_fp16 ffn_mxfp4_w4a4 ffn_rot_mxfp4_w4a4 ffn_rot_lm_w4a4 ffn_rot_lm_w3a4 \
    --block-size "${block_size}" \
    --mxfp4-group-size 32 \
    --rotation-seed 11 \
    --dtype float16 \
    --device mps \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split test \
    --max-samples 8 \
    --sequence-length 128
done
