#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_local.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage_b \
  --linear-methods direct_absmax_w4a4 rot_absmax_w4a4 rot_lm_w4a4 rot_lm_w3a4 rot_lm_w4a3 rot_lm_w3a3 rot_lm_w2a4 \
  --ffn-methods ffn_fp16 ffn_direct_absmax_w4a4 ffn_rot_absmax_w4a4 ffn_rot_lm_w4a4 ffn_rot_lm_w3a4 ffn_rot_lm_w4a3 ffn_rot_lm_w3a3 \
  --block-size 128 \
  --dtype float16 \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 8 \
  --sequence-length 128
