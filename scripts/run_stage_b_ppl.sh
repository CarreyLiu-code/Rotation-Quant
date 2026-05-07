#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_ppl.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage_b \
  --methods fp16 ffn_direct_absmax_w4a4 ffn_rot_absmax_w4a4 ffn_rot_lm_w4a4 ffn_rot_lm_w3a4 ffn_rot_lm_w4a3 \
  --block-size 128 \
  --dtype float16 \
  --device mps \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 512 \
  --sequence-length 2048 \
  --stride 2048
