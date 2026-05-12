#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_local.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage1_supplement \
  --linear-methods direct_absmax_w4a4 mxfp4_w4a4 rot_absmax_w4a4 rot_lm_w4a4 rot_lm_w3a4 rot_lm_w4a3 rot_lm_w3a3 randhadamard_lm_w4a4 randhadamard_lm_w3a4 randhadamard_lm_w4a3 randortho_lm_w4a4 randortho_lm_w3a4 randortho_lm_w4a3 \
  --ffn-methods ffn_fp16 ffn_direct_absmax_w4a4 ffn_mxfp4_w4a4 ffn_rot_absmax_w4a4 ffn_rot_lm_w4a4 ffn_rot_lm_w3a4 ffn_rot_lm_w4a3 ffn_rot_lm_w3a3 ffn_randhadamard_lm_w4a4 ffn_randhadamard_lm_w3a4 ffn_randhadamard_lm_w4a3 ffn_randortho_lm_w4a4 ffn_randortho_lm_w3a4 ffn_randortho_lm_w4a3 \
  --block-size 128 \
  --mxfp4-group-size 32 \
  --rotation-seed 11 \
  --dtype float16 \
  --device mps \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 8 \
  --sequence-length 128
