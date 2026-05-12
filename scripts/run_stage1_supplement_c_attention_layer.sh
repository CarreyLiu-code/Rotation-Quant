#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_c_attention_layer.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage1_supplement \
  --methods fp16 attn_kv_hlm_k4v4 attn_mxfp4_w4a4_hlm_k4v4 attn_rot_lm_w4a4_hlm_k4v4 attn_rot_lm_w3a4_hlm_k3v4 attn_randhadamard_lm_w4a4_hlm_k4v4 attn_randhadamard_lm_w3a4_hlm_k3v4 attn_randortho_lm_w4a4_hlm_k4v4 attn_randortho_lm_w3a4_hlm_k3v4 \
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
