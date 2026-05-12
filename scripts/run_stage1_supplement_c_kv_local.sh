#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_c_kv_local.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage1_supplement \
  --methods fp16 absmax_k4v4 hadamard_lm_k4v4 hadamard_lm_k3v4 hadamard_lm_k4v3 randhadamard_lm_k4v4 randhadamard_lm_k3v4 randhadamard_lm_k4v3 randortho_lm_k4v4 randortho_lm_k3v4 randortho_lm_k4v3 \
  --rotation-seed 11 \
  --dtype float16 \
  --device mps \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 8 \
  --sequence-length 128
