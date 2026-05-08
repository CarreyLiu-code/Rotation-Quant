#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_c_kv_local.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage_c \
  --methods fp16 absmax_k4v4 absmax_k3v4 absmax_k4v3 hadamard_lm_k4v4 hadamard_lm_k3v4 hadamard_lm_k4v3 hadamard_lm_k3v3 hadamard_lm_k2v4 \
  --dtype float16 \
  --device mps \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 8 \
  --sequence-length 128
