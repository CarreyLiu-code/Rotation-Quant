#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_full_ppl.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage2_full \
  --methods \
    fp16 \
    full_direct_absmax_w4a4_absmax_k4v4 \
    full_rot_lm_w4a4_hlm_k4v4 \
  --block-size 64 \
  --mxfp4-group-size 32 \
  --rotation-seed 0 \
  --dtype float16 \
  --device mps \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 2 \
  --sequence-length 64 \
  --stride 64
