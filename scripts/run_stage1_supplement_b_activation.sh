#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_activation.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage1_supplement \
  --methods direct_absmax mxfp4 rot_absmax rot_lm \
  --bits 4 3 2 \
  --block-size 128 \
  --mxfp4-group-size 32 \
  --rotation-seed 11 \
  --dtype float16 \
  --device mps \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 32 \
  --sequence-length 512
