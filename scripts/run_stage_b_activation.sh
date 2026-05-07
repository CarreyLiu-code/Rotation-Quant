#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src conda run -n rotationquant python experiments/stage_b_activation.py \
  --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --output-dir outputs/stage_b \
  --methods direct_absmax rot_absmax rot_lm \
  --bits 4 3 2 \
  --block-size 128 \
  --dtype float16 \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 32 \
  --sequence-length 512
