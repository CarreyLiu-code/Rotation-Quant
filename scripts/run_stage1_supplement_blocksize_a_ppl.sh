#!/usr/bin/env bash
set -euo pipefail

for block_size in 32 64 128; do
  methods=(mxfp4 hadamard_mxfp4 hadamard_lm)
  if [[ "${block_size}" == "32" ]]; then
    methods=(fp16 "${methods[@]}")
  fi
  PYTHONPATH=src conda run -n rotationquant python experiments/stage_a_ppl.py \
    --model-dir models/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --output-dir outputs/stage1_supplement \
    --methods "${methods[@]}" \
    --bits 4 3 \
    --block-size "${block_size}" \
    --mxfp4-group-size 32 \
    --rotation-seed 11 \
    --dtype float16 \
    --device mps \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split test \
    --max-samples 512 \
    --sequence-length 2048 \
    --stride 2048
done
