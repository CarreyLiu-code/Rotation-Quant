# Stage C Refine: Value Rotation + o_proj Absorb

本文档记录 Stage C refine 的实验目标、计算路径、代码入口、产物规范和结果模板。它是 `docs/stage_c.md` 的后续修正实验，重点处理 C5 中 KV-local / Attention-local 指标较好但 model-level PPL 崩溃的问题。

本阶段仍然只解释 fake quant 数值质量和低比特可行性，不解释为真实推理加速。

## 核心原则

TinyLlama attention 使用 32 个 query heads、4 个 KV heads、head dim 64。Refine 阶段固定采用：

| Component | Decision |
| --- | --- |
| Q/K score rotation | per-head `H64` |
| Cross-head H128 for Q/K score | 不使用 |
| Value path | 新增 `o_proj_absorb` |
| `o_proj` absorb | 32 个 independent `H64` blocks |
| QJL | 本轮不纳入 |
| Output root | `outputs/stage_c_refine/<run_id>/` |

## 计算路径

旧 C5 的 Hadamard-LM Value 路径是 reconstruction：

```text
V_H_hat = Q(H64(V))
V_hat = H64^-1(V_H_hat)
O = P @ V_hat
Y = O @ W_o^T
```

Refine 新增 `o_proj_absorb`：

```text
Q_H = H64(Q_rope)
K_H = H64(K_rope)
score = Q_H @ K_H^T / sqrt(d)
P = softmax(score)

V_H_hat = Q(H64(V))
O_H = P @ V_H_hat
W_o_abs = W_o H64_blockdiag
Y = O_H @ W_o_abs^T
```

`P` 是 sequence 维度的 token mixing 权重，不属于 Value 特征域。因此：

```text
P @ (V H) = (P @ V) H
```

这使得 `V` 的 head-wise rotation 可以被推到 `o_proj` input side，并由 `W_o_abs = W_o H64_blockdiag` 吸收。

## Method Registry

Refine methods 定义在 `src/rotationquant/stage_c.py` 的 `STAGE_C_REFINE_ATTENTION_SPECS`。

| Method | Linear | KV | Value path | Purpose |
| --- | --- | --- | --- | --- |
| `attn_identity_fp16` | FP16 | FP16 | reconstruct | wrapper identity |
| `attn_kv_only_hlm_k4v4_reconstruct` | FP16 | HLM K4V4 | reconstruct | old path baseline |
| `attn_kv_only_hlm_k4v4_oabsorb` | FP16 | HLM K4V4 | o_proj_absorb | absorb path baseline |
| `attn_kv_only_hlm_k3v4_oabsorb` | FP16 | HLM K3V4 | o_proj_absorb | key bit reduction |
| `attn_kv_only_hlm_k4v3_oabsorb` | FP16 | HLM K4V3 | o_proj_absorb | value bit reduction |
| `attn_rot_lm_w4a4_hlm_k4v4_oabsorb` | Rot-LM W4A4 | HLM K4V4 | o_proj_absorb | structured baseline |
| `attn_rot_lm_w3a4_hlm_k3v4_oabsorb` | Rot-LM W3A4 | HLM K3V4 | o_proj_absorb | low-bit key/weight |
| `attn_rot_lm_w4a3_hlm_k4v3_oabsorb` | Rot-LM W4A3 | HLM K4V3 | o_proj_absorb | low-bit activation/value |

## Code Entrypoints

| File | Purpose |
| --- | --- |
| `src/rotationquant/stage_c.py` | refine registry and `quantized_kv_attention_o_proj_absorb` |
| `src/rotationquant/stage_c_model.py` | `StageCRefineAttentionWrapper` and model replacement |
| `experiments/stage_c_refine_attention_layer.py` | local Attention-layer metrics |
| `experiments/stage_c_refine_ppl.py` | model-level Attention-only PPL |
| `experiments/summarize_stage_c_refine.py` | summary helper |
| `scripts/run_stage_c_refine_attention_layer.sh` | C4-refine full local run |
| `scripts/run_stage_c_refine_ppl.sh` | C5-refine full PPL run |

## Run Plan

### R0 Wrapper Identity

Run:

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/stage_c_refine_ppl.py \
  --methods fp16 attn_identity_fp16 \
  --max-samples 512 \
  --sequence-length 2048 \
  --stride 2048 \
  --device mps
```

成功标准：`attn_identity_fp16` PPL 接近 FP16。如果失败，停止后续 PPL，优先修 wrapper 与 Hugging Face LLaMA attention forward 的一致性。

### R1 KV-only

Run local and PPL:

```text
attn_kv_only_hlm_k4v4_reconstruct
attn_kv_only_hlm_k4v4_oabsorb
attn_kv_only_hlm_k3v4_oabsorb
attn_kv_only_hlm_k4v3_oabsorb
```

目的：确认 C2 的 KV-local 结论是否能传递到 model-level。

### R2 Structured Attention Local

默认 full local：

```bash
scripts/run_stage_c_refine_attention_layer.sh
```

输出指标：

| Metric | Meaning |
| --- | --- |
| `score_relative_mse` | Q/K score error |
| `softmax_kl` | attention distribution drift |
| `pre_o_output_relative_mse` | pre-`o_proj` output in reference domain |
| `pre_o_output_cosine` | pre-`o_proj` output cosine |
| `layer_output_relative_mse` | final Attention layer output error |
| `layer_output_cosine` | final Attention layer output cosine |

### R3 Structured Attention PPL

默认 full PPL：

```bash
scripts/run_stage_c_refine_ppl.sh
```

默认数据集为 WikiText2 raw test，`max_samples=512`，`sequence_length=2048`，`stride=2048`，device `mps`。

## Output Layout

Local run:

```text
outputs/stage_c_refine/<run_id>/attention_layer_metrics.jsonl
outputs/stage_c_refine/<run_id>/attention_layer_metrics.csv
outputs/stage_c_refine/<run_id>/summary_by_method.csv
outputs/stage_c_refine/<run_id>/summary_by_layer.csv
outputs/stage_c_refine/<run_id>/summary.md
outputs/stage_c_refine/<run_id>/run_metadata.json
```

PPL run:

```text
outputs/stage_c_refine/<run_id>/ppl_runs.jsonl
outputs/stage_c_refine/<run_id>/ppl.csv
outputs/stage_c_refine/<run_id>/summary.md
outputs/stage_c_refine/<run_id>/run_metadata.json
```

Every output directory name must match `run_metadata.json.run_id`.

## Result Tables

### Local Attention-layer

| Run ID | Method | Score rel MSE | Softmax KL | Pre-o cosine | Layer output rel MSE | Layer output cosine | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| | `attn_identity_fp16` | | | | | | |
| | `attn_kv_only_hlm_k4v4_reconstruct` | | | | | | |
| | `attn_kv_only_hlm_k4v4_oabsorb` | | | | | | |
| | `attn_rot_lm_w4a4_hlm_k4v4_oabsorb` | | | | | | |

### PPL

| Run ID | Method | PPL | Interpretation |
| --- | --- | ---: | --- |
| | `fp16` | | baseline |
| | `attn_identity_fp16` | | wrapper identity |
| | `attn_kv_only_hlm_k4v4_oabsorb` | | KV-only absorb |
| | `attn_kv_only_hlm_k3v4_oabsorb` | | key bit reduction |
| | `attn_rot_lm_w4a4_hlm_k4v4_oabsorb` | | structured baseline |
| | `attn_rot_lm_w3a4_hlm_k3v4_oabsorb` | | W/K low-bit |
| | `attn_rot_lm_w4a3_hlm_k4v3_oabsorb` | | A/V low-bit |

## Current Status

Implementation is prepared. The first smoke runs passed after adding a causal-mask fallback in the custom Stage C attention path.

### Smoke: Local Attention-layer

| Item | Value |
| --- | --- |
| Run ID | `20260508_181231+0800_stage_c_refine_attention_layer` |
| Output dir | `outputs/stage_c_refine/20260508_181231+0800_stage_c_refine_attention_layer` |
| Scope | layer 0, max samples 2, sequence length 64 |
| Records | 5 |

| Method | Score rel MSE | Softmax KL | Pre-o cosine | Layer output rel MSE | Layer output cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_identity_fp16` | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| `attn_kv_only_hlm_k4v4_reconstruct` | 0.007055 | 0.002513 | 0.992267 | 0.003829 | 0.998133 |
| `attn_kv_only_hlm_k4v4_oabsorb` | 0.007055 | 0.002513 | 0.992267 | 0.003830 | 0.998133 |
| `attn_rot_lm_w4a4_hlm_k4v4_oabsorb` | 0.010309 | 0.005061 | 0.903032 | 0.037540 | 0.982361 |

Conclusion: local identity passes exactly. On layer 0, reconstruct and `o_proj_absorb` are numerically equivalent for KV-only K4V4, as expected.

### Smoke: PPL

| Item | Value |
| --- | --- |
| Run ID | `20260508_181110+0800_stage_c_refine_ppl` |
| Output dir | `outputs/stage_c_refine/20260508_181110+0800_stage_c_refine_ppl` |
| Scope | WikiText2 raw test, max samples 2, sequence length/stride 64/64 |
| Records | 3 |

| Method | PPL |
| --- | ---: |
| `fp16` | 617.328152 |
| `attn_identity_fp16` | 617.328152 |
| `attn_kv_only_hlm_k4v4_oabsorb` | 647.099665 |

Conclusion: model-level identity wrapper passes the PPL gate. The refine path is ready for formal C4-refine and C5-refine runs.

## Formal Full Runs

### C4-refine: Attention-layer Local Full

| Item | Value |
| --- | --- |
| Run ID | `20260508_182056+0800_stage_c_refine_attention_layer` |
| Output dir | `outputs/stage_c_refine/20260508_182056+0800_stage_c_refine_attention_layer` |
| Device | `mps` |
| Scope | all 22 Attention layers, WikiText2 test, max samples 8, sequence length 128 |
| Records | 198 |
| Duration | 37.700 seconds |

Mean metrics:

| Method | Linear bits | KV bits | Value path | Score rel MSE | Softmax KL | Pre-o cosine | Layer output rel MSE | Layer output cosine |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | FP16 | K16V16 | reference | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| `attn_identity_fp16` | FP16 | K16V16 | reconstruct | 0.000000 | 0.000000 | 1.000026 | 0.000001 | 1.000005 |
| `attn_kv_only_hlm_k4v4_reconstruct` | FP16 | K4V4 | reconstruct | 0.003558 | 0.009540 | 0.986765 | 0.024622 | 0.988034 |
| `attn_kv_only_hlm_k4v4_oabsorb` | FP16 | K4V4 | o_proj_absorb | 0.003558 | 0.009540 | 0.986765 | 0.024623 | 0.988034 |
| `attn_kv_only_hlm_k3v4_oabsorb` | FP16 | K3V4 | o_proj_absorb | 0.013185 | 0.035261 | 0.959010 | 0.083876 | 0.961825 |
| `attn_kv_only_hlm_k4v3_oabsorb` | FP16 | K4V3 | o_proj_absorb | 0.003558 | 0.009540 | 0.978642 | 0.036435 | 0.981916 |
| `attn_rot_lm_w4a4_hlm_k4v4_oabsorb` | W4A4 | K4V4 | o_proj_absorb | 0.005677 | 0.013771 | 0.936548 | 0.089098 | 0.956899 |
| `attn_rot_lm_w3a4_hlm_k3v4_oabsorb` | W3A4 | K3V4 | o_proj_absorb | 0.020985 | 0.046599 | 0.851703 | 0.255800 | 0.881672 |
| `attn_rot_lm_w4a3_hlm_k4v3_oabsorb` | W4A3 | K4V3 | o_proj_absorb | 0.010173 | 0.019506 | 0.903753 | 0.162295 | 0.920949 |

Conclusion: the identity wrapper is numerically consistent with FP16. For KV-only K4V4, reconstruct and `o_proj_absorb` are effectively identical in local metrics, as expected from the no-quant algebra. Structured `Rot-LM W4A4 + HLM K4V4` remains the strongest quantized Attention-layer setting; `W4A3/K4V3` is better than `W3A4/K3V4` in this local run.

### C5-refine: Attention-only PPL Full

| Item | Value |
| --- | --- |
| Run ID | `20260508_182156+0800_stage_c_refine_ppl` |
| Output dir | `outputs/stage_c_refine/20260508_182156+0800_stage_c_refine_ppl` |
| Device | `mps` |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |
| Records | 7 |
| Duration | 1058.468 seconds |

PPL results:

| Method | PPL | Interpretation |
| --- | ---: | --- |
| `fp16` | 8.048573 | baseline |
| `attn_identity_fp16` | 8.048573 | wrapper identity |
| `attn_kv_only_hlm_k4v4_oabsorb` | 8.204482 | KV-only HLM K4V4 |
| `attn_kv_only_hlm_k3v4_oabsorb` | 8.658396 | KV-only HLM K3V4 |
| `attn_rot_lm_w4a4_hlm_k4v4_oabsorb` | 8.614497 | structured W4A4 + K4V4 |
| `attn_rot_lm_w3a4_hlm_k3v4_oabsorb` | 11.355422 | structured W3A4 + K3V4 |
| `attn_rot_lm_w4a3_hlm_k4v3_oabsorb` | 9.387665 | structured W4A3 + K4V3 |

Conclusion: the C5-refine identity gate passes exactly, so the full Attention wrapper now preserves the Hugging Face TinyLlama path. Value rotation plus `o_proj` absorb removes the previous C5 PPL collapse: the best structured setting improves from the old C5 thousands-level PPL to single-digit PPL. KV-only `K4V4` is very close to FP16, and `K3V4` remains usable. Among structured low-bit settings, `W4A4/K4V4` is the clean baseline, while `W4A3/K4V3` is a more promising bit-reduction direction than `W3A4/K3V4` for Attention.
