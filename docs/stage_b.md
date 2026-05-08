# Stage B QuaRot-style W/A Fake Quantization

本文档记录 Stage B 的实验代码框架、分组命名、运行入口、产物规范和正式结果。Stage B 只解释 W/A fake quant 的数值质量和低比特可行性，不解释为真实 low-bit GEMM 加速。

## 实验目标

验证在旋转域 W/A fake quant 中，Lloyd-Max 是否能支持 `W3A4` / `W4A3`，而不仅是在 `W4A4` 同 bit 下略好。

核心判断：

> `Rot-LM W3A4` 或 `Rot-LM W4A3` 是否接近或优于 `Rot-Absmax W4A4`？

## 实验模型

| Item | Value |
| --- | --- |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Local path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Architecture | LLaMA, 22 layers, hidden size 2048, intermediate size 5632 |
| Attention | 32 query heads, 4 KV heads, head dim 64 |
| FFN naming | 文档统一称 FFN；Hugging Face 模块路径仍叫 `mlp` |
| W/A block size | 128 |

Stage B 使用 block-wise orthonormal Hadamard along last dimension。Linear 权重 shape 为 `[out_features, in_features]`，权重旋转沿最后一维，即 input/features dimension。

## 编号与命名

| Stage | Scope | Output root |
| --- | --- | --- |
| B1 | activation tensor-level quantization | `outputs/stage_b/<run_id>/` |
| B2 | local Linear / FFN W/A fake quant | `outputs/stage_b/<run_id>/` |
| B3 | FFN-only model-level PPL | `outputs/stage_b/<run_id>/` |

### B1 Activation

| Method key | Bits | Description |
| --- | --- | --- |
| `direct_absmax` | A4 / A3 / A2 | original activation + symmetric absmax |
| `rot_absmax` | A4 / A3 / A2 | block Hadamard + symmetric absmax |
| `rot_lm` | A4 / A3 / A2 | block Hadamard + RMS-normalized Lloyd-Max |

B1 采集对象：

| Site | Position |
| --- | --- |
| `attn_input` | `input_layernorm` output |
| `ffn_input` | `post_attention_layernorm` output |
| `ffn_intermediate` | `SiLU(gate_proj(x)) * up_proj(x)` |
| `q_proj_out` | `q_proj` output |
| `k_proj_out` | `k_proj` output |
| `v_proj_out` | `v_proj` output |

### B2 Local Linear / FFN

Linear method keys：

| Method key | Bits | Description |
| --- | --- | --- |
| `direct_absmax_w4a4` | W4A4 | baseline W/A fake quant |
| `rot_absmax_w4a4` | W4A4 | QuaRot-style local rotation + uniform quant |
| `rot_lm_w4a4` | W4A4 | rotated Lloyd-Max same-bit comparison |
| `rot_lm_w3a4` | W3A4 | weight bit reduction |
| `rot_lm_w4a3` | W4A3 | activation bit reduction |
| `rot_lm_w3a3` | W3A3 | aggressive W/A combination |
| `rot_lm_w2a4` | W2A4 | failure boundary |

FFN method keys：

| Method key | Bits | Description |
| --- | --- | --- |
| `ffn_fp16` | FP16 | baseline |
| `ffn_direct_absmax_w4a4` | W4A4 | baseline FFN W/A fake quant |
| `ffn_rot_absmax_w4a4` | W4A4 | rotated uniform FFN fake quant |
| `ffn_rot_lm_w4a4` | W4A4 | rotated Lloyd-Max same-bit comparison |
| `ffn_rot_lm_w3a4` | W3A4 | weight bit reduction |
| `ffn_rot_lm_w4a3` | W4A3 | activation bit reduction |
| `ffn_rot_lm_w3a3` | W3A3 | aggressive W/A combination |

### B3 FFN-only PPL

Model-level method keys：

| Method key | Interpretation |
| --- | --- |
| `fp16` | baseline |
| `ffn_direct_absmax_w4a4` | baseline FFN W/A fake quant |
| `ffn_rot_absmax_w4a4` | rotated uniform FFN fake quant |
| `ffn_rot_lm_w4a4` | rotated Lloyd-Max same-bit comparison |
| `ffn_rot_lm_w3a4` | weight bit reduction |
| `ffn_rot_lm_w4a3` | activation bit reduction |

B3 只替换 FFN，不修改 Attention。

## 代码框架

| Path | Role |
| --- | --- |
| `src/rotationquant/stage_b.py` | Stage B method registry、W/A bit spec、block Hadamard、Linear/FFN fake quant、FFN-only model wrapper |
| `src/rotationquant/activation_capture.py` | TinyLlama activation hook 与 local Linear/FFN input-output capture |
| `src/rotationquant/rotations.py` | FWHT |
| `src/rotationquant/quantizers.py` | symmetric absmax 与 Gaussian Lloyd-Max fake quant |
| `src/rotationquant/metrics.py` | tensor metrics、distribution metrics、outlier ratio |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL |
| `src/rotationquant/run_metadata.py` | run id、时间、Git 状态、包版本、torch runtime metadata |
| `experiments/stage_b_activation.py` | B1 activation tensor-level sweep |
| `experiments/stage_b_local.py` | B2 Linear / FFN local output error |
| `experiments/stage_b_ppl.py` | B3 FFN-only model-level PPL |
| `experiments/summarize_stage_b.py` | Stage B summary |
| `scripts/run_stage_b_activation.sh` | B1 shell 入口 |
| `scripts/run_stage_b_local.sh` | B2 shell 入口 |
| `scripts/run_stage_b_ppl.sh` | B3 shell 入口 |

## 运行入口

B1 activation：

```bash
scripts/run_stage_b_activation.sh
```

B2 local Linear / FFN：

```bash
scripts/run_stage_b_local.sh
```

B3 FFN-only PPL：

```bash
scripts/run_stage_b_ppl.sh
```

汇总任意 Stage B run：

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/summarize_stage_b.py \
  outputs/stage_b/<run_id>
```

正式 B3 PPL 默认使用 WikiText2 raw test split、`max_samples=512`、`sequence_length=2048`、`stride=2048`、`device=mps`。

## 产物规范

Stage B 产物写入 `outputs/stage_b/<run_id>/`，目录名中的 `<run_id>` 必须与 `run_metadata.json.run_id` 一致。

B1 输出：

```text
activation_metrics.jsonl
activation_metrics.csv
histograms.json
summary_by_tensor.csv
summary.md
run_metadata.json
```

B2 输出：

```text
linear_metrics.jsonl
linear_metrics.csv
ffn_metrics.jsonl
ffn_metrics.csv
summary_linear_by_method.csv
summary_ffn_by_method.csv
summary.md
run_metadata.json
```

B3 输出：

```text
ppl_runs.jsonl
ppl.csv
run_metadata.json
```

所有正式 run 需要在 `docs/experiment_runs.md` 中记录 run id、output dir、Git commit / dirty status、核心表格和结论。

## 实验结果

### B1 Activation Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_090340+0800_stage_b_activation` |
| Output dir | `outputs/stage_b/20260508_090340+0800_stage_b_activation` |
| Device | `mps` |
| Scope | all 22 layers, WikiText2 test, max samples 32, sequence length 512 |
| Records | 1188 |
| Captured activations | 132 |
| Duration | 35.697 seconds |

Mean metrics across activation sites:

| Method | Bits | Relative MSE | Cosine | SQNR dB |
| --- | ---: | ---: | ---: | ---: |
| Direct Absmax | 4 | 0.327630 | 0.820784 | 6.579584 |
| Rot Absmax | 4 | 0.075469 | 0.963145 | 12.490349 |
| Rot LM | 4 | 0.009297 | 0.995561 | 20.390117 |
| Rot LM | 3 | 0.034815 | 0.983110 | 14.620511 |
| Rot LM | 2 | 0.120526 | 0.940698 | 9.243337 |

Core comparison:

| Site | Rot-Absmax A4 MSE | Rot-LM A3 MSE |
| --- | ---: | ---: |
| `attn_input` | 0.049948 | 0.032012 |
| `ffn_input` | 0.046747 | 0.033693 |
| `ffn_intermediate` | 0.214194 | 0.034427 |
| `k_proj_out` | 0.029189 | 0.032644 |
| `q_proj_out` | 0.060537 | 0.042478 |
| `v_proj_out` | 0.052202 | 0.033637 |

### B2 Local Linear / FFN Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_090430+0800_stage_b_local` |
| Output dir | `outputs/stage_b/20260508_090430+0800_stage_b_local` |
| Device | `mps` |
| Scope | all 154 Linear layers and 22 FFN modules, WikiText2 test, max samples 8, sequence length 128 |
| Linear records | 1078 |
| FFN records | 154 |
| Duration | 166.134 seconds |

Linear mean metrics:

| Method | Relative MSE | Cosine | SQNR dB |
| --- | ---: | ---: | ---: |
| Direct Absmax W4A4 | 0.613288 | 0.633419 | 3.553233 |
| Rot Absmax W4A4 | 0.228609 | 0.901790 | 9.529216 |
| Rot LM W4A4 | 0.015854 | 0.992299 | 20.221924 |
| Rot LM W3A4 | 0.037209 | 0.982336 | 16.415885 |
| Rot LM W4A3 | 0.037381 | 0.982784 | 16.350577 |
| Rot LM W3A3 | 0.058215 | 0.973365 | 14.323914 |
| Rot LM W2A4 | 0.109925 | 0.950351 | 11.338703 |

FFN mean metrics:

| Method | Relative MSE | Cosine | SQNR dB |
| --- | ---: | ---: | ---: |
| FFN Direct Absmax W4A4 | 0.969996 | 0.244356 | 0.800832 |
| FFN Rot Absmax W4A4 | 0.477501 | 0.810050 | 3.600135 |
| FFN Rot LM W4A4 | 0.031212 | 0.984782 | 15.330151 |
| FFN Rot LM W3A4 | 0.073063 | 0.964408 | 11.562464 |
| FFN Rot LM W4A3 | 0.079189 | 0.964333 | 11.106795 |
| FFN Rot LM W3A3 | 0.121197 | 0.944023 | 9.253756 |

### B3 FFN-only PPL Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_090737+0800_stage_b_ppl` |
| Output dir | `outputs/stage_b/20260508_090737+0800_stage_b_ppl` |
| Device | `mps` |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |
| Records | 6 |
| Duration | 928.889 seconds |

| Method | PPL |
| --- | ---: |
| FP16 | 8.048573 |
| FFN Direct Absmax W4A4 | 46090.420308 |
| FFN Rot Absmax W4A4 | 2733.085720 |
| FFN Rot LM W4A4 | 8.586876 |
| FFN Rot LM W3A4 | 9.978913 |
| FFN Rot LM W4A3 | 9.352936 |

## 结论

1. B1 中 `Rot-LM A3` 平均优于 `Rot-Absmax A4`，说明 activation 在 rotated domain 中有从 A4 推向 A3 的潜力。
2. B2 中 `Rot-LM W3A4` 和 `Rot-LM W4A3` 在 Linear 和完整 FFN local output 上都明显优于 `Rot-Absmax W4A4`。
3. B3 中 FFN-only model-level PPL 与 local 结果一致：`FFN Rot LM W3A4` PPL 为 `9.978913`，`FFN Rot LM W4A3` PPL 为 `9.352936`，都接近 FP16 `8.048573`。
4. `FFN Rot Absmax W4A4` PPL 为 `2733.085720`，说明 uniform rotated W4A4 不足以支撑当前 FFN-only W/A fake quant。
