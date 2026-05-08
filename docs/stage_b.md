# Stage B QuaRot-style W/A Fake Quantization

本文档记录 B 阶段的代码框架、运行入口、产物规范和结果记录模板。目标对应 `第一阶段实验思路.md` 中的 B 线：

> 验证在旋转域 W/A fake quant 中，Lloyd-Max 是否能支持 W3A4 / W4A3，而不仅仅是在 W4A4 同 bit 下略好。

Stage B 首版实现 B1、B2、B4 三层实验；B3 Structured QuaRot-FFN 暂不实现，只记录设计和进入条件。Stage B 仍只解释数值质量和低比特可行性，不把 Lloyd-Max fake quant 解释成真实推理加速。

## 实验模型

| Item | Value |
| --- | --- |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Local path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Architecture | LLaMA, 22 layers, hidden size 2048, intermediate size 5632 |
| Attention | 32 query heads, 4 KV heads, head dim 64 |
| FFN naming | 文档统一称 FFN；Hugging Face 代码路径仍叫 `mlp` |
| Block size | 128 |

Stage B 默认使用 block-wise orthonormal Hadamard along last dimension，不使用 Stage A 的 flatten 全权重 polar rotation。

## 实验组

### B1 Activation

| Method | Bits | Compute interpretation |
| --- | --- | --- |
| `direct_absmax` | A4 / A3 / A2 | uniform activation fake quant |
| `rot_absmax` | A4 / A3 / A2 | block Hadamard + uniform fake quant |
| `rot_lm` | A4 / A3 / A2 | block Hadamard + RMS-normalized Lloyd-Max fake quant |

采集对象：

| Site | 位置 |
| --- | --- |
| `attn_input` | `input_layernorm` output |
| `ffn_input` | `post_attention_layernorm` output |
| `ffn_intermediate` | `SiLU(gate_proj(x)) * up_proj(x)` |
| `q_proj_out` | `q_proj` output |
| `k_proj_out` | `k_proj` output |
| `v_proj_out` | `v_proj` output |

核心判断：

> `Rot-LM A3` 是否接近或优于 `Rot-Absmax A4`？

### B2 Local Linear / FFN

Linear 实验组：

| Method key | Bits | 说明 |
| --- | --- | --- |
| `direct_absmax_w4a4` | W4A4 | 普通 W/A fake quant baseline |
| `rot_absmax_w4a4` | W4A4 | QuaRot-style local rotation + uniform |
| `rot_lm_w4a4` | W4A4 | 同 bit 非均匀 codebook 对比 |
| `rot_lm_w3a4` | W3A4 | weight 降位宽 |
| `rot_lm_w4a3` | W4A3 | activation 降位宽 |
| `rot_lm_w3a3` | W3A3 | 极限组合 |
| `rot_lm_w2a4` | W2A4 | 失败边界 |

FFN 实验组：

| Method key | Bits | 说明 |
| --- | --- | --- |
| `ffn_fp16` | FP16 | baseline |
| `ffn_direct_absmax_w4a4` | W4A4 | 普通联合 fake quant |
| `ffn_rot_absmax_w4a4` | W4A4 | rotation + uniform |
| `ffn_rot_lm_w4a4` | W4A4 | 同 bit 对比 |
| `ffn_rot_lm_w3a4` | W3A4 | weight 降位宽 |
| `ffn_rot_lm_w4a3` | W4A3 | activation 降位宽 |
| `ffn_rot_lm_w3a3` | W3A3 | 极限组合 |

核心判断：

> `Rot-LM W3A4` 是否接近或优于 `Rot-Absmax W4A4`？非线性和 gate/up/down 结构是否放大量化误差？

### B4 FFN-only PPL

Model-level 只替换 FFN，不修改 Attention。

| Method key | PPL | Interpretation |
| --- | ---: | --- |
| `fp16` | | baseline |
| `ffn_direct_absmax_w4a4` | | uniform fake quant |
| `ffn_rot_absmax_w4a4` | | rotation + uniform |
| `ffn_rot_lm_w4a4` | | non-uniform same-bit comparison |
| `ffn_rot_lm_w3a4` | | weight bit reduction |
| `ffn_rot_lm_w4a3` | | activation bit reduction |

核心判断：

> `FFN Rot-LM W3A4` 或 `FFN Rot-LM W4A3` 的 PPL 是否接近或优于 `FFN Rot-Absmax W4A4`？

## 代码框架

| Path | Role |
| --- | --- |
| `src/rotationquant/stage_b.py` | Stage B method registry、W/A bit spec、block Hadamard、Linear/FFN fake quant、FFN-only model wrapper |
| `src/rotationquant/activation_capture.py` | TinyLlama activation hook 与 local Linear/FFN input-output capture |
| `src/rotationquant/rotations.py` | 复用 FWHT |
| `src/rotationquant/quantizers.py` | 复用 symmetric absmax 与 Gaussian Lloyd-Max codebook |
| `src/rotationquant/metrics.py` | tensor metrics、distribution metrics、outlier ratio |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL |
| `src/rotationquant/run_metadata.py` | 统一记录 run id、时间、Git 状态、包版本和 torch runtime |
| `experiments/stage_b_activation.py` | B1 activation tensor-level sweep |
| `experiments/stage_b_local.py` | B2 Linear / FFN local output error |
| `experiments/stage_b_ppl.py` | B4 FFN-only model-level PPL |
| `experiments/summarize_stage_b.py` | 汇总 B1/B2/B4 CSV 到 summary |
| `scripts/run_stage_b_activation.sh` | B1 shell 入口 |
| `scripts/run_stage_b_local.sh` | B2 shell 入口 |
| `scripts/run_stage_b_ppl.sh` | B4 shell 入口 |

## 运行入口

B1 activation：

```bash
scripts/run_stage_b_activation.sh
```

B2 local Linear / FFN：

```bash
scripts/run_stage_b_local.sh
```

B4 FFN-only PPL：

```bash
scripts/run_stage_b_ppl.sh
```

汇总任意 Stage B run：

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/summarize_stage_b.py \
  outputs/stage_b/<run_id>
```

当前 Codex 沙箱内无法枚举 MPS/Metal GPU；需要用 GPU 跑 B4 PPL 时，应授权外部执行，并传入：

```bash
--device mps
```

## 产物规范

Stage B 脚本会在 `outputs/stage_b/<run_id>/` 下写入数据。目录名中的 `<run_id>` 与 `run_metadata.json` 中的 `run_id` 字段保持一致。

B1 输出：

```text
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_activation>/activation_metrics.jsonl
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_activation>/activation_metrics.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_activation>/histograms.json
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_activation>/summary_by_tensor.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_activation>/summary.md
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_activation>/run_metadata.json
```

B2 输出：

```text
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/linear_metrics.jsonl
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/linear_metrics.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/ffn_metrics.jsonl
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/ffn_metrics.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/summary_linear_by_method.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/summary_ffn_by_method.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/summary.md
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_local>/run_metadata.json
```

B4 输出：

```text
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_ppl>/ppl_runs.jsonl
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_ppl>/ppl.csv
outputs/stage_b/<YYYYMMDD_HHMMSS+0800_stage_b_ppl>/run_metadata.json
```

每次正式 Stage B 运行后，在 `docs/experiment_runs.md` 追加 run id、output dir、Git commit、dirty status、核心表格和结论。

## B3 Structured QuaRot-FFN 后续设计

B3 暂不在首版代码中实现。进入条件：

1. B2 local FFN 中 `Rot-LM W3A4` 接近或优于 `Rot-Absmax W4A4`；
2. B4 FFN-only PPL 中 `FFN Rot-LM W3A4` 仍保持优势；
3. 需要把结论进一步靠近 QuaRot 的结构级 computational invariance。

后续 B3 实现要按 QuaRot FFN 思路做结构适配：

1. RMSNorm scale 融入 gate/up；
2. hidden state 进入旋转域；
3. gate/up 权重吸收对应旋转；
4. down_proj 前插入 Hadamard；
5. down_proj 权重吸收对应旋转；
6. 再把普通量化器替换为 Lloyd-Max fake quant。

这时才称为 **QuaRot-LM FFN**。

## 当前结果模板

### B1 Activation Quantization Error

| Site | Bits | Direct absmax | Rot absmax | Rot LM |
| --- | ---: | ---: | ---: | ---: |
| `attn_input` | 4 | | | |
| `attn_input` | 3 | | | |
| `ffn_input` | 4 | | | |
| `ffn_input` | 3 | | | |
| `ffn_intermediate` | 4 | | | |
| `ffn_intermediate` | 3 | | | |
| `q_proj_out` | 4 | | | |
| `k_proj_out` | 4 | | | |
| `v_proj_out` | 4 | | | |

### B2 Local W/A Joint Fake Quant

| Method | W4A4 | W3A4 | W4A3 | W3A3 | Compute interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct absmax | | | | | uniform fake quant |
| Rot absmax | | | | | rotation + uniform fake quant |
| Rot LM | | | | | non-uniform codebook fake quant |

### B4 Model-level FFN-only PPL

| Method | PPL | Interpretation |
| --- | ---: | --- |
| FP16 | | baseline |
| FFN Direct-Absmax W4A4 | | uniform fake quant |
| FFN Rot-Absmax W4A4 | | rotation + uniform |
| FFN Rot-LM W4A4 | | non-uniform same-bit comparison |
| FFN Rot-LM W3A4 | | weight bit reduction |
| FFN Rot-LM W4A3 | | activation bit reduction |

## 当前总结

Stage B 代码框架已对齐 A 阶段的 run metadata 和输出目录规范。当前已完成低成本 smoke：

1. B1 activation smoke：确认 activation hook、histogram、CSV/JSONL、summary 和 metadata 均能落盘。
2. B2 local smoke：确认 Linear / FFN local fake quant 路径均能落盘。
3. B4 PPL smoke：确认 FFN-only wrapper 能在 MPS 上完成 FP16 + `ffn_rot_lm_w4a4` PPL。

B4 wrapper 会在初始化时预量化 FFN 权重，forward 中只量化 activation。`rot_lm` 的模型级 smoke 仍然较慢，因为 22 个 FFN 的 gate/up/down 权重需要做一次 Lloyd-Max codebook fake quant；正式 full PPL 运行应预留更长时间。

## 正式全量结果

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

B1 conclusion: `Rot-LM A3` is better than `Rot-Absmax A4` on average and on most sites. The only observed exception is `k_proj_out`, where `Rot-Absmax A4` is slightly better.

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

B2 conclusion: both local Linear and full FFN results strongly support the Stage B hypothesis. `Rot-LM W3A4` and `Rot-LM W4A3` are much better than `Rot-Absmax W4A4`, even after the FFN gate/up/down nonlinearity.

### B4 FFN-only PPL Full Run

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

PPL results:

| Method | PPL |
| --- | ---: |
| FP16 | 8.048573 |
| FFN Direct Absmax W4A4 | 46090.420308 |
| FFN Rot Absmax W4A4 | 2733.085720 |
| FFN Rot LM W4A4 | 8.586876 |
| FFN Rot LM W3A4 | 9.978913 |
| FFN Rot LM W4A3 | 9.352936 |

B4 conclusion: `FFN Rot-LM W3A4` and `FFN Rot-LM W4A3` are both close to FP16 and dramatically better than `FFN Rot-Absmax W4A4`. This supports the B-line claim that Lloyd-Max in the rotated domain can reduce usable W/A bit-width, but it remains fake quant numerical evidence rather than hardware speed evidence.

## 当前结论

1. B1: `Rot-LM A3` is generally better than `Rot-Absmax A4`, so activation bit-width can likely move from A4 toward A3 in the rotated domain.
2. B2: `Rot-LM W3A4` and `Rot-LM W4A3` remain much better than `Rot-Absmax W4A4` for both individual Linear layers and full FFN output.
3. B4: FFN-only model-level PPL confirms the local trend. `FFN Rot-LM W3A4` PPL is `9.978913`, and `FFN Rot-LM W4A3` PPL is `9.352936`, both close to FP16 `8.048573`.
4. Uniform rotated W4A4 is not enough for FFN-only W/A fake quant in this implementation; `FFN Rot-Absmax W4A4` PPL is `2733.085720`.
5. Stage B passes its first-phase success criterion. The next reasonable step is B3 Structured QuaRot-FFN or moving to C-line attention/KV experiments, depending on whether the priority is FFN structural invariance or attention/KV behavior.
