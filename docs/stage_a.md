# Stage A Weight-only Experiments

本文档记录 A 阶段的代码框架、运行入口、实验产物和当前结论。目标对应 `第一阶段实验思路.md` 中的 Stage 1：

> 验证 Hadamard rotation + Lloyd-Max 是否能让 weight-only quantization 从 W4 推进到 W3/W2，同时保持可接受的 tensor reconstruction quality 和 model-level PPL。

注意：`hadamard_lm` 返回 dequantized centroid value。第一阶段只验证数值质量和低比特可行性，不把 Lloyd-Max fake quant 解释成真实 INT GEMM 或推理加速。

## 实验模型

| Item | Value |
| --- | --- |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Local path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Architecture | LLaMA, 22 layers, hidden size 2048, intermediate size 5632 |
| Attention | 32 query heads, 4 KV heads, head dim 64 |
| Stage A target weights | 154 q/k/v/o + FFN Linear weights |
| Block size | 128 |

TinyLlama 的 `k_proj` / `v_proj` 使用 GQA，shape 为 `[256, 2048]`；`q_proj` / `o_proj` 为 `[2048, 2048]`；FFN 的 `gate_proj` / `up_proj` 为 `[5632, 2048]`，`down_proj` 为 `[2048, 5632]`。所有 Stage A 目标权重都能按 128 block 对齐。

## 实验组

代码中用 `src/rotationquant/stage_a.py` 的 `STAGE_A_METHODS` 管理 A 线主实验组：

| Method | Weight domain | Quantizer | Bits | Compute interpretation |
| --- | --- | --- | --- | --- |
| `direct_absmax` | original weight | symmetric absmax | 4 / 3 / 2 | uniform integer-like, INT-GEMM friendly |
| `hadamard_absmax` | block-wise rotated weight | symmetric absmax | 4 / 3 / 2 | uniform integer-like, needs rotation handling |
| `hadamard_lm` | block-wise rotated weight | Gaussian Lloyd-Max | 4 / 3 / 2 | non-uniform codebook, fake quant only |

核心判断不是“同 bit 谁更好”，而是：

> `Hadamard-LM W3` 是否接近或优于 `Hadamard-Absmax W4`？

## 代码框架

| Path | Role |
| --- | --- |
| `src/rotationquant/rotations.py` | FWHT、block flatten/padding、PolarQuant-style normalize + Hadamard forward/inverse |
| `src/rotationquant/quantizers.py` | symmetric absmax fake quant、Gaussian Lloyd-Max codebook 与 fake quant |
| `src/rotationquant/metrics.py` | relative MSE、cosine、SQNR、kurtosis 等 tensor-level 指标 |
| `src/rotationquant/modeling.py` | TinyLlama 本地路径、LLaMA q/k/v/o 与 FFN Linear 层筛选、模型加载入口 |
| `src/rotationquant/stage_a.py` | A 线方法定义与单层权重量化记录生成 |
| `src/rotationquant/stage_a_model.py` | A16Wb model-level 原地权重替换工具 |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL 评估工具 |
| `src/rotationquant/run_metadata.py` | 写入实验时间、Git commit、包版本、运行参数和 torch runtime metadata |
| `experiments/inspect_tinyllama_arch.py` | 从 TinyLlama config 生成 A 线目标层 shape 和 block alignment 报告 |
| `experiments/stage_a_weight_only.py` | A 线 tensor-level sweep 脚本，输出 JSONL/CSV |
| `experiments/summarize_stage_a_tensor.py` | 汇总 tensor sweep 的 method/group/projection 统计表 |
| `experiments/stage_a_ppl.py` | A 线 model-level PPL 脚本，逐方法重载 FP checkpoint 后量化 |
| `configs/stage_a_tinyllama.yaml` | TinyLlama + A 线默认实验配置 |
| `scripts/run_stage_a_tensor_sweep.sh` | 一键运行 tensor-level sweep 的 shell 入口 |
| `scripts/run_stage_a_ppl.sh` | 一键运行 model-level PPL grid 的 shell 入口 |

## 运行入口

Tensor-level sweep：

```bash
scripts/run_stage_a_tensor_sweep.sh
```

汇总 tensor-level sweep：

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/summarize_stage_a_tensor.py \
  outputs/stage_a/<run_id>
```

Model-level A16Wb PPL：

```bash
scripts/run_stage_a_ppl.sh
```

当前 Codex 沙箱内无法枚举 MPS/Metal GPU，但授权外部命令可以使用 MPS：

```text
inside sandbox: torch.backends.mps.is_available() == False
outside sandbox: torch.backends.mps.is_available() == True, device_count == 1
```

因此需要用 GPU 跑 model-level PPL 时，应授权外部执行，并传入：

```bash
--device mps
```

## 产物规范

A 阶段脚本会在 `outputs/stage_a/<run_id>/` 下写入数据。目录名中的 `<run_id>` 与 `run_metadata.json` 中的 `run_id` 字段保持一致。

Tensor sweep 输出：

```text
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/tensor_metrics.jsonl
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/tensor_metrics.csv
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/summary_by_method.csv
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/summary_by_group.csv
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/summary_by_projection.csv
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/summary.md
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_tensor_sweep>/run_metadata.json
```

PPL 输出：

```text
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_ppl>/ppl_runs.jsonl
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_ppl>/ppl.csv
outputs/stage_a/<YYYYMMDD_HHMMSS+0800_stage_a_ppl>/run_metadata.json
```

## 实验结果

### Tensor Sweep

| Item | Value |
| --- | --- |
| Run ID | `20260507_192029+0800_stage_a_tensor_sweep` |
| Output dir | `outputs/stage_a/20260507_192029+0800_stage_a_tensor_sweep` |
| Records | 1386 |
| Target weights | 154 |
| Methods | `direct_absmax`, `hadamard_absmax`, `hadamard_lm` |
| Bits | W4 / W3 / W2 |

Method-level mean metrics:

| Method | Bits | Relative MSE | Cosine | SQNR dB |
| --- | ---: | ---: | ---: | ---: |
| Direct Absmax | 4 | 0.537248 | 0.690929 | 3.935982 |
| Direct Absmax | 3 | 0.844156 | 0.341958 | 0.951054 |
| Direct Absmax | 2 | 0.995216 | 0.070846 | 0.020994 |
| Hadamard Absmax | 4 | 0.042540 | 0.980618 | 13.747699 |
| Hadamard Absmax | 3 | 0.231185 | 0.902234 | 6.394859 |
| Hadamard Absmax | 2 | 0.959623 | 0.311654 | 0.179672 |
| Hadamard LM | 4 | 0.009364 | 0.996558 | 20.286120 |
| Hadamard LM | 3 | 0.034078 | 0.984053 | 14.675942 |
| Hadamard LM | 2 | 0.116225 | 0.941289 | 9.347435 |

Tensor-level 结论：`Hadamard-LM W3` 的 relative MSE 和 cosine 已经优于 `Hadamard-Absmax W4`。`Hadamard-LM W2` 在 tensor reconstruction 上仍明显优于 `Hadamard-Absmax W3`，但是否可用需要看 model-level PPL。

### PPL Smoke Test

| Item | Value |
| --- | --- |
| Run ID | `20260507_192603+0800_stage_a_ppl` |
| Output dir | `outputs/stage_a/20260507_192603+0800_stage_a_ppl` |
| Dataset | WikiText2 raw test split |
| Max samples | 64 |
| Sequence length / stride | 512 / 512 |

| Method | Bits | PPL |
| --- | ---: | ---: |
| FP16 | 16 | 10.549959 |
| Hadamard Absmax | 4 | 20.447154 |
| Hadamard Absmax | 3 | 25289.088084 |
| Hadamard LM | 4 | 11.561565 |
| Hadamard LM | 3 | 17.966018 |

Smoke test 结论：小样本验证中 `Hadamard-LM W3` 已经优于 `Hadamard-Absmax W4`，值得跑完整 PPL grid。

### Full PPL Grid

| Item | Value |
| --- | --- |
| Run ID | `20260507_193316+0800_stage_a_ppl` |
| Output dir | `outputs/stage_a/20260507_193316+0800_stage_a_ppl` |
| Device | `mps` |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |
| Records | 10 |
| Duration | 964.826 seconds |

| Method | Bits | PPL |
| --- | ---: | ---: |
| FP16 | 16 | 8.048573 |
| Direct Absmax | 4 | 310770.672475 |
| Direct Absmax | 3 | 54040.991034 |
| Direct Absmax | 2 | 29191.459708 |
| Hadamard Absmax | 4 | 15.395369 |
| Hadamard Absmax | 3 | 18801.124177 |
| Hadamard Absmax | 2 | 186900.274024 |
| Hadamard LM | 4 | 8.627246 |
| Hadamard LM | 3 | 12.992705 |
| Hadamard LM | 2 | 19646.836571 |

Full grid 结论：`Hadamard-LM W3` 的 PPL 明显优于 `Hadamard-Absmax W4`，说明 Lloyd-Max 在 Hadamard rotation 后确实能把 weight-only 的数值可用位宽从 W4 推向 W3。`Hadamard-LM W2` 在 tensor reconstruction 上仍可看，但 model-level PPL 已经崩溃，应作为失败边界。

## 当前总结

1. Direct absmax 不适合作为 TinyLlama weight-only 低比特主方法。原始权重中的 outlier 会让 W4/W3/W2 的 tensor error 和 PPL 都严重恶化。
2. Hadamard rotation 对 uniform absmax 有明显帮助，但只到 W4 比较可用；W3/W2 在 model-level PPL 上崩溃。
3. Hadamard-LM 是当前 A 阶段最有价值的组合。W4 接近 FP16，W3 优于 Hadamard-Absmax W4，满足第一阶段“用非均匀 codebook 换取更低位宽”的核心判断。
4. Lloyd-Max 的结果只能解释为数值质量和低比特可行性。后续若讨论加速，需要另行设计 LUT/codebook-aware MAC、bit packing 或专用 kernel。

## 后续工作

1. 补充 layer output error：固定 calibration activation，比较 `xW` 和 `xW_hat`。
2. 扩大 PPL 验证：增加 WikiText2 样本数，或加入 C4 子集确认 W3 结论稳定性。
3. 进入 B 阶段：把 activation quantization 和 FFN fake quant 统一接到当前 quantizer/rotation 组件上。
