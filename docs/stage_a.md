# Stage A Weight-only Experiments

本文档记录 Stage A 的实验代码框架、分组命名、运行入口、产物规范和正式结果。Stage A 只验证 weight-only fake quant 的数值质量和低比特可行性，不把 Lloyd-Max fake quant 解释为真实 INT GEMM 或推理加速。

## 实验目标

验证 Hadamard rotation + Lloyd-Max 是否能让 TinyLlama 的 weight-only quantization 从 W4 推进到 W3/W2，同时保持可接受的 tensor reconstruction quality 和 model-level PPL。

核心判断：

> `Hadamard-LM W3` 是否接近或优于 `Hadamard-Absmax W4`？

## 实验模型

| Item | Value |
| --- | --- |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Local path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Architecture | LLaMA, 22 layers, hidden size 2048, intermediate size 5632 |
| Attention | 32 query heads, 4 KV heads, head dim 64 |
| Target weights | 154 q/k/v/o + FFN Linear weights |
| Block size | 128 |

TinyLlama 的 Stage A 目标权重包括每层 attention 的 `q_proj` / `k_proj` / `v_proj` / `o_proj`，以及 FFN 的 `gate_proj` / `up_proj` / `down_proj`。所有目标权重都能按 128 block 对齐。

## 编号与命名

| Group | Method key | Bits | Weight domain | Quantizer | Compute interpretation |
| --- | --- | --- | --- | --- | --- |
| A0 | `fp16` | W16 | original weight | none | baseline |
| A1 | `direct_absmax` | W4 / W3 / W2 | original weight | symmetric absmax | uniform integer-like |
| A2 | `hadamard_absmax` | W4 / W3 / W2 | block-wise rotated weight | symmetric absmax | uniform integer-like, requires rotation handling |
| A3 | `hadamard_lm` | W4 / W3 / W2 | block-wise rotated weight | Gaussian Lloyd-Max | non-uniform codebook fake quant |

结果表中的显示名统一为 `Direct Absmax Wx`、`Hadamard Absmax Wx`、`Hadamard LM Wx`。其中 `hadamard_lm` 输出的是 dequantized centroid value，不是可直接进入 INT GEMM 的 integer tensor。

## 代码框架

| Path | Role |
| --- | --- |
| `src/rotationquant/rotations.py` | FWHT、block flatten/padding、Hadamard forward/inverse |
| `src/rotationquant/quantizers.py` | symmetric absmax fake quant、Gaussian Lloyd-Max codebook 与 fake quant |
| `src/rotationquant/metrics.py` | relative MSE、cosine、SQNR、kurtosis 等 tensor 指标 |
| `src/rotationquant/modeling.py` | TinyLlama 路径、模型加载、LLaMA Linear 层筛选 |
| `src/rotationquant/stage_a.py` | Stage A method registry 与单层权重量化记录生成 |
| `src/rotationquant/stage_a_model.py` | A16Wb model-level 原地权重替换 |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL |
| `src/rotationquant/run_metadata.py` | run id、时间、Git 状态、包版本、torch runtime metadata |
| `experiments/inspect_tinyllama_arch.py` | TinyLlama 结构与 target Linear shape 报告 |
| `experiments/stage_a_weight_only.py` | tensor-level weight sweep |
| `experiments/summarize_stage_a_tensor.py` | tensor sweep summary |
| `experiments/stage_a_ppl.py` | model-level A16Wb PPL |
| `scripts/run_stage_a_tensor_sweep.sh` | Stage A tensor sweep shell 入口 |
| `scripts/run_stage_a_ppl.sh` | Stage A PPL shell 入口 |

## 运行入口

Tensor-level sweep：

```bash
scripts/run_stage_a_tensor_sweep.sh
```

Tensor sweep 汇总：

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/summarize_stage_a_tensor.py \
  outputs/stage_a/<run_id>
```

Model-level A16Wb PPL：

```bash
scripts/run_stage_a_ppl.sh
```

正式 PPL 默认使用 WikiText2 raw test split、`max_samples=512`、`sequence_length=2048`、`stride=2048`、`device=mps`。

## 产物规范

Stage A 产物写入 `outputs/stage_a/<run_id>/`，目录名中的 `<run_id>` 必须与 `run_metadata.json.run_id` 一致。

Tensor sweep 输出：

```text
tensor_metrics.jsonl
tensor_metrics.csv
summary_by_method.csv
summary_by_group.csv
summary_by_projection.csv
summary.md
run_metadata.json
```

PPL 输出：

```text
ppl_runs.jsonl
ppl.csv
run_metadata.json
```

所有正式 run 需要在 `docs/experiment_runs.md` 中记录 run id、output dir、Git commit / dirty status、核心表格和结论。

## 实验结果

### A1-A3 Tensor Sweep

| Item | Value |
| --- | --- |
| Run ID | `20260507_192029+0800_stage_a_tensor_sweep` |
| Output dir | `outputs/stage_a/20260507_192029+0800_stage_a_tensor_sweep` |
| Records | 1386 |
| Target weights | 154 |
| Methods | `direct_absmax`, `hadamard_absmax`, `hadamard_lm` |
| Bits | W4 / W3 / W2 |

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

Tensor-level 结论：`Hadamard LM W3` 的 relative MSE 和 cosine 优于 `Hadamard Absmax W4`；`Hadamard LM W2` 的 tensor reconstruction 仍可观察，但需要由 PPL 判断是否可用。

### A0-A3 Model-level PPL

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

## 结论

1. `direct_absmax` 不适合作为 TinyLlama weight-only 低比特主方法，W4/W3/W2 的 tensor error 和 PPL 都严重恶化。
2. `hadamard_absmax` 在 W4 上可用，但 W3/W2 在 model-level PPL 上崩溃。
3. `hadamard_lm` 是 Stage A 最有效的组合：W4 接近 FP16，W3 明显优于 `Hadamard Absmax W4`，满足“用非均匀 codebook 换取更低位宽”的核心判断。
4. `Hadamard LM W2` 是当前失败边界；tensor 指标较好不能直接推出 model-level 可用。
