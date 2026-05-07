# Stage A Weight-only Experiment Framework

本文档记录 A 线代码框架，目标对应 `第一阶段实验思路.md` 中的 Stage 1：

> 验证 Hadamard rotation + Lloyd-Max 是否能让 weight-only quantization 从 W4 推进到 W3/W2，同时保持可接受的 layer output error 和 PPL。

## 当前实验组

代码中用 `src/rotationquant/stage_a.py` 的 `STAGE_A_METHODS` 管理 A 线主实验组：

| Method | Weight domain | Quantizer | Bits | Compute interpretation |
| --- | --- | --- | --- | --- |
| `direct_absmax` | original weight | symmetric absmax | 4 / 3 / 2 | uniform integer-like, INT-GEMM friendly |
| `hadamard_absmax` | block-wise rotated weight | symmetric absmax | 4 / 3 / 2 | uniform integer-like, needs rotation handling |
| `hadamard_lm` | block-wise rotated weight | Gaussian Lloyd-Max | 4 / 3 / 2 | non-uniform codebook, fake quant only |

注意：`hadamard_lm` 返回 dequantized centroid value，第一阶段只验证数值质量，不解释成真实低比特 GEMM 加速。

## 目录结构

| Path | Role |
| --- | --- |
| `src/rotationquant/rotations.py` | FWHT、block flatten/padding、PolarQuant-style normalize + Hadamard forward/inverse |
| `src/rotationquant/quantizers.py` | symmetric absmax fake quant、Gaussian Lloyd-Max codebook 与 fake quant |
| `src/rotationquant/metrics.py` | relative MSE、cosine、SQNR、kurtosis 等 tensor-level 指标 |
| `src/rotationquant/modeling.py` | TinyLlama 本地路径、LLaMA q/k/v/o 与 FFN linear 层筛选、模型加载入口 |
| `src/rotationquant/stage_a.py` | A 线方法定义与单层权重量化记录生成 |
| `experiments/stage_a_weight_only.py` | A 线 tensor-level sweep 脚本，输出 JSONL/CSV |
| `configs/stage_a_tinyllama.yaml` | TinyLlama + A 线默认实验配置 |
| `scripts/run_stage_a_tensor_sweep.sh` | 一键运行 tensor-level sweep 的 shell 入口 |

## 已实现层级

当前可直接支撑：

1. TinyLlama 目标 Linear 层枚举；
2. A1 / A2 / A3 三类 weight-only fake quant；
3. W4 / W3 / W2 bit sweep；
4. tensor-level 指标表输出；
5. 每个记录包含 `quantizer_type` 和 `int_gemm_friendly`，避免把 Lloyd-Max fake quant 误解释为硬件加速。

## 后续补齐层级

后续需要在同一框架上继续补：

1. `layer_metrics.py`：固定 calibration activation，比较 `xW` 和 `xW_hat`；
2. `apply_weight_patch.py`：把 `W_hat` 写回模型副本，支撑 A16Wb PPL；
3. `ppl.py`：WikiText2 / C4 subset 困惑度评估；
4. 汇总脚本：对比 `Hadamard-LM W3` 与 `Hadamard-Absmax W4`。

## 运行入口

模型下载完成并安装实验依赖后，可运行：

```bash
scripts/run_stage_a_tensor_sweep.sh
```

输出位置：

```text
outputs/stage_a/tensor_metrics.jsonl
outputs/stage_a/tensor_metrics.csv
```
