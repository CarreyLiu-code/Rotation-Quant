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
| `src/rotationquant/stage_a_model.py` | A16Wb model-level 原地权重替换工具 |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL 评估工具 |
| `src/rotationquant/run_metadata.py` | 写入实验时间、Git commit、包版本、运行参数等 metadata |
| `experiments/stage_a_weight_only.py` | A 线 tensor-level sweep 脚本，输出 JSONL/CSV |
| `experiments/stage_a_ppl.py` | A 线 model-level PPL 脚本，逐方法重载 FP checkpoint 后量化 |
| `experiments/inspect_tinyllama_arch.py` | 从 TinyLlama config 生成 A 线目标层 shape 和 block alignment 报告 |
| `configs/stage_a_tinyllama.yaml` | TinyLlama + A 线默认实验配置 |
| `scripts/run_stage_a_tensor_sweep.sh` | 一键运行 tensor-level sweep 的 shell 入口 |

## 已实现层级

当前可直接支撑：

1. TinyLlama 目标 Linear 层枚举；
2. A1 / A2 / A3 三类 weight-only fake quant；
3. W4 / W3 / W2 bit sweep；
4. tensor-level 指标表输出；
5. 每个记录包含 `quantizer_type` 和 `int_gemm_friendly`，避免把 Lloyd-Max fake quant 误解释为硬件加速。
6. A16Wb PPL 入口：每个 method/bits 组合重新加载 FP 模型，再只替换目标 Linear 权重。
7. TinyLlama 架构预检查：确认 GQA 下 `k_proj/v_proj` 的 shape 和 128-block 对齐。
8. 实验产物记录：A 线脚本会在 `outputs/stage_a/<run_id>/` 下写 `run_metadata.json`，目录名和 metadata 里的 `run_id` 对齐。

## 后续补齐层级

后续需要在同一框架上继续补：

1. `layer_metrics.py`：固定 calibration activation，比较 `xW` 和 `xW_hat`；
2. C4 / WikiText2 数据下载缓存策略；
3. 汇总脚本：对比 `Hadamard-LM W3` 与 `Hadamard-Absmax W4`。

## 运行入口

模型下载完成并安装实验依赖后，可运行：

```bash
scripts/run_stage_a_tensor_sweep.sh
```

运行 model-level PPL：

```bash
scripts/run_stage_a_ppl.sh
```

在当前 Codex 环境中，MPS/Metal GPU 只有在授权外部命令中可见。已验证：

```text
inside sandbox: torch.backends.mps.is_available() == False
outside sandbox: torch.backends.mps.is_available() == True, device_count == 1
```

因此需要用 GPU 跑 model-level PPL 时，应以授权方式执行，并传入：

```bash
--device mps
```

输出位置：

```text
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_tensor_sweep>/tensor_metrics.jsonl
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_tensor_sweep>/tensor_metrics.csv
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_tensor_sweep>/run_metadata.json
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_ppl>/ppl_runs.jsonl
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_ppl>/ppl.csv
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_ppl>/run_metadata.json
```
