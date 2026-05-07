# Experiment Runs

本文档作为实验产物索引。脚本会在 `outputs/` 下写入机器可读 metadata；这里记录人工可读的版本、时间和结论。

## 2026-05-07：Stage A 准备

| Item | Value |
| --- | --- |
| Timezone | Asia/Shanghai |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Model path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Downloaded files | `model.safetensors`, `pytorch_model.bin`, tokenizer/config files |
| Model size on disk | about 8.2 GB |
| Architecture check | 154 Stage-A target Linear weights, all aligned to block size 128 |
| Current Stage-A code commit | `ce2201b` before the metadata/logging update |

## 2026-05-07：MPS 环境修复

| Item | Value |
| --- | --- |
| Timezone | Asia/Shanghai |
| Official PyTorch wheel | `torch==2.11.0`, `torchvision==0.26.0`, `torchaudio==2.11.0` |
| Other core packages | `transformers==5.8.0`, `safetensors==0.7.0`, `datasets==4.8.5` |
| Hardware | Apple M4 GPU, Metal Supported |
| Sandbox check | `mps built=True`, `mps available=False`, `device_count=0` |
| Authorized external check | `mps built=True`, `mps available=True`, `device_count=1` |

结论：PyTorch 官方 wheel 支持 MPS；当前 Codex 沙箱内无法枚举 MPS，后续 GPU 实验需要授权外部执行命令。

后续 A 阶段运行会记录到带时间戳的独立目录：

```text
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_tensor_sweep>/run_metadata.json
outputs/stage_a/<YYYYMMDD_HHMMSS_stage_a_ppl>/run_metadata.json
```

目录名中的 `<YYYYMMDD_HHMMSS_...>` 与 `run_metadata.json` 中的 `run_id` 字段保持一致。

## 2026-05-07：Stage A Tensor Sweep

| Item | Value |
| --- | --- |
| Run ID | `20260507_192029+0800_stage_a_tensor_sweep` |
| Output dir | `outputs/stage_a/20260507_192029+0800_stage_a_tensor_sweep` |
| Git commit | `009c87ab42176d36cc9b50acc27d63bd7cdea4a6` |
| Device context | Authorized external run, MPS available |
| Target weights | 154 TinyLlama q/k/v/o + FFN Linear weights |
| Methods | `direct_absmax`, `hadamard_absmax`, `hadamard_lm` |
| Bits | W4 / W3 / W2 |
| Records | 1386 |

Method-level mean metrics:

| Method | Bits | Relative MSE | Cosine | SQNR dB |
| --- | ---: | ---: | ---: | ---: |
| Direct Absmax | 4 | 0.537248 | 0.690929 | 3.935982 |
| Hadamard Absmax | 4 | 0.042540 | 0.980618 | 13.747699 |
| Hadamard LM | 4 | 0.009364 | 0.996558 | 20.286120 |
| Hadamard LM | 3 | 0.034078 | 0.984053 | 14.675942 |
| Hadamard LM | 2 | 0.116225 | 0.941289 | 9.347435 |

阶段性结论：tensor reconstruction 层面，`Hadamard-LM W3` 已经优于 `Hadamard-Absmax W4`；`Hadamard-LM W2` 也明显优于 `Hadamard-Absmax W3`。这只说明低比特数值质量和位宽可行性，不解释成真实 INT GEMM 加速。

## 2026-05-07：Stage A PPL Smoke Test

| Item | Value |
| --- | --- |
| Run ID | `20260507_192603+0800_stage_a_ppl` |
| Output dir | `outputs/stage_a/20260507_192603+0800_stage_a_ppl` |
| Git commit | `009c87ab42176d36cc9b50acc27d63bd7cdea4a6` |
| Device | `mps` |
| Dataset | WikiText2 raw test split |
| Max samples | 64 |
| Sequence length / stride | 512 / 512 |
| Duration | 114.707 seconds |

PPL results:

| Method | Bits | PPL |
| --- | ---: | ---: |
| FP16 | 16 | 10.549959 |
| Hadamard Absmax | 4 | 20.447154 |
| Hadamard Absmax | 3 | 25289.088084 |
| Hadamard LM | 4 | 11.561565 |
| Hadamard LM | 3 | 17.966018 |

阶段性结论：在小规模 PPL 验证里，`Hadamard-LM W3` 仍然优于 `Hadamard-Absmax W4`，而 `Hadamard-Absmax W3` 明显崩溃。这支持 A 线核心判断：Hadamard rotation + Lloyd-Max 的价值在于把 weight-only 可用位宽从 4 bit 推向 3 bit。

## 2026-05-07：Stage A Full PPL Grid

| Item | Value |
| --- | --- |
| Run ID | `20260507_193316+0800_stage_a_ppl` |
| Output dir | `outputs/stage_a/20260507_193316+0800_stage_a_ppl` |
| Git commit | `009c87ab42176d36cc9b50acc27d63bd7cdea4a6` |
| Device | `mps` |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |
| Records | 10 |
| Duration | 964.826 seconds |

注意：`run_metadata.json` 中保留了运行时 dirty worktree 状态；后续文档整理和脚本修正会在实验收口后提交。

PPL results:

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

A 阶段核心判断：`Hadamard-LM W3` 的 PPL 明显优于 `Hadamard-Absmax W4`，说明 Lloyd-Max 在 Hadamard rotation 后确实能把 weight-only 的数值可用位宽从 W4 推向 W3。`Hadamard-LM W2` 在 tensor reconstruction 上仍可看，但 model-level PPL 已经崩溃，暂时应作为失败边界。
