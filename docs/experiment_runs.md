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
