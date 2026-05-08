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

## 2026-05-07：Stage B Implementation Smoke Tests

| Item | Value |
| --- | --- |
| Git commit before implementation | `a7402f0c1f8270a84771bbc6270e479a133cdb30` |
| Note | Smoke runs were executed before committing Stage B code, so `run_metadata.json` records a dirty worktree. |

### B1 Activation Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260507_204757+0800_stage_b_activation` |
| Output dir | `outputs/stage_b/20260507_204757+0800_stage_b_activation` |
| Scope | layer 0 only, WikiText2 test, max samples 2, sequence length 64 |
| Records | 54 |
| Duration | 20.976 seconds |

Selected B1 observations:

| Site | Comparison | Relative MSE |
| --- | --- | ---: |
| `attn_input` | Rot-LM A3 vs Rot-Absmax A4 | 0.036233 vs 0.056836 |
| `ffn_input` | Rot-LM A3 vs Rot-Absmax A4 | 0.031058 vs 0.063336 |
| `ffn_intermediate` | Rot-LM A3 vs Rot-Absmax A4 | 0.033353 vs 0.151254 |
| `k_proj_out` | Rot-LM A3 vs Rot-Absmax A4 | 0.029491 vs 0.033453 |
| `q_proj_out` | Rot-LM A3 vs Rot-Absmax A4 | 0.034286 vs 0.102861 |
| `v_proj_out` | Rot-LM A3 vs Rot-Absmax A4 | 0.032232 vs 0.027583 |

Smoke conclusion: B1 pipeline works. On this tiny sample, Rot-LM A3 is usually better than Rot-Absmax A4, except `v_proj_out` where Rot-Absmax A4 is slightly better.

### B2 Local Linear / FFN Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260507_205123+0800_stage_b_local` |
| Output dir | `outputs/stage_b/20260507_205123+0800_stage_b_local` |
| Scope | layer 0 only, WikiText2 test, max samples 2, sequence length 64 |
| Linear records | 49 |
| FFN records | 7 |
| Duration | 23.239 seconds |

Selected B2 mean relative MSE:

| Group | Method | Relative MSE |
| --- | --- | ---: |
| Linear | Rot-Absmax W4A4 | 0.291163 |
| Linear | Rot-LM W3A4 | 0.076058 |
| Linear | Rot-LM W4A3 | 0.067143 |
| FFN | FFN Rot-Absmax W4A4 | 0.387884 |
| FFN | FFN Rot-LM W3A4 | 0.042874 |
| FFN | FFN Rot-LM W4A3 | 0.040741 |

Smoke conclusion: B2 pipeline works. On this tiny sample, Rot-LM W3A4 / W4A3 are both much better than Rot-Absmax W4A4 for local Linear and FFN output error.

### B4 FFN-only PPL Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260507_211859+0800_stage_b_ppl` |
| Output dir | `outputs/stage_b/20260507_211859+0800_stage_b_ppl` |
| Device | `mps` |
| Scope | WikiText2 test, max samples 2, sequence length 64 |
| Records | 2 |
| Duration | 332.475 seconds |

PPL smoke results:

| Method | PPL |
| --- | ---: |
| FP16 | 617.328152 |
| FFN Rot-LM W4A4 | 579.691680 |

Smoke conclusion: B4 model wrapper and PPL output path work on MPS. This short-context PPL is only a path check, not a formal quality conclusion.

## 2026-05-08：Stage B Full Runs

| Item | Value |
| --- | --- |
| Git commit | `030993fcc59af05c6b01f8c336a74f5aecfd3a99` |
| Note | Runs include a dirty worktree with Stage B PPL performance fixes: CPU-side FFN weight prequantization and MPS-friendly Lloyd-Max threshold indexing. |

### B1 Activation Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_090340+0800_stage_b_activation` |
| Output dir | `outputs/stage_b/20260508_090340+0800_stage_b_activation` |
| Device | `mps` |
| Scope | all 22 layers, WikiText2 test, max samples 32, sequence length 512 |
| Records | 1188 |
| Duration | 35.697 seconds |

Mean metrics across activation sites:

| Method | Bits | Relative MSE | Cosine | SQNR dB |
| --- | ---: | ---: | ---: | ---: |
| Direct Absmax | 4 | 0.327630 | 0.820784 | 6.579584 |
| Rot Absmax | 4 | 0.075469 | 0.963145 | 12.490349 |
| Rot LM | 4 | 0.009297 | 0.995561 | 20.390117 |
| Rot LM | 3 | 0.034815 | 0.983110 | 14.620511 |
| Rot LM | 2 | 0.120526 | 0.940698 | 9.243337 |

Conclusion: `Rot-LM A3` beats `Rot-Absmax A4` on average. Per-site, the only small exception is `k_proj_out`, where `Rot-LM A3` MSE is `0.032644` versus `Rot-Absmax A4` MSE `0.029189`.

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

Selected mean relative MSE:

| Group | Method | Relative MSE | Cosine |
| --- | --- | ---: | ---: |
| Linear | Rot-Absmax W4A4 | 0.228609 | 0.901790 |
| Linear | Rot-LM W3A4 | 0.037209 | 0.982336 |
| Linear | Rot-LM W4A3 | 0.037381 | 0.982784 |
| FFN | FFN Rot-Absmax W4A4 | 0.477501 | 0.810050 |
| FFN | FFN Rot-LM W3A4 | 0.073063 | 0.964408 |
| FFN | FFN Rot-LM W4A3 | 0.079189 | 0.964333 |

Conclusion: local Linear and FFN results both strongly support `Rot-LM W3A4` / `Rot-LM W4A3` over `Rot-Absmax W4A4`.

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

Stage B conclusion: B-line succeeds. `FFN Rot-LM W3A4` and `FFN Rot-LM W4A3` remain close to FP16 and are far better than `FFN Rot-Absmax W4A4`. This is numerical fake-quant evidence for lower W/A bit-width, not direct hardware acceleration evidence.
