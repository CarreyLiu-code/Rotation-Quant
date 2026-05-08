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

### B3 FFN-only PPL Smoke

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

Smoke conclusion: B3 model wrapper and PPL output path work on MPS. This short-context PPL is only a path check, not a formal quality conclusion.

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

## 2026-05-08：Stage C Planning and Run Record Policy

Stage C 的实验计划和代码架构记录在 `docs/stage_c.md`。本节只是后续 C 阶段正式运行的人工记录入口，不包含实验结果。

| Item | Value |
| --- | --- |
| Stage C doc | `docs/stage_c.md` |
| Main scope | Attention/KV cache inner-product preservation, QJL residual, structured Attention fake quant |
| Model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Output root | `outputs/stage_c/` |
| Run metadata rule | output directory name must match `run_metadata.json.run_id` |

Planned run families:

| Run family | Output directory pattern | Purpose |
| --- | --- | --- |
| C1 invariance | `outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_invariance>` | post-RoPE Q/K head-wise rotation invariance sanity |
| C2 KV local | `outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>` | Hadamard-LM K/V low-bit attention-local metrics |
| C3 QJL residual | `outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_qjl>` | Key residual inner-product correction |
| C4 Attention layer | `outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>` | q/k/v/o W/A fake quant plus KV cache quant, layer-level |
| C5 PPL | `outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_ppl>` | Attention-only model-level PPL |
| C5 accuracy | `outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_accuracy>` | optional zero-shot accuracy |

Each formal Stage C run should append:

| Field | Required |
| --- | --- |
| Run ID | yes |
| Output dir | yes |
| Git commit | yes |
| Dirty status | yes |
| Device and dataset | yes |
| Scope | layer limit, max samples, sequence length |
| Core table | yes |
| Conclusion | numerical quality only; no native low-bit acceleration claim |

## 2026-05-08：Stage C Implementation Smoke Tests

| Item | Value |
| --- | --- |
| Git commit before implementation | `e053af4b9f6aa7ab21d4047389572d085dfcda97` |
| Note | Smoke runs were executed while Stage C files were still uncommitted, so `run_metadata.json` records a dirty worktree. |

### C1 Invariance Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260508_100235+0800_stage_c_invariance` |
| Output dir | `outputs/stage_c/20260508_100235+0800_stage_c_invariance` |
| Scope | layer 0, WikiText2 test, max samples 2, sequence length 64 |
| Records | 1 |
| Duration | 23.197 seconds |

Selected metrics:

| Metric | Value |
| --- | ---: |
| Score relative MSE | 0.0 |
| Max score abs diff | 0.00000167 |
| Output relative MSE | 0.0 |
| Output cosine | 1.00000048 |

Conclusion: post-RoPE Q/K head-wise Hadamard rotation preserves attention score and output within numerical tolerance.

### C2 KV Local Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260508_100425+0800_stage_c_kv_local` |
| Output dir | `outputs/stage_c/20260508_100425+0800_stage_c_kv_local` |
| Scope | layer 0, selected KV methods, WikiText2 test, max samples 2, sequence length 64 |
| Records | 4 |
| Duration | 21.074 seconds |

Selected mean metrics:

| Method | Bits | Score rel MSE | Softmax KL | Output cosine |
| --- | --- | ---: | ---: | ---: |
| Absmax | K4V4 | 0.045601 | 0.012718 | 0.968360 |
| Hadamard-LM | K4V4 | 0.007246 | 0.002742 | 0.992596 |
| Hadamard-LM | K3V4 | 0.029824 | 0.013261 | 0.966418 |

Conclusion: C2 output path works. On this tiny sample, `Hadamard-LM K4V4` is clearly better than `Absmax K4V4`; `Hadamard-LM K3V4` is numerically plausible but needs full local validation.

### C3 QJL Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260508_100709+0800_stage_c_qjl` |
| Output dir | `outputs/stage_c/20260508_100709+0800_stage_c_qjl` |
| Scope | layer 0, K2/K3 pure and QJL methods, max samples 2, sequence length 64 |
| Records | 4 |
| Duration | 22.347 seconds |

Selected mean metrics:

| Method | Bits | Score rel MSE | Softmax KL | Output cosine |
| --- | --- | ---: | ---: | ---: |
| Hadamard-LM K2 | K2 | 0.081561 | 0.028200 | 0.962390 |
| Hadamard-LM K2 + QJL | K2+QJL | 0.105033 | 0.043932 | 0.929466 |
| Hadamard-LM K3 | K3 | 0.029824 | 0.013261 | 0.966418 |
| Hadamard-LM K3 + QJL | K3+QJL | 0.037097 | 0.016127 | 0.967218 |

Conclusion: QJL implementation runs and writes metrics. This tiny sample does not show QJL improvement, so QJL should not be promoted into C5 full PPL unless the larger C3 run shows a benefit.

### C4 Attention-layer Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260508_101201+0800_stage_c_attention_layer` |
| Output dir | `outputs/stage_c/20260508_101201+0800_stage_c_attention_layer` |
| Scope | layer 0, `fp16` and `attn_rot_lm_w4a4_hlm_k4v4`, max samples 2, sequence length 64 |
| Records | 2 |
| Duration | 20.579 seconds |

Selected mean metrics:

| Method | Projection rel MSE | Score rel MSE | Softmax KL | Layer output rel MSE | Layer output cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attn Rot-LM W4A4 + HLM K4V4 | 0.058351 | 0.010309 | 0.004012 | 0.042903 | 0.978361 |

Conclusion: q/k/v/o W/A fake quant plus post-RoPE KV fake quant can be run as a complete Attention-layer simulation.

### C5 PPL Smoke

| Item | Value |
| --- | --- |
| Run ID | `20260508_101252+0800_stage_c_ppl` |
| Output dir | `outputs/stage_c/20260508_101252+0800_stage_c_ppl` |
| Scope | `fp16` and `attn_rot_lm_w4a4_hlm_k4v4`, WikiText2 test, max samples 2, sequence length/stride 64/64 |
| Records | 2 |
| Duration | 30.706 seconds |

PPL path-check results:

| Method | PPL |
| --- | ---: |
| FP16 | 617.328152 |
| Attn Rot-LM W4A4 + HLM K4V4 | 6.384535 |

Conclusion: the Attention-only model wrapper runs through causal LM PPL on MPS. This short-context smoke is only an interface/path check and should not be interpreted as a model quality result.

## 2026-05-08：Stage C Full Runs

| Item | Value |
| --- | --- |
| Git commit | `e053af49591b23541ed63cd5794fff911a6d0f29` |
| Dirty status | Stage C code/docs were uncommitted; `ZJU-Beamer-Template-main/` was also untracked and unrelated. |
| Runtime | `torch 2.11.0`, `transformers 5.8.0`, MPS available |
| Model | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Interpretation boundary | Fake quant numerical quality only; no native low-bit speedup claim. |

### C1 Invariance Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_105020+0800_stage_c_invariance` |
| Output dir | `outputs/stage_c/20260508_105020+0800_stage_c_invariance` |
| Device | `mps` |
| Scope | all 22 layers, WikiText2 test, max samples 8, sequence length 128 |
| Records | 22 |
| Duration | 19.756 seconds |

Mean metrics:

| Score rel MSE | Max score abs diff | Output rel MSE | Output cosine |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.00001214 | 0.0 | 1.00001176 |

Conclusion: post-RoPE Q/K head-wise Hadamard invariance holds across all 22 layers under the formal local setting.

### C2 KV Local Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_102337+0800_stage_c_kv_local` |
| Output dir | `outputs/stage_c/20260508_102337+0800_stage_c_kv_local` |
| Device | `mps` |
| Scope | all 22 layers, WikiText2 test, max samples 8, sequence length 128 |
| Records | 198 |
| Duration | 27.978 seconds |

Mean metrics:

| Method | Bits | Score rel MSE | Softmax KL | Top-k overlap | Output cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Absmax | K4V4 | 0.012673 | 0.238097 | 0.688388 | 0.970790 |
| Absmax | K3V4 | 0.057856 | 0.810398 | 0.504865 | 0.908665 |
| Absmax | K4V3 | 0.012673 | 0.238097 | 0.688388 | 0.957904 |
| Hadamard-LM | K4V4 | 0.003110 | 0.052050 | 0.818279 | 0.992406 |
| Hadamard-LM | K3V4 | 0.011561 | 0.183433 | 0.690212 | 0.976442 |
| Hadamard-LM | K4V3 | 0.003110 | 0.052050 | 0.818279 | 0.988012 |
| Hadamard-LM | K3V3 | 0.011561 | 0.183433 | 0.690212 | 0.972211 |
| Hadamard-LM | K2V4 | 0.042143 | 0.565414 | 0.534261 | 0.927240 |

Conclusion: C2 supports the KV-local hypothesis. `Hadamard-LM K4V4` is much better than `Absmax K4V4`, and `Hadamard-LM K3V4` is close to or slightly better than `Absmax K4V4` on score error, KL, top-k overlap, and output cosine. `K2V4` is a clear failure boundary.

### C3 QJL Sensitive-layer Gate Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_102515+0800_stage_c_qjl` |
| Output dir | `outputs/stage_c/20260508_102515+0800_stage_c_qjl` |
| Device | `mps` |
| Scope | C2 KL-sensitive layers `16,19,20,21`, WikiText2 test, max samples 8, sequence length 128 |
| Records | 16 |
| Duration | 21.539 seconds |

Mean metrics:

| Method | Bits | Score rel MSE | Softmax KL | Top-k overlap | Output cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Hadamard-LM K2 | K2 | 0.037739 | 1.234568 | 0.504553 | 0.887354 |
| Hadamard-LM K2 + QJL | K2+QJL | 0.061972 | 2.057504 | 0.453162 | 0.859132 |
| Hadamard-LM K3 | K3 | 0.011456 | 0.392211 | 0.660675 | 0.967200 |
| Hadamard-LM K3 + QJL | K3+QJL | 0.017640 | 0.622243 | 0.610620 | 0.947417 |

Conclusion: QJL residual does not improve the selected sensitive layers. It reduces mean inner-product bias, but score relative MSE, softmax KL, top-k overlap, and output cosine all get worse.

### C3 QJL All-layer Supplemental Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_104441+0800_stage_c_qjl` |
| Output dir | `outputs/stage_c/20260508_104441+0800_stage_c_qjl` |
| Device | `mps` |
| Scope | all 22 layers, WikiText2 test, max samples 8, sequence length 128 |
| Records | 88 |
| Duration | 23.839 seconds |

Mean metrics:

| Method | Bits | Score rel MSE | Softmax KL | Top-k overlap | Output cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Hadamard-LM K2 | K2 | 0.042143 | 0.565414 | 0.534261 | 0.927240 |
| Hadamard-LM K2 + QJL | K2+QJL | 0.059468 | 0.966018 | 0.491696 | 0.889142 |
| Hadamard-LM K3 | K3 | 0.011561 | 0.183433 | 0.690212 | 0.976442 |
| Hadamard-LM K3 + QJL | K3+QJL | 0.017579 | 0.297445 | 0.645228 | 0.960473 |

Conclusion: all-layer C3 confirms the gate result. QJL is not included in C5 PPL because it worsens score relative MSE, softmax KL, top-k overlap, and output cosine for both K2 and K3 bases.

### C4 Attention-layer Structured Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_102632+0800_stage_c_attention_layer` |
| Output dir | `outputs/stage_c/20260508_102632+0800_stage_c_attention_layer` |
| Device | `mps` |
| Scope | all 22 Attention layers, WikiText2 test, max samples 8, sequence length 128 |
| Records | 132 |
| Duration | 36.826 seconds |

Mean metrics:

| Method | Linear bits | KV bits | Projection rel MSE | Score rel MSE | Softmax KL | Layer output rel MSE | Layer output cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Direct Absmax + Absmax KV | W4A4 | K4V4 | 0.363401 | 0.128626 | 0.599042 | 1.314227 | 0.250457 |
| Rot Absmax + HLM KV | W4A4 | K4V4 | 0.143706 | 0.043899 | 0.160127 | 1.618342 | 0.597002 |
| Rot-LM + HLM KV | W4A4 | K4V4 | 0.023228 | 0.005036 | 0.057384 | 0.105883 | 0.947851 |
| Rot-LM + HLM KV | W3A4 | K3V4 | 0.054862 | 0.018479 | 0.187293 | 0.287546 | 0.864589 |
| Rot-LM + HLM KV | W4A3 | K4V3 | 0.053374 | 0.008580 | 0.072355 | 0.186937 | 0.907371 |

Conclusion: C4 local Attention-layer metrics strongly favor Rot-LM over direct/rotated absmax. `Rot-LM W3A4 + HLM K3V4` keeps much better layer output quality than `Rot-Absmax W4A4 + HLM K4V4`, but it is still visibly worse than `Rot-LM W4A4 + HLM K4V4`.

### C5 Attention-only PPL Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_102751+0800_stage_c_ppl` |
| Output dir | `outputs/stage_c/20260508_102751+0800_stage_c_ppl` |
| Device | `mps` |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |
| Records | 6 |
| Duration | 689.864 seconds |

PPL results:

| Method | PPL |
| --- | ---: |
| FP16 | 8.048573 |
| Attention Direct Absmax W4A4 + Absmax K4V4 | 2758.251695 |
| Attention Rot Absmax W4A4 + HLM K4V4 | 17607.464742 |
| Attention Rot-LM W4A4 + HLM K4V4 | 12044.785422 |
| Attention Rot-LM W3A4 + HLM K3V4 | 8105.302082 |
| Attention Rot-LM W4A3 + HLM K4V3 | 8362.464387 |

Stage C conclusion: KV-local and single-layer Attention results are encouraging, but the current Attention-only model-level wrapper does not preserve PPL. This suggests the first C5 implementation is not yet a usable full-model quantization path; likely next checks are residual stream / `o_proj` interaction, accumulation of attention-layer errors across depth, and whether value rotation should be structurally absorbed rather than reconstructed locally. The useful positive result remains C2: post-RoPE Hadamard-LM can make `K3V4` locally competitive with uniform `K4V4`; the negative result is that this does not yet survive full Attention-only replacement.

## 2026-05-08：Stage C Value-Absorb Integration Plan

The value rotation + `o_proj` absorb path has been integrated into the main Stage C code and is now documented in `docs/stage_c.md`.

| Item | Value |
| --- | --- |
| Stage C doc | `docs/stage_c.md` |
| Output root for future runs | `outputs/stage_c/` |
| Main change | Value rotation remains in attention output and is absorbed into `o_proj.weight` using independent H64 blocks |
| Q/K rule | keep per-head H64 score rotation; no cross-head H128 mixing |

## 2026-05-08：Stage C Value-Absorb Formal Runs

The following runs were originally produced during the value-absorb development pass and are now treated as the reference results for integrated Stage C C4/C5. Their historical run IDs and output paths are kept unchanged for traceability.

| Run | Run ID | Output dir | Summary |
| --- | --- | --- | --- |
| C4 integrated Attention-layer local full | `20260508_182056+0800_stage_c_refine_attention_layer` | `outputs/stage_c_refine/20260508_182056+0800_stage_c_refine_attention_layer` | identity passes; best structured local method is `attn_rot_lm_w4a4_hlm_k4v4` |
| C5 integrated Attention-only PPL full | `20260508_182156+0800_stage_c_refine_ppl` | `outputs/stage_c_refine/20260508_182156+0800_stage_c_refine_ppl` | identity matches FP16; `KV-only K4V4` PPL 8.204482; best structured method `W4A4/K4V4` PPL 8.614497 |

## 2026-05-09：Stage C Main-Entry Integration Smoke

After merging the value-absorb path back into the main Stage C code, the canonical C4/C5 entrypoints were smoke-tested under `outputs/stage_c/`.

| Run | Run ID | Scope | Result |
| --- | --- | --- | --- |
| C4 main entry smoke | `20260509_002909+0800_stage_c_attention_layer` | layer 0, max samples 2, sequence length 64; `fp16`, `attn_identity_fp16`, `attn_kv_hlm_k4v4`, `attn_rot_lm_w4a4_hlm_k4v4` | identity relative MSE `7.03e-08`; KV-only K4V4 layer output cosine `0.998133`; Rot-LM W4A4 + HLM K4V4 cosine `0.982361` |
| C5 main entry smoke | `20260509_002951+0800_stage_c_ppl` | WikiText2 path check, max samples 2, sequence length/stride 64/64; `fp16`, `attn_identity_fp16`, `attn_kv_hlm_k4v4` | FP16 and identity both `617.328152`; KV-only K4V4 `647.099665`; short-context PPL is only a path check |
| C4 cleanup smoke | `20260509_003502+0800_stage_c_attention_layer` | layer 0, max samples 1, sequence length 64; after removing the old model-level wrapper | identity relative MSE `6.99e-08`; KV-only K4V4 layer output cosine `0.997237` |
