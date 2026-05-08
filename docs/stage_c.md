# Stage C Attention/KV Cache Experiments

本文档记录 Stage C 的实验代码框架、分组命名、运行入口、产物规范和正式结果。Stage C 只解释 Attention/KV fake quant 的数值质量和低比特可行性，不解释为真实 KV cache 压缩收益、low-bit attention kernel 加速或专用硬件收益。

## 实验目标

验证 Hadamard rotation + Lloyd-Max 是否能在更低 bit-width 下保持 KV cache 的 attention score 和 attention output；验证 QJL residual 是否能修正 Key quantization 的 inner product 误差；最后把 q/k/v/o 线性 W/A fake quant 与 post-RoPE rotated KV cache quant 串成完整 Attention 层，评估 local error 与 model-level PPL。

核心判断：

> `Hadamard-LM K3V4` 是否能在 KV-local 指标上接近或优于 4-bit baseline；value rotation + `o_proj` absorb 后，C4 local 结果是否能与 C5 PPL 趋势保持一致？

## 实验模型

| Item | Value |
| --- | --- |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Local path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Architecture | LLaMA, 22 layers, hidden size 2048, intermediate size 5632 |
| Attention | 32 query heads, 4 KV heads, head dim 64 |
| GQA rule | K/V heads are repeated to match Q heads before attention score |
| Target modules | 22 self-attention modules, including q/k/v/o projections and KV cache tensors |
| W/A block size | 128 for q/k/v/o Linear fake quant |
| KV rotation size | 64, equal to head dim |

TinyLlama 的 `q_proj` / `o_proj` shape 为 `[2048, 2048]`，`k_proj` / `v_proj` shape 为 `[256, 2048]`。q/k/v projection 输出 reshape 后分别是 Q `[batch, seq, 32, 64]`、K/V `[batch, seq, 4, 64]`。

## 编号与命名

| Stage | Scope | Primary output |
| --- | --- | --- |
| C0 | computation convention | defines RoPE / rotation / value path |
| C1 | Attention capture + invariance sanity | `invariance_metrics.csv` |
| C2 | Attention-local KV quant | `kv_metrics.csv` |
| C3 | Key-only QJL residual | `qjl_metrics.csv` |
| C4 | Attention-layer structured quant | `attention_layer_metrics.csv` |
| C5 | Attention-only model-level PPL / accuracy | `ppl.csv` / `accuracy.csv` |

### C0 Computation Convention

Key quantization is post-RoPE. C 阶段先得到真实 `Q_rope` / `K_rope`，再对每个 head 的 64 维向量做 Hadamard rotation：

```text
Q_H = Q_rope H64
K_H = K_rope H64
score = Q_H @ K_H.T / sqrt(d_h)
```

Value 不经过 RoPE。C4/C5 的正式 structured path 使用 value rotation + `o_proj` absorb：

```text
V_H_hat = Q(V H64)
O_H = P @ V_H_hat
W_o_abs = W_o H64_blockdiag
Y = O_H @ W_o_abs.T
```

这里 `P` 只在 sequence 维做 token mixing，不作用在 feature/head 维，因此 `P @ (V H) = (P @ V) H`。`attn_kv_hlm_k4v4_reconstruct` 只作为 C4 value path diagnostic，不作为 C5 默认方法。

## 实验组

### C1 Invariance

| Method | Description | Key metric |
| --- | --- | --- |
| `fp16` | original attention computation | baseline |
| no-quant H64 path | post-RoPE Q/K head-wise Hadamard without quantization | score relative MSE close to 0 |

C1 采集 `q_proj_out`、`k_proj_out`、`v_proj_out`、`q_rope`、`k_rope`、attention score、attention probability 和 attention output，用于验证后续 C2-C5 的数据路径。

### C2 Attention-local KV Quant

| Method key | K/V bits | Quantizer type | Interpretation |
| --- | --- | --- | --- |
| `fp16` | K16V16 | none | baseline |
| `absmax_k4v4` | K4V4 | uniform fake quant | integer-like KV baseline |
| `absmax_k3v4` | K3V4 | uniform fake quant | key bit reduction baseline |
| `absmax_k4v3` | K4V3 | uniform fake quant | value bit reduction baseline |
| `hadamard_lm_k4v4` | K4V4 | head-wise rotation + Lloyd-Max | non-uniform same-bit comparison |
| `hadamard_lm_k3v4` | K3V4 | head-wise rotation + Lloyd-Max | key bit reduction |
| `hadamard_lm_k4v3` | K4V3 | head-wise rotation + Lloyd-Max | value bit reduction |
| `hadamard_lm_k3v3` | K3V3 | head-wise rotation + Lloyd-Max | aggressive KV combination |
| `hadamard_lm_k2v4` | K2V4 | head-wise rotation + Lloyd-Max | failure boundary |

Key 优先看 inner product bias / variance、score MSE、softmax KL 和 top-k overlap；Value 优先看 reconstruction error、attention output relative MSE 和 output cosine。

### C3 QJL Residual

| Method key | Base K bits | Residual bits | Value | Interpretation |
| --- | ---: | ---: | --- | --- |
| `hadamard_lm_k3` | 3 | 0 | V4 fixed | pure reconstruction |
| `hadamard_lm_k2` | 2 | 0 | V4 fixed | aggressive base |
| `hadamard_lm_k2_qjl` | 2 | 1 | V4 fixed | inner-product corrected |
| `hadamard_lm_k3_qjl` | 3 | 1 | V4 fixed | stronger corrected baseline |

QJL 默认使用 Gaussian projection，projection dim `m=d_h=64`。Residual sign bits 与 norm metadata 单独记录，不折入简单的 K/V bit label。

### C4 Attention-layer Structured Quant

| Method key | Attention Linear | KV cache | Value path | Interpretation |
| --- | --- | --- | --- | --- |
| `fp16` | FP16 | FP16 | reference | baseline |
| `attn_identity_fp16` | FP16 wrapper | FP16 | identity | wrapper sanity |
| `attn_kv_hlm_k4v4_reconstruct` | FP16 | HLM K4V4 | reconstruct | value path diagnostic |
| `attn_kv_hlm_k4v4` | FP16 | HLM K4V4 | o_proj absorb | KV-only 4-bit |
| `attn_kv_hlm_k3v4` | FP16 | HLM K3V4 | o_proj absorb | key bit reduction |
| `attn_kv_hlm_k4v3` | FP16 | HLM K4V3 | o_proj absorb | value bit reduction |
| `attn_rot_lm_w4a4_hlm_k4v4` | Rot-LM W4A4 | HLM K4V4 | o_proj absorb | structured same-bit baseline |
| `attn_rot_lm_w3a4_hlm_k3v4` | Rot-LM W3A4 | HLM K3V4 | o_proj absorb | weight/key bit reduction |
| `attn_rot_lm_w4a3_hlm_k4v3` | Rot-LM W4A3 | HLM K4V3 | o_proj absorb | activation/value bit reduction |

C4 输出 q/k/v projection error、score relative MSE、softmax KL、top-k overlap、pre-`o_proj` output error、final attention output relative MSE / cosine 和 per-layer sensitivity。

### C5 Attention-only PPL

| Method key | Attention Linear | KV cache | Value path | Interpretation |
| --- | --- | --- | --- | --- |
| `fp16` | FP16 | FP16 | reference | baseline |
| `attn_identity_fp16` | FP16 wrapper | FP16 | identity | wrapper sanity |
| `attn_kv_hlm_k4v4` | FP16 | HLM K4V4 | o_proj absorb | KV-only 4-bit |
| `attn_kv_hlm_k3v4` | FP16 | HLM K3V4 | o_proj absorb | key bit reduction |
| `attn_rot_lm_w4a4_hlm_k4v4` | Rot-LM W4A4 | HLM K4V4 | o_proj absorb | structured same-bit baseline |
| `attn_rot_lm_w3a4_hlm_k3v4` | Rot-LM W3A4 | HLM K3V4 | o_proj absorb | weight/key bit reduction |
| `attn_rot_lm_w4a3_hlm_k4v3` | Rot-LM W4A3 | HLM K4V3 | o_proj absorb | activation/value bit reduction |

C5 只替换 Attention，不替换 FFN。正式 PPL 默认使用 WikiText2 raw test split、`max_samples=512`、`sequence_length=2048`、`stride=2048`、`device=mps`。

## 代码框架

| Path | Role |
| --- | --- |
| `src/rotationquant/stage_c.py` | Stage C method registry、KV quant spec、head-wise Hadamard、post-RoPE K/V quant、QJL estimator、score/output metrics |
| `src/rotationquant/attention_capture.py` | TinyLlama attention wrapper / hook，采集 q/k/v、RoPE 后 Q/K、score、prob、attention output |
| `src/rotationquant/stage_c_model.py` | Attention-only model wrapper，接入 q/k/v/o W/A fake quant、KV fake quant、value rotation + `o_proj` absorb |
| `src/rotationquant/stage_b.py` | q/k/v/o Linear W/A fake quant |
| `src/rotationquant/rotations.py` | FWHT |
| `src/rotationquant/quantizers.py` | symmetric absmax 与 Gaussian Lloyd-Max fake quant |
| `src/rotationquant/metrics.py` | tensor metrics、score bias/variance、softmax KL、top-k overlap |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL |
| `src/rotationquant/run_metadata.py` | run id、时间、Git 状态、包版本、torch runtime metadata |
| `experiments/stage_c_invariance.py` | C1 capture 与 no-quant invariance |
| `experiments/stage_c_kv_local.py` | C2 Attention-local KV quant sweep |
| `experiments/stage_c_qjl.py` | C3 QJL residual sweep |
| `experiments/stage_c_attention_layer.py` | C4 structured Attention layer output error |
| `experiments/stage_c_ppl.py` | C5 Attention-only model-level PPL |
| `experiments/stage_c_accuracy.py` | C5 optional zero-shot accuracy |
| `experiments/summarize_stage_c.py` | Stage C summary |
| `scripts/run_stage_c_invariance.sh` | C1 shell 入口 |
| `scripts/run_stage_c_kv_local.sh` | C2 shell 入口 |
| `scripts/run_stage_c_qjl.sh` | C3 shell 入口 |
| `scripts/run_stage_c_attention_layer.sh` | C4 shell 入口 |
| `scripts/run_stage_c_ppl.sh` | C5 PPL shell 入口 |
| `scripts/run_stage_c_accuracy.sh` | C5 accuracy shell 入口 |

## 运行入口

C1 invariance：

```bash
scripts/run_stage_c_invariance.sh
```

C2 Attention-local KV：

```bash
scripts/run_stage_c_kv_local.sh
```

C3 QJL residual：

```bash
scripts/run_stage_c_qjl.sh
```

C4 Attention-layer structured quant：

```bash
scripts/run_stage_c_attention_layer.sh
```

C5 Attention-only PPL：

```bash
scripts/run_stage_c_ppl.sh
```

C5 optional accuracy：

```bash
scripts/run_stage_c_accuracy.sh
```

汇总任意 Stage C run：

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/summarize_stage_c.py \
  outputs/stage_c/<run_id>
```

## 产物规范

Stage C 产物写入 `outputs/stage_c/<run_id>/`，目录名中的 `<run_id>` 必须与 `run_metadata.json.run_id` 一致。

C1 输出：

```text
invariance_metrics.jsonl
invariance_metrics.csv
summary.md
run_metadata.json
```

C2 输出：

```text
kv_metrics.jsonl
kv_metrics.csv
summary_by_method.csv
summary_by_layer.csv
summary.md
run_metadata.json
```

C3 输出：

```text
qjl_metrics.jsonl
qjl_metrics.csv
summary_by_method.csv
summary.md
run_metadata.json
```

C4 输出：

```text
attention_layer_metrics.jsonl
attention_layer_metrics.csv
summary_by_method.csv
summary_by_layer.csv
summary.md
run_metadata.json
```

C5 PPL 输出：

```text
ppl_runs.jsonl
ppl.csv
summary.md
run_metadata.json
```

C5 accuracy 输出：

```text
accuracy_runs.jsonl
accuracy.csv
summary.md
run_metadata.json
```

所有正式 run 需要在 `docs/experiment_runs.md` 中记录 run id、output dir、Git commit / dirty status、核心表格和结论。

## 实验结果

### C1 Invariance Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_105020+0800_stage_c_invariance` |
| Output dir | `outputs/stage_c/20260508_105020+0800_stage_c_invariance` |
| Scope | all 22 layers, max samples 8, sequence length 128 |

| Metric | Value |
| --- | ---: |
| Mean score relative MSE | 0.000000 |
| Mean output cosine | 1.00001176 |

C1 结论：post-RoPE Q/K head-wise H64 rotation 的 no-quant attention score invariance 成立。

### C2 Attention-local KV Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_102337+0800_stage_c_kv_local` |
| Output dir | `outputs/stage_c/20260508_102337+0800_stage_c_kv_local` |
| Scope | all 22 layers, max samples 8, sequence length 128 |

| Method | Score relative MSE | Softmax KL | Note |
| --- | ---: | ---: | --- |
| `absmax_k4v4` | reference baseline | reference baseline | uniform KV baseline |
| `hadamard_lm_k3v4` | 0.011561 | 0.183433 | close to or better than `absmax_k4v4` |
| `hadamard_lm_k2v4` | failure boundary | failure boundary | not promoted |

C2 结论：`Hadamard-LM K3V4` 在 attention-local score / softmax 指标上具备可行性；`K2V4` 是当前失败边界。

### C3 QJL Residual Full Run

| Item | Value |
| --- | --- |
| Run ID | `20260508_104441+0800_stage_c_qjl` |
| Output dir | `outputs/stage_c/20260508_104441+0800_stage_c_qjl` |
| Scope | all 22 layers, max samples 8, sequence length 128 |

| Method family | Result |
| --- | --- |
| `hadamard_lm_k2_qjl` | worse score MSE / softmax KL / top-k overlap / output cosine than pure base |
| `hadamard_lm_k3_qjl` | worse score MSE / softmax KL / top-k overlap / output cosine than pure base |

C3 结论：当前 QJL residual 不优于 pure Hadamard-LM base，因此不进入 C5 默认正式组。

### C4 Attention-layer Structured Full Run

| Item | Value |
| --- | --- |
| Result source | 2026-05-08 value-absorb full run, recorded in `docs/experiment_runs.md` |
| Scope | all 22 Attention layers, max samples 8, sequence length 128 |
| Current entry | `experiments/stage_c_attention_layer.py` |

| Method | Value path | Layer output cosine |
| --- | --- | ---: |
| `attn_identity_fp16` | identity | near 1.0 |
| `attn_rot_lm_w4a4_hlm_k4v4` | o_proj absorb | 0.956899 |
| `attn_rot_lm_w4a3_hlm_k4v3` | o_proj absorb | 0.920949 |

C4 结论：value rotation + `o_proj` absorb 后，structured Attention local output 不再出现旧 reconstruct path 的严重错配；`W4A4 + K4V4` 是当前最稳 structured baseline。

### C5 Attention-only PPL Full Run

| Item | Value |
| --- | --- |
| Result source | 2026-05-08 value-absorb full run, recorded in `docs/experiment_runs.md` |
| Current entry | `experiments/stage_c_ppl.py` |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |

| Method | PPL |
| --- | ---: |
| FP16 | 8.048573 |
| Attn Identity FP16 | 8.048573 |
| Attn KV-HLM K4V4 | 8.204482 |
| Attn KV-HLM K3V4 | 8.658396 |
| Attn Rot-LM W4A4 + HLM K4V4 | 8.614497 |
| Attn Rot-LM W3A4 + HLM K3V4 | 11.355422 |
| Attn Rot-LM W4A3 + HLM K4V3 | 9.387665 |

## 结论

1. C 阶段 Key 量化应放在 RoPE 之后，使用 per-head H64 score-domain rotation。
2. Key 的核心指标是 inner product / attention score / softmax KL，不是 reconstruction MSE；Value 的核心指标是 reconstruction 和 final attention output。
3. C2 显示 `Hadamard-LM K3V4` 在 KV-local 层面可行，但 `K2V4` 和当前 QJL residual 不适合作为 C5 默认组。
4. C4/C5 使用 value rotation + `o_proj` absorb 后，local 指标和 model-level PPL 趋势重新对齐。
5. C5 当前最好结果来自 KV-only `Attn KV-HLM K4V4`，PPL 为 `8.204482`；structured `Attn Rot-LM W4A4 + HLM K4V4` PPL 为 `8.614497`，仍属于可接受数值质量范围。
