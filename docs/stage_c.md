# Stage C Attention/KV Cache Experiments

本文档记录 C 阶段的实验计划、代码框架、运行入口、产物规范和结果记录模板。目标对应 `第一阶段实验思路.md` 中的 C 线：

> 验证 Hadamard rotation + Lloyd-Max 是否能在更低 bit-width 下保持 KV cache 的 attention score 和 attention output；进一步验证 QJL residual 是否能修正 Key quantization 的 inner product 误差；最后把 q/k/v/o 线性 fake quant 与 post-RoPE rotated KV cache quant 串成完整 Attention 层，评估准确度和 PPL。

Stage C 仍只解释数值质量和低比特可行性，不把 Lloyd-Max fake quant、QJL sign sketch 或 KV cache fake quant 解释成真实推理加速。真实收益需要后续 bit packing、codebook-aware MAC、low-bit attention kernel 或专用硬件支持。

## 实验模型

| Item | Value |
| --- | --- |
| Model repo | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Local path | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Architecture | LLaMA, 22 layers, hidden size 2048, intermediate size 5632 |
| Attention | 32 query heads, 4 KV heads, head dim 64 |
| GQA rule | K/V heads are repeated to match Q heads before attention score |
| Stage C target modules | 22 self-attention modules, including q/k/v/o projections and KV cache tensors |
| W/A block size | 128 for q/k/v/o Linear fake quant |
| KV head rotation size | 64, equal to head dim |

TinyLlama 的 `q_proj` / `o_proj` shape 为 `[2048, 2048]`，`k_proj` / `v_proj` shape 为 `[256, 2048]`。q/k/v projection 输出 reshape 后分别是 Q `[batch, seq, 32, 64]`、K/V `[batch, seq, 4, 64]`。C 阶段的 KV cache 量化在 head dim 上做 normalized Hadamard rotation，因此使用 `head_dim=64` 的 head-wise rotation，而不是 B 阶段默认的 `block_size=128`。

## 术语与编号

为避免与 `第一阶段实验思路.md` 中后期追加的 C3/C4/C5 发生混乱，本文档采用下面的 Stage C 编号：

| Stage C 编号 | 对应内容 | 说明 |
| --- | --- | --- |
| C0 | 约束与 RoPE 策略 | 明确 post-RoPE KV cache quantization 和 fake-quant 边界 |
| C1 | Attention capture + invariance sanity | 抽取 Q/K/V、验证旋转不变量 |
| C2 | Attention-local KV quant | 对应原 C3，测试 K/V low-bit |
| C3 | QJL residual | 对应原 C4，只修正 Key inner product |
| C4 | Attention-layer structured quant | 对应原 C5-layer，整层 Attention 输出误差 |
| C5 | Attention-only model-level PPL / accuracy | 对应原 C5-model，只替换 Attention，不替换 FFN |

## C0 约束与 RoPE 策略

LLaMA attention 中 query 和 key 会经过 RoPE。QuaRot 的 KV cache 路径应理解为 **post-RoPE caching**：

```text
Q = X W_q
K = X W_k
V = X W_v
Q_rope = RoPE(Q)
K_rope = RoPE(K)
```

然后在每个 head 的 64 维向量上使用相同的 orthonormal Hadamard rotation：

```text
Q_H = Q_rope H
K_H = K_rope H
```

由于 `H` 正交，未量化时有：

```text
Q_rope K_rope^T == Q_H K_H^T
```

因此 C 阶段采用：

> post-RoPE Q/K + head-wise Hadamard rotation + rotated key cache quantization。

Key cache 存储 rotated and quantized `K_H`；decode 时当前 query 在 RoPE 后在线旋转为 `Q_H`，再与 dequantized rotated key cache 做 dot product。Value 不经过 RoPE，第一阶段先采用 value reconstruction 路径：`V -> V_H -> Q(V_H) -> inverse rotate -> V_hat`，再计算 `P V_hat`。这保持评估简单，后续若需要更贴近完整结构适配，再考虑把 value rotation 与 `o_proj` 权重吸收串起来。

首版 rotation operator 默认取 deterministic `H_64`，与 Stage B 保持一致；代码中预留 seeded random sign diagonal `D`，可切换为 `R_64 = H_64 D`。因为 `R_64` 仍为正交变换，no-quant attention score invariance 不变。是否启用 `D` 必须写入 `run_metadata.json`。

## 实验组

### C1 Attention Capture + Invariance Sanity

C1 是所有后续 C 线实验的前置检查。

采集对象：

| Tensor | Shape after reshape | 位置 |
| --- | --- | --- |
| `attn_input` | `[batch, seq, 2048]` | self-attention input |
| `q_proj_out` | `[batch, seq, 32, 64]` | before RoPE |
| `k_proj_out` | `[batch, seq, 4, 64]` | before RoPE |
| `v_proj_out` | `[batch, seq, 4, 64]` | value cache candidate |
| `q_rope` | `[batch, seq, 32, 64]` | after RoPE |
| `k_rope` | `[batch, seq, 4, 64]` | after RoPE |
| `attn_scores` | `[batch, 32, seq, seq]` | after GQA repeat and causal mask |
| `attn_probs` | `[batch, 32, seq, seq]` | softmax output |
| `attn_output` | `[batch, seq, 2048]` | before / after `o_proj`, both recorded when possible |

Sanity checks：

| Check | Expected |
| --- | --- |
| `H^T H` | identity within fp tolerance |
| `q_rope @ k_rope.T` vs `(q_rope H) @ (k_rope H).T` | relative error close to 0 |
| no-quant structured attention vs FP16 attention | output error close to numerical precision |
| GQA repeat placement | repeat K/V after head-wise quant/dequant, before score/value matmul |

C1 输出指标：

* score relative MSE；
* max absolute score difference；
* attention output relative MSE；
* attention output cosine；
* per-layer sanity pass / fail。

### C2 Attention-local KV Quant

C2 只测试 KV cache 量化，不替换 q/k/v/o 线性层。它回答：

> Hadamard-LM 是否让 K3V4、K4V3、K3V3 更可用？

Hadamard-LM 的 scale policy 与 Stage B 的 block RMS 思路保持一致，但粒度改为 per token / per head：

```text
k_scale = rms(k_rope, dim=head_dim)
k_norm = k_rope / k_scale
k_R = R_64 k_norm
k_R_hat = Q_LM(k_R)
score_hat = (R_64 q_rope) @ k_R_hat.T * k_scale / sqrt(d_h)
```

Value 侧同样先按 per token / per KV head RMS 标准化，再旋转量化并反旋回原 value 空间：

```text
v_scale = rms(v, dim=head_dim)
v_norm = v / v_scale
v_R = R_64 v_norm
v_R_hat = Q(v_R)
v_hat = R_64^T v_R_hat * v_scale
```

这些 scale / norm 属于 metadata，不计入简单的 K/V bit label，但会在 CSV 与 summary 中单独记录。

实验组：

| Method key | K/V bits | Quantizer type | Compute interpretation |
| --- | --- | --- | --- |
| `fp16` | K16V16 | none | baseline |
| `absmax_k4v4` | K4V4 | uniform fake quant | integer-like KV baseline |
| `absmax_k3v4` | K3V4 | uniform fake quant | key bit reduction baseline |
| `absmax_k4v3` | K4V3 | uniform fake quant | value bit reduction baseline |
| `hadamard_lm_k4v4` | K4V4 | head-wise rotation + Lloyd-Max | non-uniform fake quant |
| `hadamard_lm_k3v4` | K3V4 | head-wise rotation + Lloyd-Max | key bit reduction |
| `hadamard_lm_k4v3` | K4V3 | head-wise rotation + Lloyd-Max | value bit reduction |
| `hadamard_lm_k3v3` | K3V3 | head-wise rotation + Lloyd-Max | aggressive KV combination |
| `hadamard_lm_k2v4` | K2V4 | head-wise rotation + Lloyd-Max | failure boundary |

Key 指标优先级：

1. inner product bias；
2. inner product variance；
3. attention score MSE；
4. softmax KL；
5. top-k attention overlap；
6. K reconstruction MSE。

Value 指标优先级：

1. V reconstruction relative MSE；
2. attention output relative MSE；
3. attention output cosine。

核心判断：

> `Hadamard-LM K3V4` 或 `Hadamard-LM K3V3` 是否接近或优于 `Absmax K4V4` / `Hadamard-LM K4V4` 的关键 attention 指标？

### C3 QJL Residual

C3 只对 Key 做 QJL residual correction，Value 固定为 V4 absmax 或 V4 Hadamard-LM。首版默认固定 `Hadamard-LM V4`，因为它与 C2 的 rotated KV 主线一致。

Base reconstruction 使用与 C2 一致的 per token / per head scale：

```text
K_scale = rms(K_rope, dim=head_dim)
K_norm = K_rope / K_scale
K_R = R_64 K_norm
K_R_base = Q_LM(K_R)
K_base = R_64^T K_R_base * K_scale
Residual = K_rope - K_base
```

QJL residual estimator：

```text
sign_code = sign(S Residual)
q_dot_residual_hat = sqrt(pi / 2) / m * ||Residual||_2 * <S q_rope, sign_code>
score_hat = q_rope @ K_base.T + q_dot_residual_hat
```

首版默认：

| Item | Default |
| --- | --- |
| Projection matrix | Gaussian random matrix `S` |
| Projection dim | `m = d_h = 64` |
| Random seed | recorded in `run_metadata.json` |
| Residual bits | 1 sign bit per projected coordinate |
| Norm metadata | record separately, not folded into base K bits |

实验组：

| Method key | Base K bits | Residual bits | Value | Effective description |
| --- | ---: | ---: | --- | --- |
| `hadamard_lm_k3` | 3 | 0 | V4 fixed | pure reconstruction |
| `hadamard_lm_k2` | 2 | 0 | V4 fixed | aggressive base |
| `hadamard_lm_k2_qjl` | 2 | 1 | V4 fixed | inner-product corrected |
| `hadamard_lm_k3_qjl` | 3 | 1 | V4 fixed | stronger corrected baseline |

核心判断：

> 在相同或相近 bit budget 下，`Hadamard-LM + QJL residual` 的 attention score MSE / softmax KL 是否明显低于 pure `Hadamard-LM`？

### C4 Attention-layer Structured Quant

C4 把 Attention 层内部的线性计算和 KV cache 量化串起来，但仍只在单层或少数层上评估输出误差。

结构：

```text
X
  -> q/k/v Linear W/A fake quant
  -> Q, K, V
  -> RoPE(Q, K)
  -> head-wise Q/K rotation
  -> quantized rotated K cache
  -> quantized V cache
  -> attention probs and attention output
  -> o_proj Linear W/A fake quant
  -> Y_hat
```

q/k/v/o 线性层沿用 Stage B 的 W/A fake quant：

```text
x W ~= Q_A(x H_128) Q_W(H_128^T W)
```

KV cache 沿用 C2/C3 的 post-RoPE head-wise quant：

```text
score_hat = (Q_rope H_64) @ Q_KV(K_rope H_64).T / sqrt(d_h)
```

实验组：

| Method key | Attention Linear | KV cache | 说明 |
| --- | --- | --- | --- |
| `fp16` | FP16 | FP16 | baseline |
| `attn_direct_absmax_w4a4_absmax_k4v4` | direct W4A4 | absmax K4V4 | 普通 attention baseline |
| `attn_rot_absmax_w4a4_hlm_k4v4` | Rot-Absmax W4A4 | Hadamard-LM K4V4 | rotation + 4-bit KV |
| `attn_rot_lm_w4a4_hlm_k4v4` | Rot-LM W4A4 | Hadamard-LM K4V4 | 同 bit 非均匀 codebook |
| `attn_rot_lm_w3a4_hlm_k3v4` | Rot-LM W3A4 | Hadamard-LM K3V4 | 核心低 bit 组合 |
| `attn_rot_lm_w4a3_hlm_k4v3` | Rot-LM W4A3 | Hadamard-LM K4V3 | activation/value 降位宽 |
| `attn_rot_lm_w3a4_hlm_k2qjl_v4` | Rot-LM W3A4 | Hadamard-LM K2 + QJL, V4 | C3 成功后启用 |

输出：

* q/k/v projection output error；
* post-RoPE score MSE；
* softmax KL；
* top-k attention overlap；
* pre-`o_proj` attention output error；
* final attention layer output relative MSE / cosine；
* per-layer sensitivity。

核心判断：

> `Attn Rot-LM W3A4 + Hadamard-LM K3V4` 是否接近或优于 `Attn Rot-Absmax W4A4 + Hadamard-LM K4V4`？

### C5 Attention-only Model-level PPL / Accuracy

C5 在模型中只替换 Attention，不替换 FFN。它回答：

> 当 q/k/v/o 线性 fake quant 和 KV cache low-bit quant 串成完整 Attention 层时，PPL 和 accuracy 是否仍可接受？

实验组：

| Method key | Attention Linear | KV cache | PPL | Accuracy | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| `fp16` | FP16 | FP16 | | | baseline |
| `attn_direct_absmax_w4a4_absmax_k4v4` | direct uniform W/A | absmax KV | | | 普通 attention baseline |
| `attn_rot_absmax_w4a4_hlm_k4v4` | rotation + uniform W/A | rotated KV | | | 计算不变性 + 4-bit KV |
| `attn_rot_lm_w4a4_hlm_k4v4` | rotation + LM W/A | rotated KV | | | 同 bit 非均匀 codebook |
| `attn_rot_lm_w3a4_hlm_k3v4` | weight 降位宽 | key 降位宽 | | | 核心低 bit 组合 |
| `attn_rot_lm_w4a3_hlm_k4v3` | activation 降位宽 | value 降位宽 | | | A/V 降位宽 |
| `attn_rot_lm_w3a4_hlm_k2qjl_v4` | weight 降位宽 | QJL corrected key | | | C3 成功后的组合 |

默认 PPL 设置与 Stage A/B 对齐：

| Item | Value |
| --- | --- |
| Dataset | WikiText2 raw test split |
| Max samples | 512 |
| Sequence length / stride | 2048 / 2048 |
| Device | `mps` for formal runs |

Accuracy 首版建议作为可选脚本实现，默认先跑一个小规模 zero-shot benchmark；具体 benchmark 需要确认。若没有额外指定，建议先用 `lambada_openai` 或 `piqa` 之一，避免一次性引入过多评测依赖。

核心判断：

> Attention structured quant 的 PPL 是否不明显劣于 FFN-only quant；`K2+QJL,V4` 是否能接近或优于 pure `K3V4`。

## 代码框架

| Path | Role |
| --- | --- |
| `src/rotationquant/stage_c.py` | Stage C method registry、KV quant spec、head-wise Hadamard、post-RoPE K/V quant、score/output metrics、QJL estimator |
| `src/rotationquant/attention_capture.py` | TinyLlama attention wrapper / hook，采集 q/k/v、RoPE 后 Q/K、score、prob、attention output |
| `src/rotationquant/stage_c_model.py` | Attention-only model wrapper，把 q/k/v/o W/A fake quant 与 KV cache fake quant 接入 model-level PPL |
| `src/rotationquant/stage_b.py` | 复用 q/k/v/o Linear 的 W/A fake quant 函数 |
| `src/rotationquant/rotations.py` | 复用 FWHT；补充或复用 head-dim 64 Hadamard helper |
| `src/rotationquant/quantizers.py` | 复用 symmetric absmax 与 Gaussian Lloyd-Max fake quant |
| `src/rotationquant/metrics.py` | 复用 tensor metrics；补充 score bias/variance、softmax KL、top-k overlap |
| `src/rotationquant/ppl.py` | causal LM sliding-window PPL |
| `src/rotationquant/run_metadata.py` | 统一记录 run id、时间、Git 状态、包版本和 torch runtime |
| `experiments/stage_c_invariance.py` | C1 capture 与 no-quant invariance sanity |
| `experiments/stage_c_kv_local.py` | C2 Attention-local KV quant sweep |
| `experiments/stage_c_qjl.py` | C3 QJL residual sweep |
| `experiments/stage_c_attention_layer.py` | C4 structured Attention layer output error |
| `experiments/stage_c_ppl.py` | C5 Attention-only model-level PPL |
| `experiments/stage_c_accuracy.py` | C5 optional zero-shot accuracy |
| `experiments/summarize_stage_c.py` | 汇总 C1/C2/C3/C4/C5 CSV 到 summary |
| `scripts/run_stage_c_invariance.sh` | C1 shell 入口 |
| `scripts/run_stage_c_kv_local.sh` | C2 shell 入口 |
| `scripts/run_stage_c_qjl.sh` | C3 shell 入口 |
| `scripts/run_stage_c_attention_layer.sh` | C4 shell 入口 |
| `scripts/run_stage_c_ppl.sh` | C5 PPL shell 入口 |
| `scripts/run_stage_c_accuracy.sh` | C5 accuracy shell 入口 |

实现注意：

1. C1 capture 最稳妥的方式是包装 `LlamaAttention` forward，而不是只靠普通 module hook，因为 RoPE 后 Q/K 和 causal score 是 forward 内部临时张量。
2. GQA 中 K/V 只有 4 个 heads；KV quant 应先在 4 个 KV heads 上做，再按 TinyLlama 原逻辑 repeat 到 32 个 query heads。
3. C2/C3 的 attention score 需要同时记录 raw score error 和 causal-mask 后 softmax KL；最终判断以后者为主。
4. C4/C5 仍是 fake quant：权重和 activation 反量化回浮点后做 PyTorch matmul，不代表 native low-bit compute。
5. QJL estimator 可能比 C2 更慢；首版先在 C3 local 上验证 score 指标，再决定是否接入 C4/C5。

## 运行入口

C1 invariance sanity：

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

当前 Codex 沙箱内无法枚举 MPS/Metal GPU；需要用 GPU 跑 C5 PPL 时，应授权外部执行，并传入：

```bash
--device mps
```

## 建议执行顺序

1. C1 smoke：layer 0，`max_samples=2`，`sequence_length=64`，确认 capture 和 no-quant invariance。
2. C2 smoke：layer 0，`max_samples=2`，`sequence_length=64`，确认 KV quant 指标和输出文件。
3. C3 smoke：layer 0，`max_samples=2`，`sequence_length=64`，确认 QJL estimator 的 bias / KL 计算。
4. C4 smoke：layer 0，`max_samples=2`，`sequence_length=64`，确认 structured Attention wrapper。
5. C5 smoke：`fp16` + 一个核心 method，`max_samples=2`，`sequence_length=64`，确认 model-level wrapper 和 PPL。
6. C2 full：all 22 layers，`max_samples=8` 或 `32`，`sequence_length=128` 或 `512`。
7. C3 full：优先少数敏感层，再扩大到 all 22 layers。
8. C4 full：all 22 layers local output error。
9. C5 full PPL：WikiText2 raw test，`max_samples=512`，`sequence_length=2048`，`stride=2048`。
10. C5 accuracy：待 benchmark 确认后运行。

## 产物规范

Stage C 脚本会在 `outputs/stage_c/<run_id>/` 下写入数据。目录名中的 `<run_id>` 与 `run_metadata.json` 中的 `run_id` 字段保持一致。

C1 输出：

```text
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_invariance>/invariance_metrics.jsonl
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_invariance>/invariance_metrics.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_invariance>/summary.md
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_invariance>/run_metadata.json
```

C2 输出：

```text
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>/kv_metrics.jsonl
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>/kv_metrics.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>/summary_by_method.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>/summary_by_layer.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>/summary.md
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_kv_local>/run_metadata.json
```

C3 输出：

```text
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_qjl>/qjl_metrics.jsonl
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_qjl>/qjl_metrics.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_qjl>/summary_by_method.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_qjl>/summary.md
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_qjl>/run_metadata.json
```

C4 输出：

```text
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>/attention_layer_metrics.jsonl
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>/attention_layer_metrics.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>/summary_by_method.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>/summary_by_layer.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>/summary.md
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_attention_layer>/run_metadata.json
```

C5 PPL 输出：

```text
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_ppl>/ppl_runs.jsonl
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_ppl>/ppl.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_ppl>/summary.md
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_ppl>/run_metadata.json
```

C5 accuracy 输出：

```text
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_accuracy>/accuracy_runs.jsonl
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_accuracy>/accuracy.csv
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_accuracy>/summary.md
outputs/stage_c/<YYYYMMDD_HHMMSS+0800_stage_c_accuracy>/run_metadata.json
```

每次正式 Stage C 运行后，在 `docs/experiment_runs.md` 追加 run id、output dir、Git commit、dirty status、核心表格和结论。

## 当前结果模板

### C1 Invariance Sanity

| Layer | Score rel MSE | Max score diff | Output rel MSE | Output cosine | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | | | | | |
| all mean | | | | | |

### C2 Attention-local KV Error

| Method | K bits | V bits | IP bias | IP var | Score MSE | Softmax KL | Top-k overlap | V MSE | Output cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absmax | 4 | 4 | | | | | | | |
| Absmax | 3 | 4 | | | | | | | |
| Absmax | 4 | 3 | | | | | | | |
| Hadamard-LM | 4 | 4 | | | | | | | |
| Hadamard-LM | 3 | 4 | | | | | | | |
| Hadamard-LM | 4 | 3 | | | | | | | |
| Hadamard-LM | 3 | 3 | | | | | | | |
| Hadamard-LM | 2 | 4 | | | | | | | |

### C3 QJL Residual

| Method | Base K bits | Residual bits | Value | IP bias | Score MSE | Softmax KL | Top-k overlap | Output cosine |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Hadamard-LM K3 | 3 | 0 | V4 fixed | | | | | |
| Hadamard-LM K2 | 2 | 0 | V4 fixed | | | | | |
| Hadamard-LM K2 + QJL residual | 2 | 1 | V4 fixed | | | | | |
| Hadamard-LM K3 + QJL residual | 3 | 1 | V4 fixed | | | | | |

### C4 Attention-layer Structured Quant

| Method | Attention Linear | KV cache | Projection err | Score MSE | Softmax KL | Layer output err | Output cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | FP16 | FP16 | | | | | |
| Direct-Absmax | W4A4 | Absmax K4V4 | | | | | |
| Rot-Absmax | W4A4 | Hadamard-LM K4V4 | | | | | |
| Rot-LM | W4A4 | Hadamard-LM K4V4 | | | | | |
| Rot-LM | W3A4 | Hadamard-LM K3V4 | | | | | |
| Rot-LM | W4A3 | Hadamard-LM K4V3 | | | | | |
| Rot-LM | W3A4 | Hadamard-LM K2+QJL,V4 | | | | | |

### C5 Attention-only PPL / Accuracy

| Method | Attention Linear | KV cache | PPL | Accuracy | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| FP16 | FP16 | FP16 | | | baseline |
| Attention Direct-Absmax W4A4 + Absmax K4V4 | direct uniform W/A | absmax KV | | | 普通 attention baseline |
| Attention Rot-Absmax W4A4 + Hadamard-LM K4V4 | rotation + uniform W/A | rotated KV | | | 计算不变性 + 4-bit KV |
| Attention Rot-LM W4A4 + Hadamard-LM K4V4 | rotation + LM W/A | rotated KV | | | 同 bit 非均匀 codebook |
| Attention Rot-LM W3A4 + Hadamard-LM K3V4 | weight 降位宽 | key 降位宽 | | | 核心低 bit 组合 |
| Attention Rot-LM W4A3 + Hadamard-LM K4V3 | activation 降位宽 | value 降位宽 | | | A/V 降位宽 |
| Attention Rot-LM W3A4 + Hadamard-LM K2+QJL,V4 | weight 降位宽 | QJL corrected key | | | C3 成功后的组合 |

## Implementation Smoke Results

| Stage | Run ID | Scope | Key observation |
| --- | --- | --- | --- |
| C1 | `20260508_100235+0800_stage_c_invariance` | layer 0, max samples 2, sequence length 64 | no-quant head-wise Hadamard invariance passes; score relative MSE `0.0`, max score diff `1.67e-06` |
| C2 | `20260508_100425+0800_stage_c_kv_local` | layer 0, selected KV methods | `Hadamard-LM K4V4` score relative MSE `0.007246`; `Hadamard-LM K3V4` score relative MSE `0.029824` |
| C3 | `20260508_100709+0800_stage_c_qjl` | layer 0, K2/K3 pure and QJL | QJL path runs; this tiny sample does not show QJL improvement, so QJL is not promoted to C5 full by default |
| C4 | `20260508_101201+0800_stage_c_attention_layer` | layer 0, `fp16` and `attn_rot_lm_w4a4_hlm_k4v4` | structured Attention wrapper runs; quantized method layer output relative MSE `0.042903`, cosine `0.978361` |
| C5 | `20260508_101252+0800_stage_c_ppl` | short-context PPL path check, max samples 2, sequence length 64 | Attention-only model wrapper runs on MPS; result is a path check, not a formal quality conclusion |

## Formal Full Results

详细人工记录见 `docs/experiment_runs.md` 中的 `2026-05-08：Stage C Full Runs`。

| Stage | Run ID | Scope | Key observation |
| --- | --- | --- | --- |
| C1 | `20260508_105020+0800_stage_c_invariance` | all 22 layers, max samples 8, sequence length 128 | post-RoPE Q/K head-wise Hadamard invariance passes across all layers; mean score relative MSE `0.0`, mean output cosine `1.00001176` |
| C2 | `20260508_102337+0800_stage_c_kv_local` | all 22 layers, max samples 8, sequence length 128 | `Hadamard-LM K3V4` score relative MSE `0.011561` and softmax KL `0.183433`, close to or better than `Absmax K4V4`; `K2V4` remains a failure boundary |
| C3 | `20260508_104441+0800_stage_c_qjl` | all 22 layers, max samples 8, sequence length 128 | QJL worsens score MSE/KL/top-k/output cosine for both K2 and K3 bases, so it is not promoted to C5 |
| C4 | `20260508_102632+0800_stage_c_attention_layer` | all 22 Attention layers, max samples 8, sequence length 128 | `Rot-LM W4A4 + HLM K4V4` layer output cosine `0.947851`; `Rot-LM W3A4 + HLM K3V4` cosine `0.864589`, better than absmax baselines but visibly degraded |
| C5 | `20260508_102751+0800_stage_c_ppl` | WikiText2, max samples 512, sequence length/stride 2048/2048 | current Attention-only full-model wrapper does not preserve PPL: FP16 `8.048573`, best quantized method still `2758.251695` |

Formal C5 PPL:

| Method | PPL |
| --- | ---: |
| FP16 | 8.048573 |
| Attention Direct Absmax W4A4 + Absmax K4V4 | 2758.251695 |
| Attention Rot Absmax W4A4 + HLM K4V4 | 17607.464742 |
| Attention Rot-LM W4A4 + HLM K4V4 | 12044.785422 |
| Attention Rot-LM W3A4 + HLM K3V4 | 8105.302082 |
| Attention Rot-LM W4A3 + HLM K4V3 | 8362.464387 |

## 当前总结

1. C 阶段的 Key 量化应放在 RoPE 之后：先得到真实 `Q_rope` / `K_rope`，再进行 head-wise Hadamard rotation 和 rotated key cache quantization。
2. C 阶段确实需要 Attention 计算不变性适配，但应分层推进：先验证 no-quant invariance，再做 KV local，再做 QJL residual，最后进入完整 Attention 层和 model-level PPL。
3. Key 的主要指标是 inner product / attention score / softmax KL，不是 reconstruction MSE；Value 的主要指标更接近 reconstruction 和 final attention output。
4. Stage C 的核心低位宽判断是：`Hadamard-LM K3V4` 在 KV-local 层面已经接近或优于 uniform `K4V4`，但 all-layer C3 显示 `K2+QJL,V4` 和 `K3+QJL,V4` 当前都不优于 pure Hadamard-LM base。
5. C4 local Attention 支持 Rot-LM 作为数值质量更好的 W/A fake quant 路线，但 C5 full-model PPL 暂时失败，说明整模型 Attention-only 替换还需要进一步结构适配或误差定位。
6. C5 的结论边界与 A/B 相同：它能说明 attention structured fake quant 的数值可行性边界，但不能说明真实 KV cache 或 attention kernel 加速。

## 待确认问题

1. C5 accuracy benchmark：默认我建议先选 `lambada_openai` 或 `piqa` 的小规模 zero-shot accuracy；如果你更关心 commonsense / reading comprehension / code，可改成别的 benchmark。
2. C3 QJL projection dim：首版默认 `m=d_h=64`，便于把 residual sign sketch 粗略理解成 +1 bit/channel；后续是否要同时扫 `m=32` / `m=16` 取决于你是否更关注压缩率。
3. C5 首版默认评估 prefill-style causal attention；decode-time KV cache 追加 token 的路径可以作为后续专项实验。
