# Stage 2 实验思路：全模型 FFN + Attention Fake Quant

## 1. 实验目标

Stage 2 目标是在 TinyLlama 上同时替换 FFN 与 Attention，直接评估全模型 W/A/KV fake quant 后的 WikiText2 PPL。该阶段重点回答四个问题：

1. 全模型 4-bit fake quant 是否仍具备可接受的数值质量；
2. 低比特退化主要来自 FFN 还是 Attention/KV cache；
3. Hadamard、randomized Hadamard、random orthogonal 三类旋转后端在端到端 PPL 上是否存在稳定差异；
4. Absmax、MXFP4、Lloyd-Max 三类量化器在全模型设置下的相对表现如何。

本阶段仍只解释 fake quant 的数值质量和低比特可行性，不解释为真实吞吐、访存或 low-bit kernel 加速。

## 2. Stage 1 依据

Stage 1 的 A/B/C 结果给出如下依据：

| 结论 | 对 Stage 2 的影响 |
| --- | --- |
| FFN W/A 低比特相对更稳 | Stage 2 增加 mixed precision 组，优先降低 FFN 位宽 |
| Attention/KV 对低比特更敏感 | Attention 主组先保持 W4A4 + K4V4，再探索 W3A4/K3V4 |
| MXFP4 是强 4-bit baseline | Stage 2 必须保留 MXFP4 W4A4 对照 |
| Rot-LM W4A4 在 B/C 中最稳 | `full_rot_lm_w4a4_hlm_k4v4` 作为主方法 |
| randomized Hadamard 与普通 Hadamard 接近，random orthogonal 不稳定 | Stage 2 做 backend 消融，但不预设改变主线 |

## 3. 方法命名规则

全模型方法由一个 FFN method 和一个 Attention method 组合而成。文档中的逻辑命名为：

```text
full_<ffn_method>__<attn_method>
```

代码中为了保持 CSV 和 shell 参数紧凑，使用等价短名：

```text
full_rot_lm_w4a4_hlm_k4v4
```

报告展示时可简写为：

```text
Full Rot-LM W4A4 + HLM K4V4
```

命名中的含义如下：

| 字段 | 含义 |
| --- | --- |
| `direct_absmax` | FFN/Attention 线性层 W/A 使用 block-wise absmax fake quant |
| `mxfp4` | FFN/Attention 线性层 W/A 使用 MXFP4 E2M1 fake quant |
| `rot_absmax` | 旋转域 W/A 使用 absmax fake quant |
| `rot_mxfp4` | 旋转域 W/A 使用 MXFP4 fake quant |
| `rot_lm` | Hadamard rotation + Lloyd-Max fake quant |
| `randhadamard_lm` | randomized Hadamard + Lloyd-Max fake quant |
| `randortho_lm` | dense random orthogonal + Lloyd-Max fake quant |
| `hlm_k4v4` | KV cache 使用 head-wise Hadamard-LM K4V4 |
| `h32` | KV head 内使用两个 H32 block，而不是默认 H64 |

## 4. 代码框架

| 文件 | 作用 |
| --- | --- |
| `src/rotationquant/stage_full.py` | 定义 `FullModelQuantSpec`，组合 FFN 与 Attention fake quant 方法 |
| `experiments/stage_full_ppl.py` | 全模型 PPL 入口，对同一个模型依次替换 Attention 与 FFN |
| `scripts/run_stage_full_ppl_smoke.sh` | 小规模 smoke run |
| `scripts/run_stage_full_ppl.sh` | 正式全模型 PPL 入口 |

全模型替换顺序固定为先替换 Attention，再替换 FFN。两类 wrapper 分别只接管 decoder layer 中的 `self_attn` 与 `mlp`，不修改 residual stream、layer norm 或 decoder layer 外层计算。

## 5. 实验组

### 5.1 Sanity

| Method | 目的 |
| --- | --- |
| `fp16` | 原始模型基线 |
| `full_identity_attention_fp16` | 只替换 identity Attention wrapper，不量化 FFN，用于确认 full entry 本身不改变基线 |

### 5.2 4-bit 全模型主组

| Method | 说明 |
| --- | --- |
| `full_direct_absmax_w4a4_absmax_k4v4` | FFN/Attention W/A 使用 block absmax，KV 使用 absmax K4V4 |
| `full_mxfp4_w4a4_hlm_k4v4` | FFN/Attention W/A 使用 MXFP4，KV 使用 HLM K4V4 |
| `full_rot_absmax_w4a4_hlm_k4v4` | 旋转域 uniform absmax baseline |
| `full_rot_mxfp4_w4a4_hlm_k4v4` | Hadamard rotation + MXFP4 |
| `full_rot_lm_w4a4_hlm_k4v4` | Hadamard rotation + Lloyd-Max，Stage 2 主方法 |

### 5.3 Rotation backend 消融

| Method | 说明 |
| --- | --- |
| `full_rot_lm_w4a4_hlm_k4v4` | 普通归一化 Hadamard |
| `full_randhadamard_lm_w4a4_hlm_k4v4` | randomized Hadamard |
| `full_randortho_lm_w4a4_hlm_k4v4` | dense random orthogonal，仅作为数值消融 |

只有当 randomized Hadamard 或 random orthogonal 在 full PPL 与 local 指标上同时稳定优于 Hadamard 时，才考虑改变主线结论。

### 5.4 低比特探索组

| Method | 目的 |
| --- | --- |
| `full_rot_lm_w3a4_hlm_k3v4` | 同时降低 weight 与 Key 位宽 |
| `full_rot_lm_w4a3_hlm_k4v3` | 同时降低 activation 与 Value 位宽 |
| `full_rot_lm_w3a3_hlm_k3v3` | 全模型 W/A/KV 更激进低比特 |
| `full_rot_lm_w2a4_hlm_k2v4` | failure-boundary smoke，不进入正式长表 |

### 5.5 Mixed precision 探索组

| Method | 目的 |
| --- | --- |
| `full_mixed_ffn_w3a4_attn_w4a4_k4v4` | FFN 更激进，Attention 保守 |
| `full_mixed_ffn_w4a3_attn_w4a4_k4v4` | 只降低 FFN activation |
| `full_mixed_ffn_w4a4_attn_w3a4_k3v4` | 只降低 Attention/Key，用于验证 C 线敏感性 |
| `full_rot_lm_w4a4_hlm_k4v4_h32` | 全模型 W4A4 下比较 KV H32 与 H64 |

## 6. 默认运行设置

| 项目 | 设置 |
| --- | --- |
| 模型 | `models/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| 数据集 | WikiText2 raw test |
| PPL 设置 | `max_samples=512`, `sequence_length=2048`, `stride=2048` |
| 设备 | `mps` |
| W/A block size | 64 |
| MXFP4 group size | 32 |
| KV rotation | 默认 head-wise H64 |
| rotation seed | 0 |

Block size 消融只对 `full_rot_lm_w4a4_hlm_k4v4` 与 `full_rot_lm_w3a4_hlm_k3v4` 额外运行 `block_size=32/128`。其余组先固定 `block_size=64`，避免实验矩阵失控。

## 7. 输出规范

正式产物写入：

```text
outputs/stage2_full/<run_id>/
```

每个 run 目录包含：

| 文件 | 内容 |
| --- | --- |
| `ppl.csv` | 所有方法的 PPL 与方法元信息 |
| `ppl_runs.jsonl` | 每个 method 的逐条运行记录 |
| `summary.md` | 简明结果表 |
| `run_metadata.json` | run id、时间、Git commit、dirty status、参数与依赖版本 |

目录名必须与 `run_metadata.json` 中的 `run_id` 一致。

## 8. 执行顺序

1. 先跑 smoke：

```bash
scripts/run_stage_full_ppl_smoke.sh
```

2. 若 smoke 中 `fp16`、`full_direct_absmax_w4a4_absmax_k4v4`、`full_rot_lm_w4a4_hlm_k4v4` 均能完成，再跑 4-bit 主组：

```bash
scripts/run_stage_full_ppl.sh
```

3. 若 `full_rot_lm_w4a4_hlm_k4v4` 的 PPL 可接受，再通过 `experiments/stage_full_ppl.py --methods ...` 补跑低比特、mixed precision 与 block size 消融。建议先分别运行 backend 消融、低比特探索和 mixed precision，避免单个长 run 失败后影响全部结果记录。

4. 若 4-bit 主组已经明显崩溃，先停止正式矩阵，检查 wrapper 叠加、residual domain、Attention 结构适配、KV value path 与 block size。

## 9. 判断标准

| 问题 | 判断标准 |
| --- | --- |
| 4-bit 全模型是否可行 | `full_rot_lm_w4a4_hlm_k4v4` PPL 明显优于 MXFP4 / Absmax baseline，并尽量接近 FP16 |
| 低 bit 是否可行 | mixed 低比特组优于统一 W3A4/K3V4，说明低 bit 更适合模块选择或层级分配 |
| backend 是否改变主线 | randomized Hadamard / random orthogonal 在 PPL 与 local 指标上均稳定优于 Hadamard |
| KV H32 是否值得保留 | `full_rot_lm_w4a4_hlm_k4v4_h32` PPL 不差于 H64，且 local 指标无明显退化 |

## 10. 结果记录

本轮正式结果汇总文件：

| 文件 | 说明 |
| --- | --- |
| `outputs/stage2_full/stage2_full_results_summary.csv` | 全部 Stage 2 PPL 结果总表 |
| `outputs/stage2_full/stage2_full_results_summary.md` | Markdown 汇总表 |

正式 run 记录如下：

| Run id | 内容 |
| --- | --- |
| `20260513_112908+0800_stage_full_ppl` | sanity 与 4-bit 主组 |
| `20260513_114736+0800_stage_full_ppl` | rotation backend 与部分低比特组；该 run 在 `full_rot_lm_w3a3_hlm_k3v3` 前因 registry 漏项停止，已保留完成方法的 `ppl_runs.jsonl` |
| `20260513_120509+0800_stage_full_ppl` | registry 修复后的低比特、mixed precision 与 KV H32 续跑 |
| `20260513_122403+0800_stage_full_ppl` | W/A block size 32 消融 |
| `20260513_123331+0800_stage_full_ppl` | W/A block size 128 消融 |

### 10.1 4-bit 全模型主结果

| Method | FFN | Attention | KV | PPL |
| --- | ---: | ---: | ---: | ---: |
| `fp16` | FP16 | FP16 | K16V16 | 8.049 |
| `full_identity_attention_fp16` | FP16 | FP16 | K16V16 | 8.049 |
| `full_direct_absmax_w4a4_absmax_k4v4` | W4A4 | W4A4 | K4V4 | 11.116 |
| `full_mxfp4_w4a4_hlm_k4v4` | W4A4 | W4A4 | K4V4 | 10.262 |
| `full_rot_absmax_w4a4_hlm_k4v4` | W4A4 | W4A4 | K4V4 | 9.596 |
| `full_rot_mxfp4_w4a4_hlm_k4v4` | W4A4 | W4A4 | K4V4 | 10.154 |
| `full_rot_lm_w4a4_hlm_k4v4` | W4A4 | W4A4 | K4V4 | 9.230 |

结论：identity wrapper 与 FP16 完全一致，说明 full-model 替换入口本身不破坏模型计算。4-bit 全模型量化中，`full_rot_lm_w4a4_hlm_k4v4` 最优，明显优于 MXFP4 与 absmax baseline，但相对 FP16 仍有约 1.18 PPL gap。

### 10.2 Rotation backend 消融

| Method | Rotation backend | PPL |
| --- | --- | ---: |
| `full_rot_lm_w4a4_hlm_k4v4` | Hadamard | 9.230 |
| `full_randhadamard_lm_w4a4_hlm_k4v4` | randomized Hadamard | 9.371 |
| `full_randortho_lm_w4a4_hlm_k4v4` | random orthogonal | 9.798 |

结论：本轮 full-model PPL 中，普通 Hadamard 优于 randomized Hadamard 和 random orthogonal。random orthogonal 不仅 PPL 更高，也不具备 Hadamard/FWHT 的硬件友好性，因此不改变当前主线。

### 10.3 低比特与 mixed precision

| Method | FFN | Attention | KV | PPL |
| --- | ---: | ---: | ---: | ---: |
| `full_rot_lm_w3a4_hlm_k3v4` | W3A4 | W3A4 | K3V4 | 17.822 |
| `full_rot_lm_w4a3_hlm_k4v3` | W4A3 | W4A3 | K4V3 | 11.473 |
| `full_rot_lm_w3a3_hlm_k3v3` | W3A3 | W3A3 | K3V3 | 30.735 |
| `full_mixed_ffn_w3a4_attn_w4a4_k4v4` | W3A4 | W4A4 | K4V4 | 11.212 |
| `full_mixed_ffn_w4a3_attn_w4a4_k4v4` | W4A3 | W4A4 | K4V4 | 10.110 |
| `full_mixed_ffn_w4a4_attn_w3a4_k3v4` | W4A4 | W3A4 | K3V4 | 12.542 |

结论：统一低比特全模型量化退化明显，尤其 `W3A3 + K3V3` 已接近失败边界。mixed precision 明显优于统一低比特，其中只降低 FFN activation 的 `full_mixed_ffn_w4a3_attn_w4a4_k4v4` 最稳；只降低 Attention/Key 的 mixed 组明显更差，验证了 Attention/KV 对低比特更敏感。

### 10.4 Block size 与 KV H32

| Method | W/A block size | KV block size | PPL |
| --- | ---: | ---: | ---: |
| `full_rot_lm_w4a4_hlm_k4v4` | 32 | 64 | 9.278 |
| `full_rot_lm_w4a4_hlm_k4v4` | 64 | 64 | 9.230 |
| `full_rot_lm_w4a4_hlm_k4v4` | 128 | 64 | 9.274 |
| `full_rot_lm_w3a4_hlm_k3v4` | 32 | 64 | 17.212 |
| `full_rot_lm_w3a4_hlm_k3v4` | 64 | 64 | 17.822 |
| `full_rot_lm_w3a4_hlm_k3v4` | 128 | 64 | 21.260 |
| `full_rot_lm_w4a4_hlm_k4v4_h32` | 64 | 32 | 9.233 |

结论：W4A4 下 block size 32/64/128 差异很小，64 略优；W3A4/K3V4 下 block size 更敏感，128 明显变差，32 略优于 64 但仍不能解决低比特退化。KV H32 与 H64 的 PPL 基本一致，说明 head 内 H32 可作为后续局部化候选。

### 10.5 阶段结论

1. 全模型 4-bit fake quant 可行，但最稳组合仍是 `Hadamard rotation + Lloyd-Max W4A4 + HLM K4V4`，PPL 为 9.230。
2. MXFP4 是有效 baseline，但在 full-model PPL 上弱于 Rot-LM；Rot-MXFP4 相比 MXFP4 略有改善，但仍弱于 Rot-Absmax 与 Rot-LM。
3. 全模型统一 3-bit 组合退化明显，不宜作为下一阶段主线。
4. mixed precision 比统一低比特更有价值，尤其“FFN 降 activation、Attention/KV 保守”的方向最稳。
5. rotation backend 暂不需要从普通 Hadamard 改为 randomized Hadamard 或 dense random orthogonal。
6. Stage 2 后续应优先探索 layer-wise / module-wise bit allocation，而不是全层统一降低 W/A/KV 位宽。
