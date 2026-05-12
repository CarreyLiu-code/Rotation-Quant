# Stage 1 Supplement: Block Absmax / MXFP4 / Rotation Ablation

本文档记录第一阶段补充实验。补充实验修正了原 A/B 线中 `direct_absmax` baseline 使用全 tensor scale 的问题，并新增 MXFP4 与三类旋转后端消融。本文档只解释 fake quant 的数值质量和低比特可行性，不解释为真实吞吐或 kernel 加速。

## 1. 补充实验动机与命名

原实验中，A/B 线的 `direct_absmax` 使用全 tensor scale，而旋转量化组在 block 内进行量化。该设置会放大 direct baseline 的 outlier 敏感性，因此本轮将 `direct_absmax` 的语义修正为 block-wise absmax，并保留原方法名以维持实验组命名连续性。

本轮新增 `mxfp4` 作为 4-bit microscaling baseline。MXFP4 使用 E2M1 fake quant，每 32 个元素共享一个 power-of-two scale；该 group size 不随 Hadamard block size 改变。

旋转后端统一命名如下：

| Rotation backend | 实验命名 | 说明 |
| --- | --- | --- |
| `hadamard` | `hadamard_lm` / `rot_lm` | 归一化 Walsh-Hadamard rotation |
| `randomized_hadamard` | `randhadamard_lm` | 随机符号对角阵后接 Hadamard rotation |
| `random_orthogonal` | `randortho_lm` | Gaussian QR 生成的 dense random orthogonal matrix |

关键参数如下：

| 项目 | 设置 |
| --- | --- |
| A/B block size | 128 |
| MXFP4 group size | 32 |
| Rotation seed | 11 |
| PPL 数据集 | WikiText2 raw test |
| PPL 设置 | `max_samples=512`, `sequence_length=2048`, `stride=2048` |
| C 线 KV rotation | head-wise 64-d rotation，不跨 head 混合 |

## 2. Run 记录

| 实验 | Run id |
| --- | --- |
| A tensor sweep | `20260511_114041+0800_stage_a_tensor_sweep` |
| A PPL | `20260511_120214+0800_stage_a_ppl` |
| B activation | `20260511_114824+0800_stage_b_activation` |
| B local Linear/FFN | `20260511_114927+0800_stage_b_local` |
| B FFN-only PPL | `20260511_124718+0800_stage_b_ppl` |
| C KV-local | `20260511_115807+0800_stage_c_kv_local` |
| C Attention local | `20260511_115914+0800_stage_c_attention_layer` |
| C Attention-only PPL | `20260511_142335+0800_stage_c_ppl` |

所有正式产物均位于 `outputs/stage1_supplement/<run_id>/`，目录名与 `run_metadata.json` 中的 `run_id` 一致。

## 3. A 线补充结果

### 3.1 Weight tensor reconstruction

| Method | Bits | Relative MSE | Cosine | SQNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| `direct_absmax` | W4 | 0.015668 | 0.992471 | 18.107 |
| `direct_absmax` | W3 | 0.083591 | 0.961262 | 10.807 |
| `mxfp4` | W4 | 0.014125 | 0.995719 | 18.509 |
| `hadamard_absmax` | W4 | 0.013778 | 0.994431 | 18.610 |
| `hadamard_absmax` | W3 | 0.075073 | 0.965755 | 11.246 |
| `hadamard_lm` | W4 | 0.009364 | 0.996558 | 20.286 |
| `hadamard_lm` | W3 | 0.034078 | 0.984053 | 14.676 |
| `randhadamard_lm` | W4 | 0.009312 | 0.996585 | 20.310 |
| `randhadamard_lm` | W3 | 0.033905 | 0.984142 | 14.698 |
| `randortho_lm` | W4 | 0.009332 | 0.996573 | 20.300 |
| `randortho_lm` | W3 | 0.033985 | 0.984099 | 14.687 |

### 3.2 Weight-only PPL

| Method | Bits | PPL |
| --- | ---: | ---: |
| `fp16` | 16 | 8.049 |
| `direct_absmax` | W4 | 8.885 |
| `direct_absmax` | W3 | 16.491 |
| `direct_absmax` | W2 | 181702.068 |
| `mxfp4` | W4 | 8.635 |
| `hadamard_absmax` | W4 | 9.082 |
| `hadamard_absmax` | W3 | 187.587 |
| `hadamard_lm` | W4 | 8.627 |
| `hadamard_lm` | W3 | 12.993 |
| `randhadamard_lm` | W4 | 8.611 |
| `randhadamard_lm` | W3 | 12.397 |
| `randortho_lm` | W4 | 8.748 |
| `randortho_lm` | W3 | 14.305 |

A 线的补充结果表明，block-wise `direct_absmax` W4 已经是有效 baseline，不应再沿用旧实验中“Direct Absmax W4 崩溃”的判断。MXFP4 W4 与 Hadamard-LM W4 的 PPL 非常接近，是更强的主流 4-bit baseline。Lloyd-Max 在旋转域中仍显著降低 reconstruction error；W3 可运行但 PPL 明显高于 W4，W2 仍是失败边界。

旋转后端方面，`randhadamard_lm` 在 A 线 W4/W3 PPL 上略优于普通 Hadamard；`randortho_lm` 在 tensor reconstruction 上接近，但 PPL 不优于 randomized Hadamard。考虑硬件代价，dense random orthogonal 只能作为数值消融参考。

## 4. B 线补充结果

### 4.1 Activation tensor quantization

下表对 `attn_input`、`ffn_input`、`ffn_intermediate`、`q_proj_out`、`k_proj_out`、`v_proj_out` 六类 activation site 取平均。

| Method | Bits | Relative MSE | Cosine | SQNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| `direct_absmax` | A4 | 0.041203 | 0.980043 | 14.518 |
| `mxfp4` | A4 | 0.021496 | 0.989680 | 16.820 |
| `rot_absmax` | A4 | 0.012864 | 0.993832 | 19.206 |
| `rot_lm` | A4 | 0.009297 | 0.995561 | 20.390 |
| `rot_lm` | A3 | 0.034815 | 0.983110 | 14.621 |
| `rot_lm` | A2 | 0.120526 | 0.940698 | 9.243 |

`rot_lm` A3 的平均 relative MSE 低于 block-wise `direct_absmax` A4，但高于 `mxfp4` A4 与 `rot_absmax` A4。因此，在更公平 baseline 下，A3 的结论应表述为“存在降位宽空间”，而不是“全面优于所有 A4 baseline”。

### 4.2 Local Linear / FFN output

| Scope | Method | Bits | Relative MSE | Cosine | SQNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear | `direct_absmax` | W4A4 | 0.044481 | 0.978356 | 15.493 |
| Linear | `mxfp4` | W4A4 | 0.024440 | 0.988190 | 17.202 |
| Linear | `rot_absmax` | W4A4 | 0.021578 | 0.989686 | 18.904 |
| Linear | `rot_lm` | W4A4 | 0.015854 | 0.992299 | 20.222 |
| Linear | `rot_lm` | W3A4 | 0.037209 | 0.982336 | 16.416 |
| Linear | `rot_lm` | W4A3 | 0.037381 | 0.982784 | 16.351 |
| FFN | `direct_absmax` | W4A4 | 0.086249 | 0.959065 | 11.802 |
| FFN | `mxfp4` | W4A4 | 0.059908 | 0.972720 | 12.333 |
| FFN | `rot_absmax` | W4A4 | 0.044207 | 0.978834 | 14.140 |
| FFN | `rot_lm` | W4A4 | 0.031212 | 0.984782 | 15.330 |
| FFN | `rot_lm` | W3A4 | 0.073063 | 0.964408 | 11.562 |
| FFN | `rot_lm` | W4A3 | 0.079189 | 0.964333 | 11.107 |

Local 结果中，`rot_lm` W4A4 仍是最优 W4A4 组；`rot_lm` W3A4 / W4A3 的误差高于 W4A4，但仍优于或接近 block-wise `direct_absmax` W4A4 的 FFN local 误差。

### 4.3 FFN-only PPL

| Method | Bits | Rotation backend | PPL |
| --- | ---: | --- | ---: |
| `fp16` | FP16 | none | 8.049 |
| `ffn_direct_absmax` | W4A4 | none | 9.425 |
| `ffn_mxfp4` | W4A4 | none | 8.942 |
| `ffn_rot_absmax` | W4A4 | hadamard | 8.998 |
| `ffn_rot_lm` | W4A4 | hadamard | 8.587 |
| `ffn_rot_lm` | W3A4 | hadamard | 9.979 |
| `ffn_rot_lm` | W4A3 | hadamard | 9.353 |
| `ffn_randhadamard_lm` | W4A4 | randomized_hadamard | 8.617 |
| `ffn_randhadamard_lm` | W3A4 | randomized_hadamard | 9.895 |
| `ffn_randhadamard_lm` | W4A3 | randomized_hadamard | 9.311 |
| `ffn_randortho_lm` | W4A4 | random_orthogonal | 8.834 |
| `ffn_randortho_lm` | W3A4 | random_orthogonal | 10.521 |
| `ffn_randortho_lm` | W4A3 | random_orthogonal | 10.954 |

B 线补充结果支持以下修正：在 FFN-only PPL 上，MXFP4 W4A4 是强 baseline，优于 `rot_absmax` W4A4；`rot_lm` W4A4 仍明显最好。W3A4 / W4A3 仍可运行，但相对于 W4A4 的 PPL gap 增大，说明低位宽收益需要在更强 baseline 下重新评估。随机 Hadamard 与普通 Hadamard 基本持平，dense random orthogonal 没有显示出稳定优势。

## 5. C 线补充结果

### 5.1 KV-local

| Method | KV bits | Score Rel. MSE | Softmax KL | Top-k overlap | Output cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `absmax` | K4V4 | 0.015507 | 0.046387 | 0.847715 | 0.944755 |
| `hadamard_lm` | K4V4 | 0.003546 | 0.009510 | 0.918928 | 0.986786 |
| `hadamard_lm` | K3V4 | 0.013198 | 0.035287 | 0.856783 | 0.958924 |
| `randhadamard_lm` | K4V4 | 0.003731 | 0.009370 | 0.918667 | 0.987079 |
| `randhadamard_lm` | K3V4 | 0.013522 | 0.035206 | 0.857150 | 0.958748 |
| `randortho_lm` | K4V4 | 0.004014 | 0.010676 | 0.915991 | 0.985192 |
| `randortho_lm` | K3V4 | 0.014519 | 0.038788 | 0.853449 | 0.955308 |

KV-local 结果保持了原 C2 的主要判断：Hadamard-LM K4V4 明显优于 absmax K4V4；Hadamard-LM K3V4 在 score、softmax 与 output 上仍具备局部可行性。随机 Hadamard 与普通 Hadamard 非常接近；dense random orthogonal 略差。

### 5.2 Structured Attention local

| Method | Linear bits | KV bits | Score Rel. MSE | Layer Rel. MSE | Layer cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_kv_hlm_k4v4` | FP16 | K4V4 | 0.003558 | 0.024623 | 0.988034 |
| `attn_mxfp4_w4a4_hlm_k4v4` | W4A4 | K4V4 | 0.012066 | 0.120040 | 0.942516 |
| `attn_rot_lm_w4a4_hlm_k4v4` | W4A4 | K4V4 | 0.005677 | 0.089098 | 0.956899 |
| `attn_rot_lm_w3a4_hlm_k3v4` | W3A4 | K3V4 | 0.020985 | 0.255800 | 0.881672 |
| `attn_randhadamard_lm_w4a4_hlm_k4v4` | W4A4 | K4V4 | 0.005769 | 0.089520 | 0.956743 |
| `attn_randhadamard_lm_w3a4_hlm_k3v4` | W3A4 | K3V4 | 0.021447 | 0.248529 | 0.884771 |
| `attn_randortho_lm_w4a4_hlm_k4v4` | W4A4 | K4V4 | 0.006916 | 0.098507 | 0.953112 |
| `attn_randortho_lm_w3a4_hlm_k3v4` | W3A4 | K3V4 | 0.026691 | 0.284049 | 0.873701 |

在 structured Attention local 上，`rot_lm` W4A4 + HLM K4V4 明显优于 MXFP4 W4A4 + HLM K4V4；W3A4 + K3V4 的误差显著增大。随机 Hadamard 与普通 Hadamard 接近，dense random orthogonal 略差。

### 5.3 Attention-only PPL

| Method | Linear bits | KV bits | Rotation backend | PPL |
| --- | ---: | ---: | --- | ---: |
| `fp16` | FP16 | K16V16 | none | 8.049 |
| `attn_kv_hlm_k4v4` | FP16 | K4V4 | hadamard | 8.204 |
| `attn_mxfp4_w4a4_hlm_k4v4` | W4A4 | K4V4 | hadamard KV | 8.995 |
| `attn_rot_lm_w4a4_hlm_k4v4` | W4A4 | K4V4 | hadamard | 8.614 |
| `attn_rot_lm_w3a4_hlm_k3v4` | W3A4 | K3V4 | hadamard | 11.355 |
| `attn_randhadamard_lm_w4a4_hlm_k4v4` | W4A4 | K4V4 | randomized_hadamard | 8.636 |
| `attn_randortho_lm_w4a4_hlm_k4v4` | W4A4 | K4V4 | random_orthogonal | 8.608 |

C 线 model-level PPL 显示，KV-only HLM K4V4 对 PPL 的影响较小。Attention 线性 W/A fake quant 加入后，`rot_lm` W4A4 + HLM K4V4 优于 MXFP4 W4A4 + HLM K4V4。W3A4 + K3V4 可以运行但 PPL 明显升高，说明 C 线 attention 结构中 3-bit 组合比 B 线 FFN 更敏感。

需要注意，`randortho_lm` W4A4 + K4V4 的 Attention-only PPL 略低于普通 Hadamard，但其 KV-local 与 structured local 指标并不优于 Hadamard，且 dense random orthogonal 不具备 Hadamard/FWHT 的硬件友好性。因此当前不能把该单点结果解释为 random orthogonal 更优，只能作为后续进一步 seed sweep 的线索。

## 6. 对原 Stage 1 结论的影响

1. `direct_absmax` baseline 需要修正为 block-wise absmax。修正后，A 线 W4 和 B 线 FFN-only W4A4 均不再崩溃，旧报告中关于 direct absmax 过弱的表述应降低强度。

2. MXFP4 W4 是必须保留的强 baseline。A 线 weight-only PPL 中，MXFP4 W4 接近 Hadamard-LM W4；B/C 的 W/A 结构中，MXFP4 也明显强于 block-wise uniform direct absmax。

3. Hadamard + Lloyd-Max 仍然是本阶段最稳定的低误差组合。它在 A/B 的 tensor/local 指标和 B/C 的 W4A4 model-level PPL 中保持优势；但在引入更强 baseline 后，“W3 接近 W4”的结论需要更谨慎地限定在局部指标或特定模块上。

4. Randomized Hadamard 与普通 Hadamard 整体接近，没有显示出足以改变主线结论的稳定收益。Dense random orthogonal 在少数 PPL 点上可接近或略优，但 local 指标不稳定、实现代价高，现阶段不作为主线方法。

5. C 线 Attention 对低 bit 更敏感。HLM K4V4 与 Rot-LM W4A4 + HLM K4V4 的 PPL 可接受；W3A4 + K3V4 仍可运行但退化明显，不宜直接等同于 B 线 FFN 中的 W3A4 可行性。

## 7. 后续建议

1. 主报告若整合补充结论，应把 A/B 中 `direct_absmax` 的定义统一改为 block-wise absmax，并在方法表中显式标注 block size。

2. 后续正式对比应保留 MXFP4 W4，作为比 uniform absmax 更接近当前主流 FP4 路线的 baseline。

3. 如果继续研究 randomized rotation，应优先对 randomized Hadamard 做多 seed sweep；dense random orthogonal 仅作为上界或诊断组，不作为硬件友好候选。

4. C 线低 bit 组合应优先围绕 K4V4 与 W4A4 改进，再考虑 K3V4 或 W3A4；否则 PPL 退化会掩盖局部 attention score 指标的收益。

## 8. Rot-MXFP4 与 Rot-LM 对比

本轮继续补充 `Hadamard rotation + MXFP4` 组，并与相同 rotation block size 下的 `Rot-LM` 直接比较。MXFP4 的 group size 固定为 32；rotation block size 分别取 32、64、128。

新增正式 run 如下：

| Scope | Run id |
| --- | --- |
| A tensor sweep | `20260512_002529+0800_stage_a_tensor_sweep`, `20260512_002716+0800_stage_a_tensor_sweep`, `20260512_002922+0800_stage_a_tensor_sweep` |
| A PPL | `20260512_005112+0800_stage_a_ppl`, `20260512_015242+0800_stage_a_ppl`, `20260512_035332+0800_stage_a_ppl` |
| B activation | `20260512_003345+0800_stage_b_activation`, `20260512_003422+0800_stage_b_activation`, `20260512_003458+0800_stage_b_activation` |
| B local | `20260512_003619+0800_stage_b_local`, `20260512_003839+0800_stage_b_local`, `20260512_004135+0800_stage_b_local` |
| B PPL | `20260512_094606+0800_stage_b_ppl`, `20260512_100238+0800_stage_b_ppl`, `20260512_101855+0800_stage_b_ppl` |
| C KV-local | `20260512_004815+0800_stage_c_kv_local` |
| C attention local | `20260512_004858+0800_stage_c_attention_layer`, `20260512_004940+0800_stage_c_attention_layer`, `20260512_005022+0800_stage_c_attention_layer` |
| C PPL | `20260512_103612+0800_stage_c_ppl`, `20260512_104902+0800_stage_c_ppl`, `20260512_110111+0800_stage_c_ppl`, `20260512_112615+0800_stage_c_ppl` |

### 8.1 A 线结果

| Method | Bits | Rotation block | Tensor Rel. MSE | Tensor cosine | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mxfp4` | W4 | - | 0.014125 | 0.995719 | 8.635 |
| `hadamard_mxfp4` | W4 | 32 | 0.012940 | 0.994761 | 8.918 |
| `hadamard_lm` | W4 | 32 | 0.008923 | 0.996783 | 8.598 |
| `hadamard_lm` | W3 | 32 | 0.032374 | 0.984937 | 12.353 |
| `hadamard_mxfp4` | W4 | 64 | 0.013139 | 0.994661 | 9.017 |
| `hadamard_lm` | W4 | 64 | 0.009161 | 0.996662 | 8.598 |
| `hadamard_lm` | W3 | 64 | 0.033359 | 0.984423 | 12.346 |
| `hadamard_mxfp4` | W4 | 128 | 0.013244 | 0.994610 | 8.952 |
| `hadamard_lm` | W4 | 128 | 0.009364 | 0.996558 | 8.627 |
| `hadamard_lm` | W3 | 128 | 0.034078 | 0.984053 | 12.993 |

A 线中，Hadamard-LM W4 在 tensor reconstruction 和 PPL 上均优于 Hadamard-MXFP4 W4。Hadamard-MXFP4 的 tensor MSE 低于未旋转 MXFP4，但 PPL 反而更高，说明 MXFP4 的固定 FP4 level 与 Hadamard 后的分布并不必然在 model-level 上更匹配。

### 8.2 B 线结果

| Method | Bits | Rotation block | Linear Rel. MSE | FFN Rel. MSE | FFN-only PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_mxfp4` | W4A4 | - | 0.024440 | 0.059908 | 8.942 |
| `ffn_rot_mxfp4` | W4A4 | 32 | 0.021605 | 0.045747 | 8.936 |
| `ffn_rot_lm` | W4A4 | 32 | 0.014332 | 0.029818 | 8.569 |
| `ffn_rot_lm` | W3A4 | 32 | 0.034288 | 0.069378 | 9.789 |
| `ffn_rot_mxfp4` | W4A4 | 64 | 0.022084 | 0.047259 | 8.975 |
| `ffn_rot_lm` | W4A4 | 64 | 0.014696 | 0.030791 | 8.606 |
| `ffn_rot_lm` | W3A4 | 64 | 0.035734 | 0.072402 | 10.022 |
| `ffn_rot_mxfp4` | W4A4 | 128 | 0.022691 | 0.045629 | 8.894 |
| `ffn_rot_lm` | W4A4 | 128 | 0.015854 | 0.031212 | 8.587 |
| `ffn_rot_lm` | W3A4 | 128 | 0.037209 | 0.073063 | 9.979 |

B 线结论更明确：Rot-MXFP4 W4A4 优于未旋转 MXFP4 的 local error，但 model-level PPL 基本持平；Rot-LM W4A4 在 local error 和 FFN-only PPL 上均明显更好。Rot-LM W3A4 仍可运行，但在加入 MXFP4 这个强 baseline 后，不能再简单表述为接近 W4A4。

### 8.3 C 线结果

| Method | Linear bits | KV bits | Rotation block | Layer Rel. MSE | Layer cosine | Attention-only PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn_rot_mxfp4_hlm` | W4A4 | K4V4 | 32 | 0.118463 | 0.942756 | 8.849 |
| `attn_rot_lm_hlm` | W4A4 | K4V4 | 32 | 0.086423 | 0.958585 | 8.621 |
| `attn_rot_lm_hlm` | W3A4 | K3V4 | 32 | 0.246934 | 0.886020 | 10.844 |
| `attn_rot_mxfp4_hlm` | W4A4 | K4V4 | 64 | 0.118201 | 0.942817 | 8.781 |
| `attn_rot_lm_hlm` | W4A4 | K4V4 | 64 | 0.084833 | 0.958540 | 8.565 |
| `attn_rot_lm_hlm` | W3A4 | K3V4 | 64 | 0.252072 | 0.884662 | 11.005 |
| `attn_rot_mxfp4_hlm` | W4A4 | K4V4 | 128 | 0.122350 | 0.941275 | 8.933 |
| `attn_rot_lm_hlm` | W4A4 | K4V4 | 128 | 0.089098 | 0.956899 | 8.615 |
| `attn_rot_lm_hlm` | W3A4 | K3V4 | 128 | 0.255800 | 0.881672 | 11.355 |

C 线中，Rot-LM W4A4 + HLM K4V4 同样优于 Rot-MXFP4 W4A4 + HLM K4V4。W3A4 + K3V4 的 PPL 明显升高，说明 Attention 结构中的低 bit 组合比 FFN 更敏感。

### 8.4 MXFP4 与 Lloyd-Max 的差异

MXFP4 是固定 E2M1 FP4 格式，每 32 个元素共享一个 power-of-two scale。它的非均匀 level 来自浮点指数/尾数结构，优势是格式规则、动态范围覆盖好、硬件实现路径相对清晰。Lloyd-Max 则是在 RMS 标准化后的近似 Gaussian 分布上使用 centroid codebook，目标是最小化数值失真；它通常能获得更低 MSE，但不属于标准 INT/FP GEMM 友好的格式。

本轮结果显示，在 Hadamard 旋转域中，Lloyd-Max 的数值质量整体优于 MXFP4，尤其体现在 B/C 的 local output error 和 PPL 上。MXFP4 的价值主要是作为更现实的 FP4 baseline；Rot-LM 的价值则是验证非均匀 codebook 在数值上是否能进一步降低误差。二者不能只因“都是非均匀量化”而视为同类方法。

## 9. Rotation Block Size 消融

| Scope | Method | Block 32 PPL | Block 64 PPL | Block 128 PPL | Observation |
| --- | --- | ---: | ---: | ---: | --- |
| A | `hadamard_lm W4` | 8.598 | 8.598 | 8.627 | 32/64 略优 |
| A | `hadamard_lm W3` | 12.353 | 12.346 | 12.993 | 128 退化更明显 |
| B | `ffn_rot_lm W4A4` | 8.569 | 8.606 | 8.587 | 差异较小，32 最好 |
| B | `ffn_rot_lm W3A4` | 9.789 | 10.022 | 9.979 | 低 bit 下 32 更稳 |
| C | `attn_rot_lm W4A4 + K4V4` | 8.621 | 8.565 | 8.615 | 64 最好，但差异很小 |
| C | `attn_rot_lm W3A4 + K3V4` | 10.844 | 11.005 | 11.355 | block 越大退化越明显 |

Rotation block size 对 W4A4 组影响有限，但对更低 bit 的 W3A4 / W3 更敏感。A/B 中 block 32 或 64 通常优于 128；C 线 W4A4 在 block 64 上略好，但 W3A4 + K3V4 随 block size 增大而持续退化。因此，当前不宜把 128 作为唯一默认设置；后续正式实验至少应保留 32/64 的消融，尤其在评价 3-bit 可行性时。

## 10. KV H64/H32 消融

| Method | KV block | Score Rel. MSE | Softmax KL | Top-k overlap | Output cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hadamard_lm K4V4` | H64 | 0.003546 | 0.009510 | 0.918928 | 0.986786 |
| `hadamard_lm K4V4` | H32 | 0.003405 | 0.009047 | 0.921389 | 0.986800 |
| `hadamard_lm K3V4` | H64 | 0.013198 | 0.035287 | 0.856783 | 0.958924 |
| `hadamard_lm K3V4` | H32 | 0.012964 | 0.033944 | 0.860501 | 0.959283 |

H32 在 KV-local 上不弱于 H64，并满足额外 PPL 条件。补跑 `attn_kv_hlm_k4v4_h32` 后，Attention-only PPL 为 8.215；与已有 H64 KV-only 的 8.204 基本同量级。对 TinyLlama 的 `head_dim=64` 而言，head 内 H32 是可行的结构选项，并且不需要跨 head 混合。

## 11. 对当前结论的进一步修正

1. MXFP4 是强 baseline，但在本轮 Rot-MXFP4 对比中没有超过 Rot-LM。其优势更多体现在格式规范和硬件友好性，而不是当前 fake quant 数值误差。

2. Hadamard-LM W4 / W4A4 仍是最稳定的数值质量方案；但 W3 / W3A4 的可行性比原始报告需要更谨慎，尤其在 PPL 指标上不能只依赖 local MSE 得出结论。

3. Rotation block size 会影响低 bit 结果。32/64 通常比 128 更稳，后续若以 3-bit 为目标，应把 block size 作为正式超参数，而不是固定沿用 128。

4. KV cache 中 H32 与 H64 均可保持 post-RoPE attention 结构的计算适配。H32 local 指标略好，PPL 与 H64 接近，因此后续 C 线可以同时保留 H32/H64 两条候选。
