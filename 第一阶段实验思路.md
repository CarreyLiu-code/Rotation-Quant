# 第一阶段实验思路

第一阶段实验目标：

> **在小模型上系统评估 Hadamard rotation 对权重、激活、Attention/KV cache 超低比特量化的数值作用，并判断 Lloyd-Max / QJL 是否能把 W/A/KV 的可用位宽从 4 bit 推进到 3 bit，甚至局部 2 bit。**

第一阶段只验证：

1. 数值质量；
2. 低比特可行性；
3. 理论存储 bit budget；
4. 后续硬件实现的算法依据。

第一阶段**不验证真实推理加速**。尤其需要明确：

> **Lloyd-Max 本身不等于加速。**

当前 PyTorch fake quant 的实际计算通常是：

$$
x \rightarrow Q(x) \rightarrow \hat{x}_{fp}
$$

$$
W \rightarrow Q(W) \rightarrow \hat{W}_{fp}
$$

$$
y=\hat{x}_{fp}\hat{W}_{fp}
$$

这不是 native W3A4 / W4A4 GEMM。Lloyd-Max 使用非均匀 codebook，未来若要转化为效率收益，需要 LUT / codebook-aware MAC / bit packing / native low-bit kernel 支持。

---

# 0. 总体实验设定

## 0.1 模型选择

第一阶段主模型仍建议使用：

| 模型 | 用途 | 原因 |
| --- | --- | --- |
| **TinyLlama-1.1B** | 主实验模型 | LLaMA-like 架构，hidden size 通常为 2048，head dim 通常为 64，适合 Hadamard/FWHT |
| Qwen2.5-0.5B / 1.5B | 备选模型 | 小，PPL 快，可验证结构泛化性 |
| 单层 / 单模块抽取模型 | layer-level 实验 | 快速定位误差来源 |

第一阶段只用 TinyLlama，避免模型结构差异干扰结论。

---

## 0.2 数据集与输入长度

建议使用：

| 数据 | 用途 |
| --- | --- |
| WikiText-2 validation | PPL 主指标 |
| 人工 prompt / 小 batch token | sanity check |

输入长度先用：

$$
L=512
$$

跑通后再扩展：

$$
L=1024,\ 2048
$$

---

## 0.3 第一阶段解释边界

低比特实验要拆成两个层次。

| 层次 | 第一阶段是否研究 | 说明 |
| --- | ---: | --- |
| 数值质量 | 是 | MSE、cosine、SQNR、attention score、softmax KL、PPL |
| 低比特可行性 | 是 | W4/W3/W2、A4/A3/A2、K3V4 等是否不崩 |
| 理论存储压缩率 | 可以记录 | bits/code、scale、norm、metadata |
| 真实吞吐 / 访存效率 | 暂不做 | Apple M4 + PyTorch fake quant 不适合下结论 |
| native low-bit compute | 暂不做 | Lloyd-Max 不是标准 INT GEMM 友好格式 |

结果表中需要加入量化器类型和计算解释：

| Method | Quantizer type | Compute interpretation |
| --- | --- | --- |
| absmax / RTN | uniform integer-like | 未来较容易映射 INT GEMM |
| Hadamard + absmax | uniform integer-like with rotation | 未来可映射 INT GEMM，但需要处理旋转 |
| Lloyd-Max | non-uniform codebook | 当前只验证数值质量，未来需 LUT / codebook-aware MAC |
| QJL | sign sketch / estimator | 当前验证 attention score，未来需专用 kernel |

---

## 0.4 统一评价指标

### Tensor-level 指标

用于 A 线和 B1：

$$
\text{Relative MSE}=\frac{\|x-\hat{x}\|_2^2}{\|x\|_2^2}
$$

$$
\text{Cosine Similarity}=\frac{x\cdot \hat{x}}{\|x\|_2\|\hat{x}\|_2}
$$

$$
\text{SQNR}=20\log_{10}\frac{\|x\|_2}{\|x-\hat{x}\|_2}
$$

同时记录：

* max / mean ratio；
* kurtosis；
* per-channel max；
* outlier ratio，例如 $(|x|>6\sigma)$ 的比例；
* histogram / QQ plot / Gaussian fit；
* 理论 bit budget 和 metadata overhead。

### Layer-level 指标

用于 B2 和 C 线：

$$
\text{RelErr}(y)=\frac{\|y_{\text{fp16}}-y_{\text{quant}}\|_2}{\|y_{\text{fp16}}\|_2}
$$

$$
\cos(y_{\text{fp16}},y_{\text{quant}})
$$

Attention 额外记录：

$$
S=QK^T/\sqrt{d_h}
$$

* inner product bias；
* inner product variance；
* attention score MSE；
* softmax KL divergence；
* attention top-k overlap；
* attention output cosine similarity。

### Model-level 指标

第一阶段最终只看：

$$
\text{PPL}
$$

---

## 0.5 统一量化器设计

### Quantizer 1：symmetric absmax / RTN

$$
s=\frac{\max |x|}{2^{b-1}-1}
$$

$$
q=\text{clip}\left(\text{round}\left(\frac{x}{s}\right), -2^{b-1}, 2^{b-1}-1\right)
$$

$$
\hat{x}=sq
$$

这是最重要的 baseline，也是未来最容易映射到 INT GEMM 的格式。

### Quantizer 2：Gaussian Lloyd-Max fake quant

对旋转并标准化后的变量：

$$
z=\frac{x}{s}
$$

使用预先计算的 Gaussian Lloyd-Max centroids：

$$
q=\arg\min_i |z-c_i|
$$

$$
\hat{x}=s c_q
$$

Lloyd-Max 的第一阶段意义是：判断它是否能在 Hadamard rotation 后把 W/A/KV 的可用位宽从 4 bit 推到 3 bit，甚至局部 2 bit。

### Quantizer 3：Gaussian-QJL estimator

只用于 C4。QJL 用 Gaussian JL projection 加 sign-bit quantization key / residual，并用非对称内积估计器恢复 $q\cdot k$。

QJL 关键点：

* key 或 residual 存 sign code；
* query 在线做同一个 JL transform，但不量化；
* 目标是无偏估计 inner product；
* 第一阶段验证 attention score，不讨论真实 kernel 加速。


---

# A 线：Weight-only Hadamard Rotation Quantization

## A0. 目标与核心问题

A 线目标：

> **验证 Hadamard rotation + Lloyd-Max 是否能让 weight-only quantization 从 W4 推进到 W3/W2，同时保持可接受的 layer output error 和 PPL。**

核心判断：

> **Hadamard-LM W3 是否接近或优于 Hadamard-Absmax W4？**

---

## A1. 实验对象

对 transformer block 内 Linear 层做 weight-only quantization：

* attention.q_proj；
* attention.k_proj；
* attention.v_proj；
* attention.o_proj；
* mlp.gate_proj；
* mlp.up_proj；
* mlp.down_proj。

第一阶段：

> embedding、norm、lm_head 保持 FP16。

---

## A2. 实验组与优先级

| 优先级 | 方法 | 是否保留 | 说明 |
| --- | --- | --- | --- |
| 高 | Direct absmax / RTN | 必须 | 主 baseline |
| 高 | Hadamard + absmax | 必须 | 验证 rotation 本身 |
| 高 | Hadamard + Lloyd-Max | 必须 | 核心方法 |
| 中 | Direct Lloyd-Max | 可选 | 诊断组，不作为主结论 |
| 低 | asymmetric min-max | 可选 | 实验量大时先不做 |

核心实验组：

| 组别 | 权重处理 | 量化器 | bits | 意义 |
| --- | --- | --- | --- | --- |
| A0 | FP16 | 无 | 16 | baseline |
| A1 | direct weight | symmetric absmax | 4 / 3 / 2 | 普通均匀量化边界 |
| A2 | Hadamard-rotated weight | symmetric absmax | 4 / 3 / 2 | rotation 是否改善均匀量化 |
| A3 | Hadamard-rotated weight | Lloyd-Max | 4 / 3 / 2 | rotation + 非均匀 codebook 是否进一步降低位宽 |

---

## A3. Hadamard Weight Rotation 方式

参考 PolarQuant，采用 block-wise rotation。

对每个 weight tensor 展平成一维：

$$
W\rightarrow \{b_i\}_{i=1}^N
$$

block size 第一阶段建议：

$$
d=128
$$

对每个 block：

$$
r_i=\|b_i\|_2
$$

$$
\bar{b}_i=\frac{b_i}{r_i}
$$

$$
z_i=H_{128}\bar{b}_i
$$

如果用 Lloyd-Max：

$$
\tilde{z}_i=Q_{\text{LM}}(\sqrt{128}\cdot z_i)
$$

dequant：

$$
\hat{b}_i=r_iH_{128}^T\frac{\tilde{z}_i}{\sqrt{128}}
$$

因为 Hadamard 矩阵正交且自逆：

$$
H^T=H,\quad H^{-1}=H
$$

实现上可以直接用同一个 Hadamard transform 做 inverse。

---

## A4. 输出指标

### A4.1 Tensor-level

每层记录：

* weight relative MSE；
* weight cosine similarity；
* SQNR；
* max / mean ratio；
* kurtosis；
* rotated weight 是否更接近 Gaussian；
* per-layer quantization error heatmap。

### A4.2 Layer-level

固定输入 activation，比较：

$$
y_{\text{fp16}}=xW
$$

$$
y_{\text{quant}}=x\hat{W}
$$

记录：

* output relative error；
* output cosine similarity；
* SQNR。

### A4.3 Model-level

把 $\hat{W}$ 替换回模型，activation 保持 FP16，评估：

$$
\text{A16W}b
$$

重点：

* A16W4；
* A16W3；
* A16W2。

结果表加入计算解释：

| Method | Bits | PPL | Weight MSE | Layer output err | Quantizer type | INT-GEMM friendly? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Direct absmax | 4 | | | | uniform integer-like | Yes |
| Hadamard absmax | 4 | | | | uniform integer-like | Yes-ish，需处理旋转 |
| Hadamard LM | 3 | | | | non-uniform codebook | No，需 LUT/dequant |

---

## A5. 预期结论

预期：

1. W4 下 Hadamard + absmax 应该优于 direct absmax；
2. W3 / W2 下 direct quantization 可能严重退化；
3. Hadamard + Lloyd-Max 可能在 W3 / W2 下体现优势；
4. direct Lloyd-Max 不一定优于 absmax，因为没有 rotation 时分布未必接近 Gaussian；
5. 如果 Hadamard-LM W3 接近 Hadamard-Absmax W4，说明 Lloyd-Max 有潜力以非均匀 codebook 换取更低位宽；
6. 这个结论不能直接写成推理更快，只能写成未来硬件实现的算法依据。

---

# B 线：QuaRot-style W/A Fake Quantization with Lloyd-Max Codebook

## B0. 范围与基本公式

核心公式为：

$$
y=xW
$$

引入 Hadamard rotation：

$$
y=(xH)(H^TW)
$$

然后量化：

$$
\hat{y}=Q_A(xH)\cdot Q_W(H^TW)
$$

---

## B1. Activation Outlier 与旋转量化误差

### B1.1 目标

验证：

1. activation 中确实存在 outlier；
2. Hadamard rotation 可以降低 activation 动态范围；
3. rotation 后 activation 是否更适合 A4/A3/A2 低比特量化；
4. Rot-LM A3 是否接近 Rot-Absmax A4。

### B1.2 实验对象

采集：

| Tensor | 位置 |
| --- | --- |
| residual hidden state | 每个 transformer block 输入 |
| post-RMSNorm hidden state | attention / FFN 输入 |
| FFN intermediate | gate/up 后乘积 |
| Q/K/V activation | attention projection 输出 |
| attention output | o_proj 输入 |

第一阶段重点：

* post-RMSNorm hidden state；
* FFN intermediate；
* Q/K/V。

### B1.3 实验组

| Method | Bits | 说明 |
| --- | --- | --- |
| Direct-Absmax | A4 / A3 / A2 | 普通 activation quant baseline |
| Rot-Absmax | A4 / A3 / A2 | rotation 是否改善均匀量化 |
| Rot-LM | A4 / A3 / A2 | rotation + non-uniform codebook |

Direct-LM 可以作为诊断组，但不是主线。

### B1.4 指标

* activation relative MSE；
* cosine similarity；
* SQNR；
* max / mean ratio；
* kurtosis；
* outlier ratio；
* histogram before/after rotation。

B1 不进入模型推理，只做 tensor-level 机制验证。

---

## B2. Local Linear / FFN 级 Fake Quant

### B2.1 定位

`RotLMFakeQuantLinear` 只作为 B2 工具。

它验证：

> 单个 Linear 或 FFN 的局部旋转量化误差。

它不验证：

> 完整 Transformer 的结构级 computational invariance。

---

### B2.2 单 Linear 层实验

对任意 Linear：

$$
y=xW
$$

构造：

| Method | 计算 | 说明 |
| --- | --- | --- |
| Direct-Absmax | $Q_A(x)Q_W(W)$ | 普通均匀 fake quant |
| Rot-Absmax | $Q_A(xH)Q_W(H^TW)$ | QuaRot-style local rotation |
| Rot-LM | $Q_{\text{LM}}(xH)Q_{\text{LM}}(H^TW)$ | 非均匀 codebook 数值验证 |

重点 bit 组合：

| Method | Bits | 解释 |
| --- | --- | --- |
| Direct-Absmax | W4A4 | 主 baseline |
| Rot-Absmax | W4A4 | rotation + uniform |
| Rot-LM | W4A4 | 同 bit 对比 |
| Rot-LM | W3A4 | weight 降位宽 |
| Rot-LM | W4A3 | activation 降位宽 |
| Rot-LM | W3A3 | 极限组合 |
| Rot-LM | W2A4 | 失败边界 |

解释规则：

* W4A4 absmax：未来较容易映射 INT4 GEMM；
* W3A4 Lloyd-Max：当前只是数值可行性，未来需要 LUT/decode 支持；
* W3A3 Lloyd-Max：探索位宽极限，不假设可加速。

核心判断：

> Rot-LM W3A4 是否接近或优于 Rot-Absmax W4A4？

---

### B2.3 FFN 模块实验

在 LLaMA-like 模型中，常被代码命名为 `mlp` 的 gated feed-forward block 在本文档中统一称为 **FFN**，与 QuaRot 的术语保持一致。

LLaMA-like FFN：

$$
u=\text{SiLU}(xW_{\text{gate}})\odot(xW_{\text{up}})
$$

$$
y=uW_{\text{down}}
$$

对 gate/up 输入侧：

$$
xW_{\text{gate}}=(xH_h)(H_h^TW_{\text{gate}})
$$

$$
xW_{\text{up}}=(xH_h)(H_h^TW_{\text{up}})
$$

对中间变量 $u$，如果 intermediate size 不是 $2^n$，第一阶段使用 block-wise Hadamard，而不是强行构造完整 Hadamard。

例如 TinyLlama 的 intermediate size 若为 5632，可以按 128 或 512 做 block Hadamard。

实验组：

| Method | Bits | 说明 |
| --- | --- | --- |
| FFN-FP16 | FP16 | baseline |
| FFN-Direct-Absmax | W4A4 | 普通联合 fake quant |
| FFN-Rot-Absmax | W4A4 | rotation + uniform |
| FFN-Rot-LM | W4A4 | 同 bit 对比 |
| FFN-Rot-LM | W3A4 | weight 降位宽 |
| FFN-Rot-LM | W4A3 | activation 降位宽 |
| FFN-Rot-LM | W3A3 | 极限组 |

核心判断：

> 非线性和 gate/up/down 结构是否放大 Rot-LM 的量化误差？

---

## B3. Model-level FFN-only Fake Quant

第一阶段 model-level 范围为：

> **只 fake quant FFN，不修改 Attention 的计算。**

原因：

* FFN 占参数和计算大头；
* 结构比 Attention 简单；
* 避免 RoPE、KV cache、GQA 干扰；
* 更容易观察 W/A 量化误差对 PPL 的影响。

实验：

| Method | PPL | Compute interpretation |
| --- | ---: | --- |
| FP16 | | baseline |
| FFN Direct-Absmax W4A4 | | fake quant，未来较易 INT4 |
| FFN Rot-Absmax W4A4 | | fake quant，需旋转处理 |
| FFN Rot-LM W4A4 | | non-uniform codebook 数值验证 |
| FFN Rot-LM W3A4 | | 降 weight bit 可行性 |
| FFN Rot-LM W4A3 | | 降 activation bit 可行性 |

Attention 的结构性适配复杂，与 C 线合并，不在 B 线第一阶段实验。

---

## B4. 结果解释规则

最重要判断：

> 如果 Rot-LM W3A4 的 PPL 接近或优于 Rot-Absmax W4A4，则说明 Lloyd-Max 的价值在于“用非均匀 codebook 换取更低位宽”。

PPL预期排序可以写成：

$$
\text{FP16}<\text{Rot-LM W4A4}\approx\text{Rot-Absmax W4A4}<\text{Direct W4A4}
$$

但更有研究价值的是：

$$
\text{Rot-LM W3A4}\stackrel{?}{\approx}\text{Rot-Absmax W4A4}
$$

以及：

$$
\text{Rot-LM W4A3}\stackrel{?}{\approx}\text{Rot-Absmax W4A4}
$$

---

# C 线：Attention/KV Cache 的 Inner Product Preservation

## C0. 目标与范围

C 线目标为：

> **验证 Hadamard rotation + Lloyd-Max 是否能在更低 bit-width 下保持 KV cache 的 attention score 和 attention output；进一步验证 QJL residual 是否能修正 Key quantization 的 inner product 误差。**


因为第一阶段没有真实 KV cache bit packing，也没有 low-bit attention kernel。

---

## C1. Key / Value 评价目标

### C1.1 Key：核心是 Inner Product

Key 参与：

$$
S=QK^T/\sqrt{d_h}
$$

所以 Key 的核心不是 reconstruction MSE，而是：

$$
q_{\text{rope}}^Tk_{\text{rope}}\approx q_{\text{rope}}^T\hat{k}_{\text{rope}}
$$

Key 指标优先级：

1. inner product bias；
2. inner product variance；
3. attention score MSE；
4. softmax KL；
5. top-k attention overlap；
6. K reconstruction MSE。

### C1.2 Value：核心更接近 Reconstruction

Value 参与：

$$
O=PV
$$

Value 指标优先级：

1. V reconstruction error；
2. attention output error；
3. output cosine similarity。

---

## C2. RoPE 位置与 Attention 计算不变性

LLaMA attention 中 key/query 会经过 RoPE。QuaRot 的 key cache 量化采用的是 **Post-RoPE caching** 路径：

$$
Q_{\text{rope}}=\text{RoPE}(XW_q)
$$

$$
K_{\text{rope}}=\text{RoPE}(XW_k)
$$

然后对每个 head 的 query 和 key 使用同一个 head-wise Hadamard rotation：

$$
Q_H=Q_{\text{rope}}H_{d_h}
$$

$$
K_H=K_{\text{rope}}H_{d_h}
$$

由于 $H_{d_h}$ 正交：

$$
Q_{\text{rope}}K_{\text{rope}}^T=(Q_{\text{rope}}H_{d_h})(K_{\text{rope}}H_{d_h})^T
$$

因此 attention score 保持不变。随后 QuaRot 把旋转后的 key 存入 KV cache，并在 cache 中量化。decode 时只需要把当前 query 在 RoPE 后在线旋转一次，再与 dequant 后的 rotated key cache 做 dot product。

这意味着第一阶段 C 线应采用：

> **post-RoPE key/query + head-wise Hadamard rotation + rotated key cache quantization**

所以 C 线确实需要适配 Attention 的计算不变性，但第一阶段只做 attention-local 适配：

1. 抽取真实 attention 中 RoPE 后的 $Q_{\text{rope}},K_{\text{rope}}$；
2. 对二者施加同一个 head-wise Hadamard / randomized Hadamard；
3. 量化并缓存 rotated key；
4. 用 rotated query 与 dequant rotated key 计算 score；
5. 比较 score MSE、softmax KL 和 attention output。

Value 侧分两层处理：

1. 在 C3 attention-local KV 实验中，可以用 reconstruction 路径：对 $V$ 做旋转量化，再反旋回原空间计算 $O=P\hat{V}$，便于直接评估 KV cache 量化误差；
2. 在 C5 structured Attention / model-level 实验中，默认采用 **value rotation + `o_proj` absorb** 路径：$V$ 保留在 head-wise rotated domain，先算 $O_H=P(VH)$，再用 $W_oH_{\text{blockdiag}}$ 吸收 value 旋转。

因为 $P$ 只在 sequence 维度做 token mixing，不作用在 feature/head 维度，所以：

$$
P(VH)=(PV)H
$$

因此 value rotation 可以合法地推到 `o_proj` 输入侧，并由 `o_proj.weight` 的 input dimension 吸收。第一阶段 C5 的正式路径以该结构为准。

---

## C3. Attention-local KV：Hadamard-LM for K/V

### C3.1 目标

验证：

> Hadamard rotation + Lloyd-Max 是否让 K3V4、K4V3、K3V3 更可用。

### C3.2 Key 处理

对每个 head 的 **post-RoPE key vector**：

$$
k_{\text{rope}}\in\mathbb{R}^{d_h}
$$

例如：

$$
d_h=64
$$

做：

$$
r_k=\|k_{\text{rope}}\|_2
$$

$$
\bar{k}=\frac{k_{\text{rope}}}{r_k}
$$

$$
k_H=HD\bar{k}
$$

量化：

$$
\hat{k}_H=Q_{\text{LM}}(k_H)
$$

计算 attention score 时，对 **post-RoPE query** 做同一个在线旋转：

$$
q_H=HDq_{\text{rope}}
$$

$$
q_{\text{rope}}^Tk_{\text{rope}}=(HDq_{\text{rope}})^T(HDk_{\text{rope}})
$$

因此：

$$
\widehat{q_{\text{rope}}^Tk_{\text{rope}}}=q_H^T\hat{k}_H\cdot r_k
$$

query 不存储，只在 RoPE 后在线旋转。

### C3.3 Value 处理

Value 更偏向 reconstruction：

$$
v\rightarrow HDv\rightarrow Q(v_H)
$$

dequant：

$$
\hat{v}=D^TH^T\hat{v}_H
$$

再计算：

$$
\text{AttnOut}=\text{softmax}(\hat{S})\hat{V}
$$

注意：这是 C3 attention-local KV 的诊断路径。进入 C5 structured Attention 后，正式 model-level 路径采用 value rotation + `o_proj` absorb，而不是把 value 反旋回原域后再进入原始 `o_proj`。

---

### C3.4 实验组

不做完整 K/V bit-width 网格，第一阶段只做关键组合：

| Method | K/V bits | 目的 |
| --- | --- | --- |
| FP16 | K16V16 | baseline |
| Absmax | K4V4 | 普通 KV baseline |
| Absmax | K3V4 | Key 降位宽 |
| Absmax | K4V3 | Value 降位宽 |
| Hadamard-LM | K4V4 | 验证 rotation + LM |
| Hadamard-LM | K3V4 | Key 3 bit 是否可行 |
| Hadamard-LM | K4V3 | Value 3 bit 是否可行 |
| Hadamard-LM | K3V3 | 极限组合 |
| Hadamard-LM | K2V4 | 失败边界 |

输出：

* Key inner product bias / variance；
* attention score MSE；
* softmax KL；
* attention top-k overlap；
* Value reconstruction error；
* attention output cosine。

---

## C4. Attention-local Key：Hadamard-LM + Gaussian-QJL Residual

### C4.1 研究目标

C3 的 Hadamard-LM 优化 base reconstruction，但 attention 的关键是 inner product：

$$
q_{\text{rope}}^Tk_{\text{rope}}
$$

TurboQuant 的观点是：MSE-optimized quantizer 可能导致 inner product bias，因此需要在 residual 上加 QJL 修正。

C4 目标：

> 验证 QJL residual 是否降低 inner product bias 和 softmax KL。

---

### C4.2 方法

先在 rotated post-RoPE key 空间中用 Hadamard-LM 得到 base reconstruction：

$$
\hat{k}_{H,\text{LM}}=Q_{\text{LM}}(HDk_{\text{rope}})
$$

恢复到 post-RoPE 原空间：

$$
\hat{k}_{\text{base}}=D^TH^T\hat{k}_{H,\text{LM}}
$$

计算 residual：

$$
r=k_{\text{rope}}-\hat{k}_{\text{base}}
$$

对 residual 做 Gaussian-QJL：

$$
\text{QJL}(r)=\text{sign}(Sr)
$$

其中：

$$
S_{ij}\sim\mathcal{N}(0,1)
$$

最终估计：

$$
q_{\text{rope}}^Tk_{\text{rope}}\approx q_{\text{rope}}^T\hat{k}_{\text{base}}+\widehat{q_{\text{rope}}^Tr}_{\text{QJL}}
$$

QJL residual estimator：

$$
\widehat{q_{\text{rope}}^Tr}_{\text{QJL}}=\frac{\sqrt{\pi/2}}{m}\|r\|_2\langle Sq_{\text{rope}},\text{sign}(Sr)\rangle
$$

第一阶段建议：

$$
m=d_h
$$

---

### C4.3 Bit Budget 计算方法

C4 的 bit budget 不是简单 K2 或 K3，而是：

$$
\text{base LM bits}+\text{QJL residual bits}+\text{norm metadata}
$$

结果表这样写：

| Method | Base K bits | Residual bits | Effective description |
| --- | ---: | ---: | --- |
| Hadamard-LM K3 | 3 | 0 | pure reconstruction |
| Hadamard-LM K2 | 2 | 0 | aggressive base |
| Hadamard-LM K2 + QJL residual | 2 | 1 | inner-product corrected |
| Hadamard-LM K3 + QJL residual | 3 | 1 | stronger corrected baseline |

注意：

> QJL residual 的 1 bit 是 residual sketch 的 bit，不一定和原始维度一一对应。

如果 $m=d_h$，可以粗略看作 +1 bit/channel；如果 $m<d_h$，则是更强压缩。

---

### C4.4 实验组

Value 固定：

* V4 absmax；
* 或 V4 Hadamard-LM。

第一阶段建议先固定 V4 absmax，把变量控制住。

| Method | Description |
| --- | --- |
| Hadamard-LM K3 | pure base |
| Hadamard-LM K2 | aggressive base |
| Hadamard-LM K2 + Gaussian-QJL residual | corrected |
| Hadamard-LM K3 + Gaussian-QJL residual | corrected stronger |

---

### C4.5 成功标准

C4 第一阶段不必直接测全模型 PPL。优先测：

$$
q_{\text{rope}}^Tk_{\text{rope}}
$$

的 bias 和 variance。

成功标准：

> 在相同或相近 bit budget 下，Hadamard-LM + QJL residual 的 attention score MSE / softmax KL 明显低于纯 Hadamard-LM。

---

## C5. Attention-layer Structured Quant 全过程适配

C3/C4 只验证 KV cache 或 key inner product。C5 进一步把 **Attention 层内部的线性计算和 KV cache 量化串起来**，作为 C 线进入 model-level PPL 前的整层实验。

目标：

> **在 Attention 层中，对 q/k/v/o 线性计算采用类似 FFN 的旋转域 W/A fake quant，对 KV cache 采用 C3/C4 讨论的 post-RoPE rotated low-bit quantization，评估整层输出误差、准确度和 PPL。**

### C5.1 计算结构

以 LLaMA attention 为例：

$$
Q=XW_q,\quad K=XW_k,\quad V=XW_v
$$

$$
Q_{\text{rope}}=\text{RoPE}(Q),\quad K_{\text{rope}}=\text{RoPE}(K)
$$

$$
P=\text{softmax}(Q_{\text{rope}}K_{\text{rope}}^T/\sqrt{d_h})
$$

$$
O=(PV)W_o
$$

C5 中分两部分做计算不变性适配。

第一部分是 q/k/v/o 线性层的 W/A fake quant，沿用 B 线的思路：

$$
xW\approx Q_A(xH)\,Q_W(H^TW)
$$

用于：

* q_proj；
* k_proj；
* v_proj；
* o_proj。

第二部分是 KV cache 的 post-RoPE head-wise rotation：

$$
Q_H=Q_{\text{rope}}H_{d_h}
$$

$$
K_H=K_{\text{rope}}H_{d_h}
$$

并对 rotated key cache 量化：

$$
\hat{K}_H=Q_{\text{KV}}(K_H)
$$

score 计算使用：

$$
\hat{S}=Q_H\hat{K}_H^T/\sqrt{d_h}
$$

Value cache 在 C5 中采用 value-absorb 正式路径：不把 dequant 后的 value 反旋回原域，而是保留在 rotated value domain：

$$
V_H=VH_{d_h}
$$

$$
\hat{V}_H=Q_{\text{KV}}(V_H)
$$

再计算：

$$
\hat{O}_H=\text{softmax}(\hat{S})\hat{V}_H
$$

由于 $P(VH)=(PV)H$，value rotation 可以被吸收到 `o_proj` 的 input dimension：

$$
W_{o,\text{abs}}=W_oH_{\text{blockdiag}}
$$

最后使用吸收后的 `o_proj`：

$$
\hat{Y}=Q_A(\hat{O}_H)\,Q_W(W_{o,\text{abs}})^T
$$

注意：

> C5 仍然是 fake quant 数值实验，不是 native low-bit attention kernel。`o_proj` absorb 是结构计算不变性适配，不代表已经有真实 low-bit attention kernel。

---

### C5.2 实验层级

C5 分为两个层级。

#### C5-layer：单层 / 少数层 Attention 输出误差

固定真实 hidden input，比较：

$$
Y_{\text{fp16}}=\text{Attention}_{\text{fp16}}(X)
$$

$$
Y_{\text{quant}}=\text{Attention}_{\text{structured-quant}}(X)
$$

输出：

* q/k/v/o projection output error；
* attention score MSE；
* softmax KL；
* attention output cosine；
* final attention layer output relative error；
* per-layer sensitivity。

#### C5-model：Attention-only Model-level PPL / Accuracy

在模型中只替换 Attention，不替换 FFN：

| Method | Attention Linear | KV cache | Value / o_proj path | 说明 |
| --- | --- | --- | --- | --- |
| Attn-FP16 | FP16 | FP16 | reference | baseline |
| Attn-Identity-FP16 | FP16 wrapper | FP16 | wrapper identity | 验证 wrapper 与 HF TinyLlama forward 一致 |
| Attn-KV-HLM K4V4 | FP16 | Hadamard-LM K4V4 | o_proj absorb | 只验证 KV cache low-bit |
| Attn-KV-HLM K3V4 | FP16 | Hadamard-LM K3V4 | o_proj absorb | key 降位宽 |
| Attn-KV-HLM K4V3 | FP16 | Hadamard-LM K4V3 | o_proj absorb | value 降位宽 |
| Attn-Rot-LM W4A4 + HLM K4V4 | rotation + LM W/A | Hadamard-LM K4V4 | o_proj absorb | 同 bit structured baseline |
| Attn-Rot-LM W3A4 + HLM K3V4 | rotation + LM W/A | Hadamard-LM K3V4 | o_proj absorb | weight/key 降位宽 |
| Attn-Rot-LM W4A3 + HLM K4V3 | rotation + LM W/A | Hadamard-LM K4V3 | o_proj absorb | activation/value 降位宽 |
| Attn-Rot-LM W3A4 + HLM K2+QJL,V4 | rotation + LM W/A | QJL corrected key | o_proj absorb | 仅当 C4 成功后再做 |

输出：

* WikiText-2 PPL；
* 小规模 zero-shot accuracy；
* attention-only 替换与 FFN-only 替换的 PPL 对比。

### C5.3 成功标准

C5 的成功标准是：

> Attention structured quant 的 PPL 不明显劣于 FFN-only quant，并且 `Attn Rot-LM W3A4 + Hadamard-LM K3V4` 接近 `Attn Rot-Absmax W4A4 + Hadamard-LM K4V4`。

如果 C4 中 QJL residual 已经显著降低 score error，则额外判断：

> `K2+QJL,V4` 是否能接近或优于 pure `K3V4`。

---

# D. 执行阶段顺序

## Stage 0：基础组件

实现：

1. normalized Hadamard / FWHT；
2. random sign diagonal $D$；
3. symmetric absmax fake quant；
4. Lloyd-Max fake quant；
5. Gaussian-QJL estimator；
6. activation hook；
7. layer output comparison；
8. PPL evaluation script；
9. attention structured fake quant wrapper；
10. post-RoPE rotated KV cache simulator。

Sanity check：

$$
H^TH=I
$$

$$
x\approx H^THx
$$

$$
xW\approx (xH)(H^TW)
$$

---

## Stage 1：A 线 Weight-only 数值验证

只做：

| Method | Bits |
| --- | --- |
| Direct-Absmax | W4 / W3 / W2 |
| Hadamard-Absmax | W4 / W3 / W2 |
| Hadamard-LM | W4 / W3 / W2 |

输出：

* weight MSE；
* layer output error；
* A16Wb PPL；
* Quantizer type；

核心判断：

> Hadamard-LM W3 是否接近或优于 Hadamard-Absmax W4？

---

## Stage 2：B1 Activation 旋转量化

对 activation 做：

| Method | Bits |
| --- | --- |
| Direct-Absmax | A4 / A3 / A2 |
| Rot-Absmax | A4 / A3 / A2 |
| Rot-LM | A4 / A3 / A2 |

输出：

* activation MSE；
* cosine；
* SQNR；
* outlier ratio；
* histogram；
* Gaussian fit。

核心判断：

> Rot-LM A3 是否接近 Rot-Absmax A4？

---

## Stage 3：B2 Local Linear / FFN Fake Quant

对 Linear 和完整 FFN 做：

| Method | Bits |
| --- | --- |
| Direct-Absmax | W4A4 |
| Rot-Absmax | W4A4 |
| Rot-LM | W4A4 |
| Rot-LM | W3A4 |
| Rot-LM | W4A3 |
| Rot-LM | W3A3 |

输出：

* Linear output error；
* FFN output error；
* cosine；
* SQNR；
* per-layer sensitivity。

核心判断：

> 非线性和 gate/up/down 结构是否放大 Rot-LM 的量化误差？

---

## Stage 4：B3 Model-level FFN-only Fake Quant

只替换 FFN，不动 Attention。

| Method | PPL |
| --- | ---: |
| FP16 | |
| FFN Direct-Absmax W4A4 | |
| FFN Rot-Absmax W4A4 | |
| FFN Rot-LM W4A4 | |
| FFN Rot-LM W3A4 | |
| FFN Rot-LM W4A3 | |

核心判断：

> Rot-LM 在 model-level 是否仍能保持优势？

---

## Stage 5：C3 Attention-local KV

只做单层或少数层 attention-local KV 实验，先不替换完整 Attention 层的 q/k/v/o 线性计算。

| Method | K/V bits |
| --- | --- |
| Absmax | K4V4 |
| Absmax | K3V4 |
| Absmax | K4V3 |
| Hadamard-LM | K4V4 |
| Hadamard-LM | K3V4 |
| Hadamard-LM | K4V3 |
| Hadamard-LM | K3V3 |
| Hadamard-LM | K2V4 |

输出：

* Key inner product error；
* attention score MSE；
* softmax KL；
* top-k overlap；
* attention output cosine。

核心判断：

> Hadamard-LM 是否让 K3V4 或 K3V3 更可用？

---

## Stage 6：C4 QJL Residual

只对 Key 做。

| Method | Description |
| --- | --- |
| Hadamard-LM K3 | pure base |
| Hadamard-LM K2 | aggressive base |
| Hadamard-LM K2 + Gaussian-QJL residual | corrected |
| Hadamard-LM K3 + Gaussian-QJL residual | corrected stronger |

Value 固定 V4 absmax 或 V4 Hadamard-LM。

核心判断：

> QJL residual 是否降低 inner product bias 和 softmax KL？

---

## Stage 7：C5 Attention-layer Structured Quant

把 q/k/v/o 线性 fake quant 与 post-RoPE rotated KV cache quant 串起来。C5 默认使用 value rotation + `o_proj` absorb。

先做 C5-layer：

| Method | Attention Linear | KV cache | Value path |
| --- | --- | --- | --- |
| Attn-FP16 | FP16 | FP16 | reference |
| Attn-Identity-FP16 | FP16 wrapper | FP16 | identity |
| Attn-KV-HLM K4V4 | FP16 | Hadamard-LM K4V4 | o_proj absorb |
| Attn-KV-HLM K3V4 | FP16 | Hadamard-LM K3V4 | o_proj absorb |
| Attn-Rot-LM W4A4 + HLM K4V4 | Rot-LM W4A4 | Hadamard-LM K4V4 | o_proj absorb |
| Attn-Rot-LM W3A4 + HLM K3V4 | Rot-LM W3A4 | Hadamard-LM K3V4 | o_proj absorb |
| Attn-Rot-LM W4A3 + HLM K4V3 | Rot-LM W4A3 | Hadamard-LM K4V3 | o_proj absorb |
| Attn-Rot-LM W3A4 + HLM K2+QJL,V4 | Rot-LM W3A4 | Hadamard-LM K2+QJL,V4 | o_proj absorb |

输出：

* projection output error；
* attention score MSE；
* softmax KL；
* final attention output error；
* per-layer sensitivity。

再做 C5-model：

| Method | PPL | Accuracy |
| --- | ---: | ---: |
| Attn-FP16 | | |
| Attn-Identity-FP16 | | |
| Attn-KV-HLM K4V4 | | |
| Attn-KV-HLM K3V4 | | |
| Attn-Rot-LM W4A4 + HLM K4V4 | | |
| Attn-Rot-LM W3A4 + HLM K3V4 | | |
| Attn-Rot-LM W4A3 + HLM K4V3 | | |
| Attn-Rot-LM W3A4 + HLM K2+QJL,V4 | | |

核心判断：

> Attention structured quant 在 PPL 上是否接近 FFN-only quant；value rotation + `o_proj` absorb 是否让 C5 model-level PPL 与 C3/C5-layer local 指标保持一致；K2+QJL 是否能接近或优于 pure K3V4。

---

# E. 最终产出表

## 表 1：A 线 Weight-only PPL 与位宽边界

| Method | W4 PPL | W3 PPL | W2 PPL | Weight MSE | Quantizer type | INT-GEMM friendly? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Direct absmax | | | | | uniform integer-like | Yes |
| Hadamard absmax | | | | | uniform integer-like | Yes-ish |
| Hadamard LM | | | | | non-uniform codebook | No |

核心比较：

$$
\text{Hadamard-LM W3}\stackrel{?}{\approx}\text{Hadamard-Absmax W4}
$$

---

## 表 2：B1 Activation Quantization Error

| Tensor | Bits | Direct absmax | Rot absmax | Rot LM |
| --- | ---: | ---: | ---: | ---: |
| post-RMSNorm hidden | 4 | | | |
| post-RMSNorm hidden | 3 | | | |
| FFN intermediate | 4 | | | |
| FFN intermediate | 3 | | | |
| K | 4 | | | |
| K | 3 | | | |
| V | 4 | | | |
| V | 3 | | | |

核心比较：

$$
\text{Rot-LM A3}\stackrel{?}{\approx}\text{Rot-Absmax A4}
$$

---

## 表 3：B2 Local W/A Joint Fake Quant

| Method | W4A4 | W3A4 | W4A3 | W3A3 | Compute interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct absmax | | | | | uniform fake quant |
| Rot absmax | | | | | rotation + uniform fake quant |
| Rot LM | | | | | non-uniform codebook fake quant |

核心比较：

$$
\text{Rot-LM W3A4}\stackrel{?}{\approx}\text{Rot-Absmax W4A4}
$$

---

## 表 4：Model-level FFN-only PPL

| Method | PPL | Interpretation |
| --- | ---: | --- |
| FP16 | | baseline |
| FFN Direct-Absmax W4A4 | | uniform fake quant |
| FFN Rot-Absmax W4A4 | | rotation + uniform |
| FFN Rot-LM W4A4 | | non-uniform same-bit comparison |
| FFN Rot-LM W3A4 | | weight bit reduction |
| FFN Rot-LM W4A3 | | activation bit reduction |

---

## 表 5：C3 Attention-local KV Error

| Method | K bits | V bits | IP bias | IP var | Score MSE | Softmax KL | Output cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absmax | 4 | 4 | | | | | |
| Absmax | 3 | 4 | | | | | |
| Absmax | 4 | 3 | | | | | |
| Hadamard-LM | 4 | 4 | | | | | |
| Hadamard-LM | 3 | 4 | | | | | |
| Hadamard-LM | 4 | 3 | | | | | |
| Hadamard-LM | 3 | 3 | | | | | |
| Hadamard-LM | 2 | 4 | | | | | |

---

## 表 6：C4 QJL Residual

| Method | Base K bits | Residual bits | Value | IP bias | Score MSE | Softmax KL | Output cosine |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Hadamard-LM K3 | 3 | 0 | V4 fixed | | | | |
| Hadamard-LM K2 | 2 | 0 | V4 fixed | | | | |
| Hadamard-LM K2 + QJL residual | 2 | 1 | V4 fixed | | | | |
| Hadamard-LM K3 + QJL residual | 3 | 1 | V4 fixed | | | | |

---

## 表 7：C5 Attention-layer Structured Quant PPL / Accuracy

| Method | Attention Linear | KV cache | Value path | Layer output err | PPL | Accuracy | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Attn-FP16 | FP16 | FP16 | reference | | | | baseline |
| Attn-Identity-FP16 | FP16 wrapper | FP16 | identity | | | | wrapper sanity |
| Attn-KV-HLM K4V4 | FP16 | HLM K4V4 | o_proj absorb | | | | KV-only 4-bit |
| Attn-KV-HLM K3V4 | FP16 | HLM K3V4 | o_proj absorb | | | | key 降位宽 |
| Attn-Rot-LM W4A4 + HLM K4V4 | Rot-LM W4A4 | HLM K4V4 | o_proj absorb | | | | structured baseline |
| Attn-Rot-LM W3A4 + HLM K3V4 | Rot-LM W3A4 | HLM K3V4 | o_proj absorb | | | | weight/key 降位宽 |
| Attn-Rot-LM W4A3 + HLM K4V3 | Rot-LM W4A3 | HLM K4V3 | o_proj absorb | | | | activation/value 降位宽 |
| Attn-Rot-LM W3A4 + HLM K2+QJL,V4 | Rot-LM W3A4 | HLM K2+QJL,V4 | o_proj absorb | | | | C4 成功后的组合 |

核心比较：

$$
\text{Attn-Rot-LM W3A4 + HLM K3V4}\stackrel{?}{\approx}\text{Attn-Rot-LM W4A4 + HLM K4V4}
$$

以及：

$$
\text{K2+QJL,V4}\stackrel{?}{\approx}\text{K3V4}
$$

---

# F. 第一阶段核心判断标准

## A 线问题

> Hadamard rotation 是否让权重量化更容易？Lloyd-Max 是否能把 weight-only 可用位宽从 W4 推到 W3？

判断依据：

* weight MSE 是否下降；
* layer output error 是否下降；
* A16W4 / A16W3 PPL 是否改善；
* Hadamard-LM W3 是否接近或优于 Hadamard-Absmax W4。

---

## B 线问题

> 在旋转域 W/A fake quant 中，Lloyd-Max 是否能支持 W3A4 / W4A3，而不仅仅是在 W4A4 同 bit 下略好？

判断依据：

* Rot-LM 是否优于 Direct-Absmax；
* Rot-LM W3A4 是否接近 Rot-Absmax W4A4；
* Rot-LM W4A3 是否接近 Rot-Absmax W4A4；
* FFN-only model-level PPL 是否仍保持优势。

解释边界：

> 这说明非均匀 codebook 有潜力降低位宽，但不能直接说明推理更快。

---

## C 线问题

> Hadamard-LM 是否能改善 KV cache quantization？QJL residual 是否能进一步保持 attention score？当 q/k/v/o 线性 fake quant 和 KV cache low-bit quant 串成完整 Attention 层时，PPL 是否仍可接受？

判断依据：

* Key inner product bias 是否下降；
* attention score MSE 是否下降；
* softmax KL 是否下降；
* attention output cosine 是否提升；
* C4 是否明显优于 C3 pure Hadamard-LM。
* C5 Attention-only model-level PPL 是否接近 FFN-only fake quant 的 PPL；
* `Attn-Rot-LM W3A4 + HLM K3V4` 是否接近 `Attn-Rot-LM W4A4 + HLM K4V4`；
* `K2+QJL,V4` 是否能接近或优于 pure `K3V4`。

---
