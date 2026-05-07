### 01 Hadamard Rotation 定义和性质

* **Hadamard 矩阵定义**
* **正交性 (Orthogonality)**

  * **描述：** **$Q \cdot Q^T = I$**。这意味着旋转操作保留了向量的范数（L2 norm）。
* **自逆性 (Self-Inverse)**

  * **描述：** **$H^{-1} = H$**。Hadamard 矩阵是正交且对称的，因此其逆矩阵就是其本身。
* **计算不变性 (Computational Invariance)**

  * **描述：**$A \times W^T = (A \times H) \times (H^T \times W^T)$。在 Transformer 架构中，将旋转矩阵插入权重和激活值之间不会改变最终的线性计算结果，且可以离线合并到权重中。
  * 配图：figures/computational invariance.png
* **分布同质化 (Gaussianity/Energy Smoothing)**

  * **描述：** 旋转能将激活值中集中的异常能量均匀打散到各个维度，使其趋向于正态分布，极大地削弱了 Outliers。
* **Kronecker 分解性质 (Kronecker Decomposition)**

  * **描述：** 针对非 2 的幂次（Non-PoT）的大维度通道，Hadamard 矩阵可以分解为 **$H_{Ch} = H_N \otimes H_M$**。

---

### 02 引入 Hadamard Rotation 的原因

* **LLM 部署的终极瓶颈：Memory Wall**
  * 在自回归解码（Autoregressive Decoding）阶段，主要延迟来自于从外部存储读取权重（Weight EMA） 。必须采用低位宽量化（如 W4A4, W4A8）来缓解带宽压力。
* **传统量化无法解决的问题：Activation Outliers**
  * 激活值中存在极少但数值极大的 Outliers（异常值），它们在低位宽下会导致严重的量化截断误差，使得模型精度崩溃 。普通的 Min-Max 甚至 PTQ（训练后量化）都难以消除这些异常值 。
* **Hadamard Rotation 的破局：Outlier-Free**
  * 引入旋转后，激活值分布变得极其平滑。
  * **收益：** 成功在保证模型精度的前提下（Perplexity 几乎不掉），实现了超低位宽量化（如 W4A8 下 PPL 显著优于传统直接量化）。
  * 配图：figures/outlier-free.png

---

### 03 Hadamard Rotation 硬件映射

**FWHT蝶形网络计算**

* **$O(N \log_2 N)$ 对数级硬件复杂度** ：打破传统矩阵变换 **$O(N^2)$** 的硬件复杂度，实现计算开销的指数级压缩。
* **极简数据路径** ：剥离高昂的乘法单元，核心部件仅由基础加减法器与前馈选择网络（MUX/Routing）构成。
* **PPA 优化** ：极简门级逻辑大幅缩短关键路径延迟，实现吞吐率 (Speed) 与功耗 (Power) 的双重提升。
* 配图：figures/FWHT.png


**全局旋转的硬件开销过大**

* **映射方案：子空间旋转 (Subspace Rotation) / 局部旋转 (Local Rotation)**
* 巨大的通道维度如果直接做全局 FWHT 会导致难以承受的面积开销（深度过深） 。
* 配合Group Quantization使用的Local Rotation不影响模型精度。
* 通过将全局旋转分解为重叠的局部旋转（Local Rotation） ，或者使用 Kronecker 分解将其拆分为 **$O(N \log_2 N)$** 的 FHT 单元和极小的矩阵乘法 (MM) 单元 ，可以在极小的面积下达到等效的精度。
* Local Rotation配图：figures/local rotation.png
* Kronecker Decompsition配图:figures/kronecker decomposition.png

---

### 04 超低位宽量化 Demo

* **Demo 实验目的：**
  * 验证 Hadamard Rotation 是否能在 W4A4（或类似超低位宽）下有效缓解 Attention (Q, K 矩阵) 的量化精度损失。
* **Demo 流程设计说明：**
  1. **数据合成与异常注入：** 随机生成全精度的 Q 和 K 矩阵，并人为在少数 Token 上注入极大的 Outlier（模拟长尾分布）。
  2. **Baseline 1 (全精度)：** 计算标准的 FP32 Attention Scores (**$Q \cdot K^T / \sqrt{d}$**)。
  3. **Baseline 2 (直接量化)：** 对注入 Outlier 的 Q 和 K 直接使用 Min-Max 对称量化至 4-bit，计算 Attention 并评估 MSE Error。
  4. **Rotation 组：** 生成对应的 Hadamard 矩阵，对 Q 和 K 分别进行旋转 (**$Q_{rot} = Q \cdot H, K_{rot} = K \cdot H$**)。在旋转域内进行 4-bit 量化，利用正交性直接计算 Attention Scores。
* **预期展示图表：**
  * **分布热力图/直方图 (Histograms)：** 直观对比旋转前（存在刺眼的 Outlier 尖峰）和旋转后（呈现完美的中心对称高斯分布）的激活值状态。
  * **误差对比柱状图：** 展示直接量化产生的巨大 MSE 误差，对比加入 Hadamard Rotation 后 MSE 误差的断崖式下降。
