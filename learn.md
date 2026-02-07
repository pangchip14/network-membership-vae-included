# 两阶段 Membership 学习框架总结（Stage 1 + Stage 2）

> 目标：  
> 从 72 个样本（12subject × 6condition），每个样本为 **140 × 50000** 的时序–体素矩阵中，  
> 学习 **一组全局共享的脑网络空间模式**（membership patterns），  
> 并为每个时间点学习其在这些网络上的激活强度。

---

## 一、问题形式化

### 1. 数据结构

- 样本数：72（12 subjects × 6 conditions）
- 每个样本：
  - 时间点数：\( B = 140 \)
  - 体素数：\( V = 50000 \)
  - 网络数（隐变量维度）：\( K = 14 \)

对第 \(n\) 个样本，观测数据为：

\[
X^{(n)} \in \mathbb{R}^{B \times V}
\]

---

### 2. 核心分解假设（共享低秩结构）

我们假设所有样本共享同一组空间 membership 模式：

\[
X^{(n)} \;\approx\; S^{(n)} Z
\]

其中：

- \( Z \in \mathbb{R}^{K \times V} \)：**全局共享**
  - 第 \(k\) 行表示第 \(k\) 个脑网络在所有 voxel 上的空间分布
  - 第 \(a\) 列表示第 \(a\) 个 voxel 在所有脑网络上的空间分布
  - 对每个 voxel 做 softmax，使其在 \(K\) 个网络上的 membership 之和为 1
- \( S^{(n)} \in \mathbb{R}^{B \times K} \)：**样本私有**
  - 每个时间点在 \(K\) 个网络上的激活强度

---

## 二、为什么要用“两阶段”？

直接用 VAE 从头学习存在严重困难：

- 初始对称性强（Z 极易收敛到均匀解）
- KL 项会把隐变量压回先验，导致 posterior collapse（后验坍塌）
- Z 的空间结构难以稳定形成

因此我们采用 **Two-Stage Strategy**：

> **Stage 1：确定性分解，先把“网络结构”学出来**  
> **Stage 2：在此基础上引入 VAE，做概率化与稳健微调**

---

## 三、Stage 1：确定性 Membership 分化（破对称阶段）

### 1. 模型形式

**Encoder（确定性）**：
\[
s = f_\theta(x), \quad s \in \mathbb{R}^{B \times K}
\]

**Decoder（线性）**：
\[
\hat{x} = s Z
\]

其中：

- \(Z = \mathrm{softmax}(Z_{\text{logits}} / \tau, \text{dim}=0)\)
- softmax 沿 \(K\) 维，对每个 voxel 归一

---

### 2. Stage 1 的核心目标

- **快速打破网络对称性**
- 学出清晰、尖锐、可解释的空间 membership
- 同时防止：
  - encoder 输出幅度爆炸
  - 少数网络“吃掉”所有 voxel

---

### 3. Stage 1 损失函数

\[
\mathcal{L}_{\text{Stage1}} =
\underbrace{\|x - sZ\|_2^2}_{\text{重建误差}} + \lambda_{\text{sharp}} \underbrace{H(Z)}_{\text{尖锐化（降熵）}} + \lambda_{\text{usage}} \underbrace{\mathrm{KL}(\bar z \,\|\, \text{Uniform})}_{\text{防网络塌缩}} + \lambda_{s} \underbrace{\|s\|_2^2}_{\text{控制 } S \text{ 幅度}}
\]

各项含义：

- **重建误差**：保证模型能解释数据
- **Entropy 正则（sharpness）**：
  - 降低 voxel-wise entropy
  - 推动每个 voxel 更偏向少数网络
- **Usage KL**：
  - 防止所有 voxel 都被分到同一个网络
- **\( \|s\|_2^2 \)**：
  - 防止 encoder 通过无限放大 \(s\) 来“作弊拟合”

---

### 4. 关键工程策略（为什么 Stage 1 能成功）

1. **Z 的学习率显著高于 encoder**
   - 强制 Z 快速破对称
2. **Temperature annealing**
   - \( \tau: 1.0 \rightarrow 0.7 \rightarrow 0.5 \)
3. **对 s 做 LayerNorm / L2 约束**
   - 保证分化是真正来自 Z
4. **不引入 KL（非概率模型）**
   - 避免早期被先验拉平

---

### 5. Stage 1 的成功判据

- entropy 明显低于 \( \log K \)
- z_usage_min / max 明显拉开
- s2（\(\|s\|_2^2\)）稳定在合理范围（非爆炸）

---

## 四、Stage 2：VAE Fine-tune（概率化与稳健阶段）

Stage 2 的前提是：

> **Stage 1 已经学到了一个“结构上正确”的 Z**

---

### 1. 生成模型（Decoder）

\[
p(x \mid s, Z) = \mathcal{N}(sZ,\; \sigma^2 I)
\]

其中：

- \(Z\) 继承自 Stage 1，并继续微调
- Decoder 仍是线性（保持可解释性）

---

### 2. 变分后验（Encoder）

\[
q_\phi(s \mid x) = \mathcal{N}(\mu_\phi(x),\; \mathrm{diag}(\sigma^2_\phi(x)))
\]

通过重参数化：

\[
s = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)
\]

---

### 3. Stage 2 损失函数（VAE）

\[
\mathcal{L}_{\text{Stage2}} =
\underbrace{\|x - sZ\|_2^2}_{\text{重建}}+ \beta \underbrace{\mathrm{KL}\big(q(s|x)\,\|\,p(s)\big)}_{\text{概率约束}}+ \lambda_{\text{sharp}} H(Z)+ \lambda_{\text{usage}} \mathrm{KL}(\bar z \,\|\, \text{Uniform})
\]

---

### 4. 防止 Posterior Collapse 的关键：Free Bits

对每个样本的 KL：

\[
\mathrm{KL}_i \leftarrow \max(\mathrm{KL}_i,\; \text{free\_nats})
\]

作用：

- 强制 latent 变量 **至少携带一定信息**
- 避免 KL → 0 导致 VAE 退化为普通自编码器

---

### 5. Stage 2 的训练策略

- **Z 初始化自 Stage 1（核心）**
- \( \beta \) 采用 warm-up（逐步增加）
- Z 的学习率 **低于 Stage 1**
- entropy 正则减弱（防止过度尖锐）
- \(\tau\) 固定在 Stage 1 的合理值（如 0.7）

---

## 五、Stage 1 vs Stage 2 的角色分工

| 阶段 | 作用 | 关注重点 |
|---|---|---|
| Stage 1 | 结构发现 | 破对称、尖锐化、空间网络成形 |
| Stage 2 | 概率化 | 稳健性、不确定性、泛化能力 |

一句话总结：

> **Stage 1 决定“网络长什么样”**  
> **Stage 2 决定“这些网络在时间上如何被激活，以及不确定性如何表达”**

---

## 六、最终产物与解释

- `Z.npy`：
  - 形状 \( (K, V) \)
  - 每个 voxel 在 14 个网络上的 membership
  - 可用于脑网络可视化、Top-1 网络划分、entropy map
- `voxel_pick.npy`：
  - 对应 voxel 索引，可映射回全脑空间

---

## 七、方法论总结

> 我们采用一个 **“先确定性破对称、再概率化微调”** 的两阶段学习框架，  
> 在保证空间网络结构可解释性的前提下，引入 VAE 对时间动态进行稳健建模，  
> 从而有效地学习到 **跨被试共享的脑网络空间模式及其不确定性表达**。
