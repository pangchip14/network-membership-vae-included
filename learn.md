# 共享 Membership 模型的概率化建模（VAE 版本）

## 1. 问题设置与记号

我们有 \(N=72\) 个样本（被试 × 条件），每个样本对应一个矩阵：

\[
X^{(n)} \in \mathbb{R}^{B \times V}, \quad n=1,\dots,N
\]
其中：

- \(B = 140\)：时间点（或特征维度）
- \(V = 50{,}000\)：体素数

目标是将每个样本分解为：

\[
X^{(n)} \approx S^{(n)} Z
\]

其中：

- \(Z \in \mathbb{R}^{K \times V}\)：**跨样本共享的空间 membership 矩阵**
- \(S^{(n)} \in \mathbb{R}^{B \times K}\)：**样本 \(n\) 特有的时间/强度系数**
- \(K = 14\)：网络（membership pattern）数量

---

## 2. Membership 矩阵 \(Z\) 的约束（体素归属建模）

我们希望表达一个**脑科学上的假设**：

> 每个体素主要属于少数几个网络（membership 尖锐），  
> 但网络之间可以重叠（soft assignment）。

因此，对 **每一个体素** \(v\)，其在 \(K\) 个网络上的 membership 满足概率约束：

\[
\sum_{k=1}^K Z_{k,v} = 1,\quad Z_{k,v} \ge 0
\]

我们通过引入未归一化参数 \(A \in \mathbb{R}^{K \times V}\)，并对 **network 维度** 做 softmax：

\[
Z_{:,v} = \mathrm{softmax}\!\left(\frac{A_{:,v}}{\tau}\right)
\]

其中：

- \(\tau > 0\) 为温度参数
  - \(\tau \downarrow\)：membership 更尖锐（接近 one-hot）
  - \(\tau = 1\)：标准 softmax

---

## 3. 生成模型（Generative Model）

我们将样本特异矩阵 \(S^{(n)}\) 视为隐变量，并建立如下生成过程：

1. **隐变量先验**  
   对每个样本 \(n\) 和每个时间点 \(b\)：
   \[
   s^{(n)}_b \sim \mathcal{N}(0, I_K)
   \]

2. **线性生成均值**
   \[
   \mu^{(n)}_b = s^{(n)}_b Z \in \mathbb{R}^{V}
   \]

3. **观测噪声模型**
   \[
   x^{(n)}_b \mid s^{(n)}_b, Z \sim \mathcal{N}\!\left(\mu^{(n)}_b, \sigma^2 I_V\right)
   \]

整体可写为：

\[
X^{(n)} = S^{(n)} Z + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
\]

---

## 4. 变分推断（Encoder）

由于后验 \(p(S^{(n)} \mid X^{(n)}, Z)\) 不可解析，我们用一个神经网络近似：

\[
q_\phi(S^{(n)} \mid X^{(n)}) = \prod_{b=1}^B \mathcal{N}
\bigl(
s^{(n)}_b \mid \mu_\phi(x^{(n)}_b), \mathrm{diag}(\sigma_\phi^2(x^{(n)}_b))
\bigr)
\]

其中：

- 编码器输入：\(x^{(n)}_b \in \mathbb{R}^{V}\)
- 输出：\(\mu_\phi(x^{(n)}_b), \log \sigma^2_\phi(x^{(n)}_b) \in \mathbb{R}^{K}\)

### 重参数化技巧

\[
s^{(n)}_b = \mu_\phi(x^{(n)}_b) + \epsilon \odot \exp\!\left(\tfrac{1}{2}\log\sigma^2_\phi(x^{(n)}_b)\right),
\quad \epsilon \sim \mathcal{N}(0,I)
\]

---

## 5. 训练目标（ELBO）

对每个样本 \(X^{(n)}\)，最大化证据下界（ELBO），等价于最小化：

\[
\mathcal{L} =
\underbrace{
\mathbb{E}_{q_\phi}\!\left[\|X^{(n)} - S^{(n)}Z\|_F^2\right]
}_{\text{重建误差}}
+
\beta\,
\underbrace{
\mathrm{KL}\!\left(q_\phi(S^{(n)} \mid X^{(n)}) \,\|\, p(S^{(n)})\right)
}_{\text{变分正则}}
\]

其中 \(\beta\) 为 KL 权重（可采用 warm-up）。

---

## 6. Membership 的结构正则化

### 6.1 使用率正则（防止网络塌缩）

定义第 \(k\) 个网络的平均使用率：

\[
\bar{z}_k = \frac{1}{V} \sum_{v=1}^V Z_{k,v}
\]

令均匀分布 \(\pi_k = \frac{1}{K}\)，加入：

\[
\mathcal{L}_{\text{usage}} =
\mathrm{KL}(\bar{z} \,\|\, \pi) =
\sum_{k=1}^K \bar{z}_k \log \frac{\bar{z}_k}{\pi_k}
\]

该项**只防止某些网络完全消失**，不强迫严格均匀。

### 6.2 尖锐度正则（voxel-wise entropy）

对每个体素 \(v\)，其 membership 熵为：

\[
H_v = -\sum_{k=1}^K Z_{k,v} \log Z_{k,v}
\]

整体 entropy 正则为：

\[
\mathcal{L}_{\text{ent}} =
\frac{1}{V} \sum_{v=1}^V H_v
\]

最小化该项可促使每个体素集中于少数网络。

---

## 7. 完整损失函数

综合所有项，训练最小化：

\[
\boxed{
\mathcal{L} =
\mathcal{L}_{\text{rec}}
+
\beta\,\mathcal{L}_{\text{KL}}
+
\lambda_{\text{usage}}\,\mathcal{L}_{\text{usage}}
+
\lambda_{\text{ent}}\,\mathcal{L}_{\text{ent}}
+
\lambda_{s}\,\mathbb{E}\|s\|_2^2
}
\]

其中最后一项用于稳定隐变量尺度（可选）。

---

## 8. 共享结构为何能被学到？

因为 **所有样本共享同一个 \(Z\)**：

- 每个样本的重建误差都会对 \(Z\) 产生梯度
- 不同样本的梯度在 \(Z\) 上累积
- 个体差异被 \(S^{(n)}\) 吸收

最终：

- \(Z\)：捕捉 **跨被试稳定的空间 membership 模式**
- \(S^{(n)}\)：表达 **个体/条件特异的时间动态**

---

## 9. 与传统矩阵分解的关系

- 若忽略 KL 项并令编码器为线性映射：  
  → 退化为 **共享字典学习 / 低秩分解**
- VAE 框架提供：  
  - 不确定性建模  
  - 结构化先验  
  - 对噪声与小样本更稳健的估计

---

**一句话总结**：  
> 本模型是一个带有体素归属约束的共享因子 VAE，  
> 在保证可解释性的同时，学习跨样本一致的脑网络空间模式。
