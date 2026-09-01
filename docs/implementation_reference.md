# 实现对照报告

权威设计是 [`docs/model.md`](model.md)。本文件只记录官方仓库/论文作为**分项数学参考**，不是端到端架构。实现前必须在 provenance 中登记：仓库 URL、commit/release、原文件/函数、许可证、改编行为、本地落点。许可证缺失或不兼容则只按论文重写，不复制代码。

不从原仓库静默引入：数据预处理、图方向、输出形状、城市/节点常数、H 步 decoder。

---

## 1. DCRNN（扩散卷积与 DCGRU）

| 项 | 内容 |
| --- | --- |
| 论文 | Li, Yu, Shahabi, Liu. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*. ICLR 2018. https://arxiv.org/abs/1707.01926 |
| 官方仓库 | https://github.com/liyaguang/DCRNN （TensorFlow） |
| 对照 commit | `602afd9d767d3aa1c9b3eac51710d6aeee12c227`（master，2024-12-09） |
| 许可证 | MIT，Copyright (c) 2017 Yaguang Li |
| 辅助 PyTorch 端口 | https://github.com/chnsh/DCRNN_PyTorch ，同样 MIT / Copyright (c) 2017 Yaguang Li。只作张量实现对照，不替代官方方程。 |
| 原文件 | `model/dcrnn_cell.py`：`DCGRUCell`、`_gconv`；`lib/utils.py`：`calculate_random_walk_matrix` |

### 要对照的方程

`filter_type=dual_random_walk`：

- $T = D_{\mathrm{out}}^{-1} A$
- 两个 support：$S_{\mathrm{fwd}} = T^\top$，$S_{\mathrm{bwd}} = (D_{\mathrm{in}}^{-1} A^\top)^\top$
- 扩散：$x^{(1)} = S x^{(0)}$，$x^{(k)} = 2 S x^{(k-1)} - x^{(k-2)}$（$k \ge 2$）
- 门：$[r,u] = \sigma(\mathrm{gconv}(x,h))$，偏置初值 1
- 候选：$c = \tanh(\mathrm{gconv}(x, r \odot h))$
- 状态：$h' = u \odot h + (1-u) \odot c$

### 复用 / 改编 / 重写

- **复用**：上述扩散与 GRU 门方程；双 support 分开传入。
- **改编**：encoder-only，输入 `[B, 7, N, 9]`，7 步后取 $H_{\mathrm{final}} \in \mathbb{R}^{B \times N \times R}$，再接概率头得到 `[B, N, 1]`。$N$ 来自 `node_order`，不写死。
- **不复制**：seq2seq decoder、curriculum / teacher forcing、固定 `adj_mx` 预处理、METR-LA 路径与节点数、无条件对称化、只留一个 support、平均正反向 support。

### 与批准架构的差异

原模型做多步交通序列；我们做单点 $Y_{t+7}$。原模型一张图；我们用融合后的 $S_{\mathrm{fwd/bwd}}$。

### 等价性测试

- 小不对称图上 $T$ 行和、$S$ 列和与官方 `random_walk(A).T` 一致。
- $K=2$ 时 Chebyshev 递推与手算 $S x$、$2Sx_1-x_0$ 一致。
- 单步 DCGRU 在固定权重下与官方门公式数值对齐（容差内）。
- 输出形状是 `[B, N, 1]`，不是 `[B, H, N, 1]`。

本地落点：`src/model/dcrnn_encoder.py`。保留 MIT 版权声明。

---

## 2. UQGNN 启发的一元概率头

| 项 | 内容 |
| --- | --- |
| 论文 | Yu et al. *UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction*. SIGSPATIAL 2025. https://arxiv.org/abs/2508.08551 |
| 官方仓库 | https://github.com/UFOdestiny/UQGNN |
| 对照 commit | `52ccc80df20409680b2476a0cbd4b4de29da829c`（2025-08-31） |
| 许可证 | MIT（GitHub `license.key=mit`） |
| 原模块 | 论文 MPP：输出 $\boldsymbol\mu$ 与正定 $\boldsymbol\Sigma$，最小化多元高斯 NLL |

### 要对照的解释（不是整网）

UQGNN 把预报看成条件分布，而不只是点估计。多元高斯时：

$$
-\log p(x\mid \mu,\Sigma) = \tfrac12 \big( d\log(2\pi) + \log\det\Sigma + (x-\mu)^\top \Sigma^{-1}(x-\mu) \big)
$$

我们只有一个动态响应，协方差退化为 $1 \times 1$ 的 $v$：

$$
v_z = \mathrm{softplus}(\mathrm{raw}) + \varepsilon, \quad
\sigma_z = \sqrt{v_z}, \quad
\mathrm{NLL} = \tfrac12 \big( \log(2\pi) + \log v_z + (y_z-\mu_z)^2 / v_z \big)
$$

对外名称必须是：**UQGNN-inspired univariate probabilistic prediction head**。不得声称复现了完整 UQGNN。

### 复用 / 改编 / 重写

- **复用**：概率输出 + 高斯 NLL 这一解释。
- **改编**：一元头；$\varepsilon=10^{-6}$。
- **不复制**：ISTE、MDGCN、ITCN、多元 $\Sigma$ 的三角填充/特征值修正、用不确定度再去改点预报的闭环、多现象输入。

### 与批准架构的差异

原模型对多个城市现象建 $\Sigma$；我们只有感染率。原模型是独立时空骨干；我们的骨干是自适应多图 DCRNN encoder。

### 等价性测试

- $d=1$ 时多元 NLL 与批准的一元 NLL 数值相同。
- $v>0$，$\sigma^2=v$。
- 反变换：$v_{\mathrm{orig}}=v_z s^2$，$\sigma_{\mathrm{orig}}=\sigma_z |s|$，不加 mean。
- 代码与文档字符串含规定名称，不含 “full UQGNN”。

本地落点：`src/model/heads.py`。

---

## 3. GeoShapley

| 项 | 内容 |
| --- | --- |
| 论文 | Li, Z. (2024). *GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models*. Annals of the AAG. https://doi.org/10.1080/24694452.2024.2350982 |
| 官方仓库 | https://github.com/Ziqi-Li/geoshapley |
| PyPI | `geoshapley`（查阅时最新为 0.2.x） |
| 对照 commit | `0d542fc40849267eb3e9f6cb498808fee4cc4a61`（2026-06-08） |
| 许可证 | MIT |
| 原文件 | `geoshapley/geoshapley.py`：`GeoShapleyExplainer`、`_shapley_kernel`、`_precompute_Z_matrix`、`explain` |

### 官方实现在做什么

- 最后 $g=2$ 列是联合坐标，前面 $k$ 列是特征。$k=6$ 时精确联盟数 $2^{k+1}=128$。
- 设计矩阵 $Z$：特征主效应 $k$ 列 + GEO 1 列 + GEO×特征 $k$ 列，共 $2k+1=13$ 列。
- $\phi_0$ **不在**回归里：`base_value = mean(predict(background))`，先从 $y$ 里减掉。
- 空/全集核权重官方写成 `return 100000000`（大权重近似）。
- 输出：`primary`、`geo`、`geo_intera`。

### 必须报告的冲突（不改批准设计）

| 官方包 | 批准的 `docs/model.md` |
| --- | --- |
| 13 个回归系数 + 背景均值作 $\phi_0$ | 14 列，含截距；$\phi_0 = f_i(\varnothing)$ |
| 空/全集用 $10^8$ 权重 | **等式约束**，不用大权重 |
| 表格 `predict_f(X)`，逐行改特征 | 图模型：只改目标 IZ，重算整图 embedding 再预报 |
| `base_value` 来自 background 均值 | baseline 是空联盟的模型输出 |

因图模型无法直接当表格包装，且端点约束与官方大权重冲突：**按论文 + 批准规格重写**，不调用 `geoshapley` 包做主路径，也不改成普通 SHAP。

### 复用 / 改编 / 重写

- **复用（方法）**：联合 location；主效应 + 内蕴位置 + location–feature 交互；128 联盟；Shapley 核 $w(S)=(n-1)/[\binom{n}{s}s(n-s)]$（$0<s<n$）。
- **改编**：target-IZ-local 联盟；冻结 scaler；坐标在最后两列但只作为联合 player。
- **重写**：带等式约束的核加权最小二乘；$\phi_0=f_i(\varnothing)$；分解严格重构 $f_i(\mathrm{observed})$。
- **不复制**：官方 `100000000` 核、background 均值 baseline、把交互对半折回普通 SHAP、解释 embedding 维。

### 等价性测试

- 128 联盟、14 列（或约束消去后的自由参数）形状正确。
- $0<s<n$ 的核与官方 `_shapley_kernel` 一致。
- $\phi_0=f(\varnothing)$；$\phi_0+\phi_{\mathrm{loc}}+\sum\phi_j+\sum\phi_{\mathrm{loc},j}=f(\mathrm{observed})$（浮点容差）。
- 联盟内：在 $S$ 中用观测值，不在 $S$ 中用参考值；其他 IZ 不变。
- 无坐标则无 `location` 也无 `location_x_*`。
- 输出名不含 `embedding_*`。

本地落点：`src/model/geoshapley.py`。THIRD_PARTY 记录论文与 MIT 版权。

---

## 4. 自适应图融合（项目扩展）

无官方仓库。不从 DCRNN/UQGNN/GeoShapley 引入。

- $\alpha=\mathrm{softmax}(\theta)$，一套全局权重。
- 先各自做 $S_{\mathrm{fwd/bwd}}$，再 $\sum_k \alpha_k S_k$。
- 同一 $\alpha$ 给上下文 encoder 与 DCRNN。
- 不预加原始 $A$，不对称化 transport/mobility，融合后不重归一。

测试：$\alpha>0$ 且和为 1；正反向分开；checkpoint 的 `graph_set` 与 $\alpha$ 维绑定。

本地落点：`src/model/diffusion.py`。

---

## 5. 版权与登记模板

实现时每个改编文件头（或 `THIRD_PARTY_NOTICES`）填写：

```text
Source: <url>
Commit: <sha>
File: <original path>
Licence: MIT
Adapted: <what changed>
Destination: src/model/<file>
```

复制实质代码时保留原版权声明。`old code/` 只作历史对照，不作为许可证来源。

---

## 6. 结论

看懂且可执行：

1. 端到端以 `docs/model.md` 为准。
2. DCRNN 对照官方 cell 的扩散与门，做成 encoder-only。
3. UQGNN 只借鉴一元高斯 NLL 头，并标明名称。
4. GeoShapley 对照 Li (2024) 的分量定义与核，但用批准的等式约束重写包装。
5. α 融合是本项目的，不假装来自上述仓库。

未发现需要改预测目标、S1 划分、图语义、不确定度定义、GeoShapley 输出或 checkpoint/校准设计的冲突；唯一登记的差异是官方 GeoShapley 用大权重 + 13 列 + 背景均值，我们保持 14 列等式约束。
