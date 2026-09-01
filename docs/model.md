# OASIS COVID-19 预报与解释模型

Probabilistic Adaptive Multi-Graph DCRNN Encoder with Contextual Node Embedding

本文是已批准的技术方案。实现必须遵守本文。不要改单点预报、共享 α、一元 UQGNN 启发方差头、或上下文嵌入。解释必须是完整的 target-local GeoShapley（含 location–feature 交互），不要只做联合位置 Shapley。

---

## 0. 范围

### 做什么

- 对每个 IZ 预报一个滚动七日感染率 $Y_{t+7}$
- 输出预测均值、方差、标准差与区间
- 用三张静态图做有向扩散，再用一套全局 α 融合
- 用六个 SIMD 变量和投影质心做上下文嵌入
- 用完整 target-local GeoShapley 解释：各原始变量、位置主效应、以及位置与每个变量的交互

### 不做什么

- 不从 raw 重建 L7_H7_S1 窗口
- 主实验外层为同一批窗口按 target_date 的 65/10/25；不覆盖已归档的 70/15/15
- 不预报 H 个未来日（无 `[B, H, N, 1]`）
- 不实现多元 UQGNN、第二动态响应、MC Dropout 主不确定度
- 不做第二套 α、节点/日期/horizon 专属 α、时间 β
- 不对称化交通图或 mobility 图
- 不先加原始邻接再归一，不平均正反向 support
- 不把 embedding 维当作解释变量
- 不静默改图集合、不静默重归一 α、不静默重排 IZ
- 层内不把 `N = 111` 写成模型常数
- 不在 GeoShapley 联盟中重算任何 scaler
- 不用 test 做 checkpoint 选择、区间校准或不确定阈值
- 不把经验校准说成具有交换性假设下的正式覆盖保证
- 校准样本不足时不截断 $k$，而是标记 calibration unavailable

### 默认实验

| 项 | 值 |
| --- | --- |
| 配置 | L7_H7_S1 |
| L | 7 个历史滚动七日率 |
| H | 7 个报告日的提前期 |
| 步长 | 1 |
| 研究区 | City of Edinburgh，2011 IZ |
| 期望节点数 | 111（数据校验，不是层常数） |
| 外层划分 | 按 target_date 的 65/10/25（同一批 S1 窗口重切） |

---

## 1. 动态响应：COVID 滚动七日率

### 数据

目录：`data/results/forecast/L7_H7_S1_20200308_20230225_split65_10_25/`

归档的 70/15/15 仍在 `data/results/forecast/L7_H7_S1_20200308_20230225/`，不要覆盖。

| 文件 | 用途 |
| --- | --- |
| `train.npz` / `validation.npz` / `test.npz` | 已切好的张量 |
| `scaler.csv` | 按 IZ 的训练均值与标准差 |
| `node_order.csv` | 规范 IZ 顺序 |
| `valid_samples.csv` / `split_manifest.csv` | 样本日期与分区 |
| `run_metadata.json` / `array_integrity.json` | 口径与形状校验 |

`load_temporal_dataset()` 只加载并校验，不调用窗口生成。

### 变量定义

$Y_d$：日期 $d$ 报告的滚动七日感染率（每 10 万人）。不是日新发病例，不是 H 日累计。

对 issue date $t$：

$$
[Y_{t-6},\ldots,Y_t] \rightarrow Y_{t+7}
$$

- `issue_date` = $t$
- `input_start_date` = $t-6$
- `target_report_date` = $t+7$
- `lookback_steps` = 7（截至 $t$ 的输入步数）
- `target_offset_days` = 7（目标相对 $t$ 的天数，不是输出步数）
- `output_steps` = 1（每个 IZ 只输出一个 $Y_{t+7}$，形状为 `[B, N, 1]`，不是 `[B, 7, N, 1]`）
- `window_stride_days` = 1

### 外层划分（同一批 S1 窗口，主实验 65/10/25）

S1 先生成合法窗口。主实验按 **target_date** 做 `floor(0.65)/floor(0.10)/余数`，得到 685 / 105 / 264。

| 划分 | 样本 | 目标日 t+7 |
| --- | --- | --- |
| train | 685 | 2020-03-21 → 2022-02-21 |
| validation | 105 | 2022-02-22 → 2022-06-06 |
| test | 264 | 2022-06-07 → 2023-02-25 |

验证/测试 lookback 可以落在前一段日历上。这是 issue 时已有的历史，不是未来标签。原先的 70/15/15 仅作归档对照。

缺日窗口已排除。缺失观测不得填 0。

### 验证集内部用法划分

外层 `train` / `validation` / `test` 不变。

在已有 **validation** 分区内，再按 target_date 做时间顺序切分：

| 内部集 | 用途 |
| --- | --- |
| `validation_selection` | early stopping、checkpoint 选择 |
| `validation_calibration` | 冻结 checkpoint 之后，只用于 $q_{95}$ 与 σ 的 P90 阈值 |

默认：按 target_date 排序后，前一半样本为 selection，后一半为 calibration。比例可配，但必须写入 provenance。

禁止：

- 用 `validation_calibration` 更新模型参数
- 用 test 做 checkpoint、校准或阈值
- 改变已写入的外层分区边界（主实验已定为 65/10/25）

保存内部划分的日期范围与样本数。

### 标准化

对 IZ $i$，只用训练输入日：

$$
z_{d,i} = \frac{Y_{d,i} - \mu^{\mathrm{covid}}_i}{s_i}, \qquad
s_i = \max(\hat\sigma_i, \varepsilon_{\mathrm{covid}})
$$

验证、测试、未来推理、GeoShapley 全部沿用同一套冻结的 $\mu^{\mathrm{covid}}, s$。不用 COVID scaler 标准化 SIMD。

### 张量

| 名称 | 形状 |
| --- | --- |
| COVID 输入 | `[B, L, N, 1]`，默认 `L = 7` |
| 目标 | `[B, N, 1]` |
| 预测 μ / variance / σ | `[B, N, 1]` |

$N$ = `len(node_order)`。爱丁堡数据应满足 $N = 111$，否则校验失败。

---

## 2. 静态上下文：六个 SIMD 变量

### 数据

`static_features.csv` 与 npz 中的 `X_static_raw`，形状 `[N, 6]`。

| 内部列名 | GeoShapley player 名 |
| --- | --- |
| `income_rate` | `income_deprivation` |
| `employment_rate` | `employment_deprivation` |
| `university_rate` | `higher_education` |
| `overcrowded_rate` | `overcrowding` |
| `crime_rate` | `crime` |
| `pt_gp_min` | `public_transport_time_to_gp` |

不发明人口、年龄、医院、pharmacy 等未校验变量。接口保持可扩展。

### 标准化

跨研究区 IZ，不跨日期。在训练配置上拟合一次后冻结：

$$
\tilde X_{i,f} = \frac{X_{i,f} - \bar X_f}{\mathrm{std}_f}
$$

若 $\mathrm{std}_f \approx 0$：该列映成 0，保留列，记 warning，禁止除零。

GeoShapley 联盟只在原始单位替换目标 IZ 的值，然后套用这套冻结均值/标准差。禁止在联盟中重算 scaler。

保存：特征名与顺序、均值、标准差、零方差处理、缺失处理、来源年、规范哈希。

---

## 3. 位置：投影质心

### 数据

优先：`data/results/graph/road/nodes.csv` 的 `Easting`, `Northing`。

备用：`covid._load_edinburgh_centroids`。

CRS：原始与投影均为 **EPSG:27700**（米）。不再从经纬度转换。

### 用法

$$
L \in \mathbb{R}^{N \times 2}, \qquad
\tilde L_{i,c} = \frac{L_{i,c} - \bar L_c}{\mathrm{std}_c}
$$

坐标 scaler 在训练配置上拟合后冻结。GeoShapley 不重算。

坐标不是第 7、8 个社会经济变量。它们只用于：

1. 局部 MLP 输入
2. GeoShapley 的一个联合 location player

GraphConv 不吃坐标。模型若没有坐标输入，禁止输出 location 贡献。

保存：原始 CRS、投影 CRS、变换说明、坐标 scaler、规范哈希。

---

## 4. 规范节点顺序与哈希

COVID `node_index` 是权威顺序。下列全部必须同一 IZ 序列：

COVID 张量、上下文、坐标、三张邻接、embedding、预报、不确定度、GeoShapley、地图表。

校验：节点数、非空唯一 IZ、唯一连续整数 `node_index`（0 到 $N-1$）、有序 IZ 序列、规范哈希。

### 规范哈希（所有产物统一）

```python
payload = "\n".join(ordered_iz_codes).encode("utf-8")
canonical_node_order_hash = sha256(payload).hexdigest()
```

爱丁堡现有 mobility 报告中的 `8f625000ca42af45709b4e887a429c93971443f30f2fbddbe07863342ca16d34` 即此算法。

### Legacy 哈希

`forecast.py` 曾用 `"|".join`，值为 `7b08ceb2…`。处理：

1. 比较完整 IZ 序列
2. 序列一致则重算规范哈希并保存
3. 旧哈希只作 `legacy_node_order_hash`
4. 之后只拿规范哈希做校验

序列、节点数或哈希不一致：停止，返回 failed，指出源，禁止静默重排。按 IZ 码对齐只允许在有记录的 harmonisation 函数中进行，并重算哈希。

$N$ 由 `node_order` 行数决定。111 只是爱丁堡期望值。

---

## 5. 三张图

### 数据（原始、未归一、不覆盖写回）

| 键 | 文件 | 含义 |
| --- | --- | --- |
| `geo` | `data/results/graph/geo/adjacency_geo.npz` | 无向 rook，0/1 |
| `transport` | `data/results/graph/road/adjacency_road.npz` | 有向多方式最短路权重 |
| `mobility` | `data/results/graph/mobility/adjacency_mobility.npz` | 有向静态平均 OD |

约定：$A[i,j] > 0$ 表示边 $i \rightarrow j$。对角为 0。geo 可对称。transport 与 mobility 保持方向，不对称化，mobility 不做 kNN，三张图不预融合。

### 校验

形状 `[N, N]`、规范哈希、有限、非负、对角 0、边数、孤立点、零入度、零出度、生成报告状态与结构化 warning code。

### 图集合

- 三图：`graph_set = [geo, transport, mobility]`
- 两图：`graph_set = [geo, transport]`

mobility 致命失败时，不得在推理时从三图 checkpoint 删掉 mobility。必须加载**单独训练并校验过的**两图 checkpoint；没有则停止并要求两图训练。回退必须写入 provenance。

---

## 6. 有向扩散 support

三张图尺度不同，禁止 $\sum_k \alpha_k A_k$。

出度、入度：

$$
d^{\mathrm{out}}_i = \sum_j A_{ij}, \qquad
d^{\mathrm{in}}_i = \sum_j A_{ji}
$$

$$
T^{\mathrm{fwd}}_{ij} =
\begin{cases}
A_{ij} / d^{\mathrm{out}}_i & d^{\mathrm{out}}_i > 0 \\
0 & \text{otherwise}
\end{cases}
\qquad
T^{\mathrm{bwd}}_{ij} =
\begin{cases}
A^\top_{ij} / d^{\mathrm{in}}_i & d^{\mathrm{in}}_i > 0 \\
0 & \text{otherwise}
\end{cases}
$$

$T$ 是**行随机**（有效行和为 1）。特征为 `[N, F]`、左乘时：

$$
S^{\mathrm{fwd}} = (T^{\mathrm{fwd}})^\top, \qquad
S^{\mathrm{bwd}} = (T^{\mathrm{bwd}})^\top, \qquad
\mathrm{propagated} = S X
$$

因此：

- 有效行：$T$ 的行和为 1
- 对应列：$S$ 的列和为 1
- **不要**要求 $S$ 的行和为 1

零出度节点产生：

- $T^{\mathrm{fwd}}$ 的一行全 0
- $S^{\mathrm{fwd}}$ 的一列全 0

校验：

```python
T.sum(axis=1)   # 有效行 ≈ 1，零出度行 = 0
S.sum(axis=0)   # 有效列 ≈ 1，零出度对应列 = 0
```

节点 $j$ 从指向它的 $i$ 接收特征。反向沿 $A^\top$。用小的不对称图单测方向。

零出度：安全除法，对应 $T$ 行 / $S$ 列保持 0。禁止 NaN/Inf，禁止改成均匀分布。

融合后检查的是 **$S_{\mathrm{fused}}$ 的列和**。某图对应列为 0 时，融合列和可以小于 1。应报告，不要重归一。不要写成“融合 support 行和小于 1”。

---

## 7. 一套全局 α

对当前 `graph_set` 中每张图一个 logit：

$$
\alpha = \mathrm{softmax}(\theta), \qquad
\alpha_k > 0, \quad \sum_k \alpha_k = 1
$$

三图时 $\alpha = [\alpha_{\mathrm{geo}}, \alpha_{\mathrm{transport}}, \alpha_{\mathrm{mobility}}]$。

$$
S^{\mathrm{fwd}}_{\mathrm{fused}} = \sum_k \alpha_k S^{\mathrm{fwd}}_k
\qquad
S^{\mathrm{bwd}}_{\mathrm{fused}} = \sum_k \alpha_k S^{\mathrm{bwd}}_k
$$

同一 α 用于上下文嵌入和 DCRNN。禁止：

- 两套 α
- 节点/日期/horizon 专属 α
- 时间 β
- 正反向平均
- 融合后再按行或按列重归一
- 推理时丢掉一张图后重归一剩下的 α

α 的含义：训练目标下对各图扩散结构的相对预测效用。不是因果、不是精度百分比、不是真实传播机制。

---

## 8. 上下文节点嵌入

### 结构

$E = 8$，2 层，dropout 可配。

$$
\begin{aligned}
Z_{\mathrm{local}} &= \mathrm{MLP}([\tilde X \| \tilde L]) \\
Z_{\mathrm{fwd}} &= \mathrm{GraphConv}(\tilde X, S^{\mathrm{fwd}}_{\mathrm{fused}}) \\
Z_{\mathrm{bwd}} &= \mathrm{GraphConv}(\tilde X, S^{\mathrm{bwd}}_{\mathrm{fused}}) \\
Z_{\mathrm{graph}} &= W_g [Z_{\mathrm{fwd}} \| Z_{\mathrm{bwd}}] + b_g \\
Z &= \mathrm{LayerNorm}(Z_{\mathrm{local}} + Z_{\mathrm{graph}})
\end{aligned}
$$

$Z \in \mathbb{R}^{N \times 8}$，对日期静态。

### GraphConv

$$
Z = \phi(S (X W) + b)
$$

等价写法：$Z = \phi(S X W + b)$，即先 `X @ W` 再 `S @ ·`。

| 矩阵 | 形状 |
| --- | --- |
| $S$ | `[N, N]` |
| $X$ | `[N, F]` |
| $W$ | `[F, E]` |
| $Z$ | `[N, E]` |

禁止 `S @ W @ X`。$\phi$ 为非线性（如 ReLU）加 dropout。

坐标只进入 $Z_{\mathrm{local}}$。GraphConv 只吃六个上下文变量。

### 拼进时序

$$
Z_{\mathrm{rep}} \in \mathbb{R}^{B \times 7 \times N \times 8}, \qquad
X_{\mathrm{model}} = [X_{\mathrm{covid}} \| Z_{\mathrm{rep}}] \in \mathbb{R}^{B \times 7 \times N \times 9}
$$

重复只为对齐张量，不把 embedding 说成动态变量，不把各维解释成原始变量。

### 诊断

各维均值与标准差、近常数维数、有限性、节点变异、规范哈希。对照：完整模型 vs 将 $Z$ 置 0。第一版不加辅助 embedding 损失，除非诊断显示塌缩。

---

## 9. DCRNN 时间编码器

Encoder-only。2 层 DCGRU，扩散阶 $K = 2$，隐维 $R$ 可配（默认 64）。无 H 步 decoder，无 teacher forcing。

输入：$X_{\mathrm{model}}$ 与同一对 $S^{\mathrm{fwd/bwd}}_{\mathrm{fused}}$。两个 support 必须分开传入，禁止平均成一张。

对 $\tau = 1, \ldots, 7$：

$$
H_\tau = \mathrm{DCGRU}(X_\tau, H_{\tau-1}; S^{\mathrm{fwd}}_{\mathrm{fused}}, S^{\mathrm{bwd}}_{\mathrm{fused}})
$$

扩散与 DCRNN 一致：对每个 support 做 $K$ 阶递推（Chebyshev：$X^{(k+1)} = 2 S X^{(k)} - X^{(k-1)}$），与 $X^{(0)}$ 拼接后线性映射，再进 GRU 门。

末状态：

$$
H_{\mathrm{final}} \in \mathbb{R}^{B \times N \times R}
$$

直接用于预报 $Y_{t+7}$。

不搬旧代码：无条件对称化、先融合原始 A、只留一个 support、平均正反向、固定城市路径、H 步输出、时间 β。

---

## 10. 一元概率头与损失

UQGNN 的多元 $(\boldsymbol\mu, \Sigma)$ 在此退化为 1×1 方差。名称：UQGNN-inspired univariate probabilistic prediction head。不声称实现了多元 UQGNN。

$$
\mu_z = g_\mu(H_{\mathrm{final}}), \qquad
v_z = \mathrm{softplus}(g_v(H_{\mathrm{final}})) + 10^{-6}, \qquad
\sigma_z = \sqrt{v_z}
$$

形状均为 `[B, N, 1]`。$v_z > 0$，$\sigma_z > 0$，有限。主不确定度不是 MC Dropout，不做认知/偶然分解。

### 损失（标准化 COVID 空间）

对有效目标：

$$
\mathrm{NLL} = \frac{1}{2}\left[\log(2\pi) + \log v_z + \frac{(y_z - \mu_z)^2}{v_z}\right]
$$

$$
\mathcal{L} = \mathrm{mean}\{\mathrm{NLL} : \text{valid cells}\}
$$

缺失目标不进损失。不得用 MSE 冒充概率训练。第一版不加任意不确定度正则。可用 weight decay、dropout、梯度裁剪。

用 **validation_selection** 的 NLL 做 early stopping 和 checkpoint 选择。不要用 validation_calibration 或 test 更新参数。

联合更新：上下文 encoder、$\theta/\alpha$、DCRNN、两个 head。

---

## 11. 反变换、区间、校准、旗标

### 反变换（按 IZ，冻结的训练 COVID scaler）

$$
\begin{aligned}
\mu_i &= \mu_{z,i}\, s_i + \mu^{\mathrm{covid}}_i \\
v_i &= v_{z,i}\, s_i^2 \\
\sigma_i &= \sigma_{z,i}\, |s_i|
\end{aligned}
$$

mean 不加到 $v$ 或 $\sigma$。检查 $\sigma_i^2 \approx v_i$。

### 未校准高斯区间（评估用未截断 raw）

80%：

$$
[\mu - 1.2815515655\,\sigma,\ \mu + 1.2815515655\,\sigma]
$$

95%：

$$
[\mu - 1.9599639845\,\sigma,\ \mu + 1.9599639845\,\sigma]
$$

禁止把 $\mu \pm \sigma$ 当成 95% 区间。覆盖率评估必须用未截断区间。

### 有限样本修正的经验校准（只用 validation_calibration）

名称：**finite-sample corrected empirical calibration**。

不要称为 finite-sample calibration guarantee，也不要声称具有 conformal prediction 的正式覆盖保证。

滚动七日 COVID 结果在时间上重叠，在空间上相关，校准样本不满足经典 conformal 的交换性假设。因此校准区间是**经验不确定区间**，不是基于交换性的覆盖保证区间。

Because rolling COVID-19 outcomes are temporally overlapping and spatially dependent, the calibrated intervals are empirical uncertainty intervals rather than intervals with a formal exchangeability-based coverage guarantee.

Checkpoint 冻结之后，在 calibration 子集上计算：

$$
\mathrm{score}_i = \frac{|y_i - \mu_i|}{\sigma_i}
$$

目标未覆盖率 $\gamma$。95% 区间取 $\gamma = 0.05$。

设最低校准样本量 $n_{\min}$（默认 20，使得 $\gamma = 0.05$ 时 $k \le n$ 无需截断）。

若 $n_{\mathrm{calibration}} < n_{\min}$：

- `calibration_status = unavailable`
- 不计算 $q_{95}$，不输出校准区间，不输出基于校准集的 P90 阈值
- 仍可输出 raw 80%/95% 区间
- 记 warning code，级别 `review_required`
- **禁止**把 $k$ 截断到 $[1, n]$ 后继续当作有效校准

若 $n_{\mathrm{calibration}} \ge n_{\min}$：

$$
k = \lceil (n_{\mathrm{calibration}} + 1)(1 - \gamma) \rceil
$$

要求 $1 \le k \le n_{\mathrm{calibration}}$。将 score 升序排序，取 1-based 第 $k$ 个顺序统计量：

$$
q_{95} = \mathrm{score}_{(k)}
$$

校准区间：

$$
[\mu - q_{95}\sigma,\ \mu + q_{95}\sigma]
$$

不要把 $q_{95}$ 除以 1.96，除非另存等价量且区间明确写成 $\mu \pm 1.96\, c\, \sigma$。实现默认只存并使用 $q_{95}$。

必须记录：`calibration_status`、`n_calibration`、`n_min`、`gamma`、`k`、`q95`（若 available）、校准日期范围、checkpoint 标识、方法名称 `finite-sample corrected empirical calibration`。

test 与未来只用冻结的校准产物。test 不参与拟合。calibration 数据不得回传更新模型参数。

### 展示（仅地图）

地图区间必须绑定明确来源，禁止 Agent 或网页自行混用 raw 80%、raw 95% 与校准区间。

```text
if calibration_status == available:
    source_interval = calibrated 95% interval
    display_interval_type = calibrated_95
else:
    source_interval = raw 95% Gaussian interval
    display_interval_type = raw_gaussian_95
```

然后只对展示做非负截断：

$$
\begin{aligned}
\mathrm{display\_mean} &= \max(0, \mu) \\
\mathrm{display\_lower} &= \max(0, \mathrm{source\_lower}) \\
\mathrm{display\_upper} &= \max(0, \mathrm{source\_upper})
\end{aligned}
$$

记录 `display_interval_type`、`mean_clipped`、`lower_bound_clipped`。覆盖率与校准诊断必须用未截断的 raw/校准区间，不用 display 区间。raw 80% 只作诊断，不作默认地图区间。

### 高不确定旗标（只用 validation_calibration，且校准必须 available）

$$
\tau = \mathrm{P}_{90}(\{\sigma_i\}_{\mathrm{validation\_calibration}})
$$

仅当 `calibration_status = available` 时计算并冻结 $\tau$。$\sigma > \tau$ 为 high。test/未来用冻结阈值。不是临床分级。校准 unavailable 时不输出该旗标。

### 含义

- $v$：$Y_{t+7}$ 的模型条件方差
- $\sigma$：该滚动七日率的预测标准差
- 大 $\sigma$：条件预测分布宽
- 不是已实现误差、不是“预报错误的概率”、不是参数不确定、不是因果不确定

---

## 12. 测试期评估

只用 S1 测试集。因只有固定 $H = 7$，不报“按 horizon”指标。

点指标：MAE、RMSE、bias、可选 $R^2$。

概率指标：Gaussian NLL；raw 80% / raw 95% 覆盖与平均宽度；若校准 available，再报经验校准区间覆盖；强度分箱覆盖；$\sigma$ 与 $|e|$ 关系（诊断）。校准覆盖率是经验诊断，不是正式覆盖保证。

竞赛不是精度排行榜。这些是有效性与可靠性检查。

历史残差图与未来 σ 图必须分开标注。

---

## 13. GeoShapley（target_iz_local）

采用完整 GeoShapley（Li, 2024），不是“六个变量 + 联合位置”的普通 Shapley。若实现不含 location–feature 交互，不得称为 GeoShapley，只能称为 GeoShapley-inspired joint-location Shapley。本项目选择前者。

### 目标

$$
f_i = \mu_i \quad \text{（原尺度均值，目标 IZ } i \text{，日期 } t+7 \text{）}
$$

解释原始变量，不解释 embedding 维。GeoShapley 解释的是某个选定日期上的预测。SIMD 特征值跨日期不变，但其贡献可随当日 COVID 历史输入变化。`dates: last` 表示所选测试集最后一个回顾性 issue date，不是业务上的未来预报。

### 分解

$$
f_i = \phi_0 + \phi_{\mathrm{location}} + \sum_{j=1}^{6} \phi_j + \sum_{j=1}^{6} \phi_{\mathrm{location},j}
$$

必须同时输出：

- $\phi_0$：baseline（全部 player 取参考值）
- $\phi_j$：每个原始变量的主效应
- $\phi_{\mathrm{location}}$：位置本身的主效应
- $\phi_{\mathrm{location},j}$：位置与每个原始变量的交互

Easting 与 Northing 仍作为**一个**联合 location player，不拆成两个坐标效应。

`explanation_scope = "target_iz_local"`。

### Players（联盟指示）

联盟空间仍是 6 个特征指示加上 1 个联合 location 指示，共 $2^7 = 128$ 个联盟。这 128 个函数值用来估计上面的 14 个 GeoShapley 分量（$\phi_0$、$\phi_{\mathrm{location}}$、6 个 $\phi_j$、6 个 $\phi_{\mathrm{location},j}$）。

| `player_name` | `component` |
| --- | --- |
| `income_deprivation` | 主效应 $\phi_j$ |
| `employment_deprivation` | 主效应 $\phi_j$ |
| `higher_education` | 主效应 $\phi_j$ |
| `overcrowding` | 主效应 $\phi_j$ |
| `crime` | 主效应 $\phi_j$ |
| `public_transport_time_to_gp` | 主效应 $\phi_j$ |
| `location` | 位置主效应 $\phi_{\mathrm{location}}$ |
| `location_x_income_deprivation` | 交互 $\phi_{\mathrm{location},j}$ |
| `location_x_employment_deprivation` | 交互 |
| `location_x_higher_education` | 交互 |
| `location_x_overcrowding` | 交互 |
| `location_x_crime` | 交互 |
| `location_x_public_transport_time_to_gp` | 交互 |

禁止输出 `embedding_*`。

### 参考值

目标 IZ 的六变量换成研究区中位数；location 换成质心中位数。不用预测值当参考。邻居始终保持观测值。

### 联盟构造

对目标 IZ $i$ 与 player $j$：

- 若 $j \in S$：用该 IZ 的**观测原始值**
- 否则：用文档化的**参考值**

上下文特征 $f$：

$$
X^{(S)}_{i,f} =
\begin{cases}
X^{\mathrm{obs}}_{i,f} & f \in S \\
\mathrm{reference}[f] & \text{otherwise}
\end{cases}
$$

联合 location：

$$
L^{(S)}_i =
\begin{cases}
L^{\mathrm{obs}}_i & \text{location} \in S \\
L^{\mathrm{reference}} & \text{otherwise}
\end{cases}
$$

其余 IZ 行保持观测值、不做改动。

每个联盟从**新拷贝**的完整矩阵开始。禁止在上一个联盟的状态上原地改完再接着用。

### 冻结预处理

每个联盟必须使用已经拟合并冻结的：

- 上下文均值与标准差
- 坐标均值与标准差
- COVID scaler

禁止在替换某个 player 之后重算任何 scaler。

流程：

```text
在原始单位构造联盟拷贝
  -> 套用冻结 scaler
  -> 重算完整 embedding
  -> 重跑预报
  -> 取出目标 IZ 的 mu
```

三张图与模型参数固定。禁止直接扰动 embedding 维。禁止把一个变量在全部 IZ 上一起改作为主解释。字段级分析若以后做，必须另表 `spatial_field_grouped`。

### 估计：Shapley 核加权最小二乘

对每个联盟 $S$，令

- $z_{\mathrm{loc}} = 1$ 若 location 在 $S$ 中，否则 0
- $z_j = 1$ 若特征 $j$ 在 $S$ 中，否则 0

回归：

$$
f_i(S) = \phi_0 + \phi_{\mathrm{location}} z_{\mathrm{loc}} + \sum_{j=1}^{6} \phi_j z_j + \sum_{j=1}^{6} \phi_{\mathrm{location},j}\, z_{\mathrm{loc}} z_j + \varepsilon_S
$$

权重为 Shapley 核。对 $n = 7$ 个指示变量、联盟大小 $s = |S|$：

$$
w(S) = \frac{n-1}{\binom{n}{s}\, s\, (n-s)}, \qquad 0 < s < n
$$

空联盟与全集**不要**靠“足够大的权重”。对其使用精确等式约束：

$$
\phi_0 = f_i(\varnothing)
$$

$$
\phi_0 + \phi_{\mathrm{location}} + \sum_{j=1}^{6} \phi_j + \sum_{j=1}^{6} \phi_{\mathrm{location},j} = f_i(\mathrm{observed})
$$

实现上可用带等式约束的加权最小二乘（或等价的约束消去）。若改用官方 GeoShapley 软件，则遵循其端点约束方式，并在 provenance 中记录实现名称与约束设定。

128 个联盟 × 14 列设计矩阵（截距、location、6 个主效应、6 个 location–feature 交互）与论文框架一致。特征–特征交互不单独输出。

可加性在约束下应对全集严格成立。`additivity_error` 只记录浮点残差。超过数值容差（默认 `1e-6`）则 `review_required`。

### Location 解释

仅当坐标是模型输入时才报告 location 主效应与全部 `location_x_*` 交互。没有坐标则既不报 `location`，也不报交互项。

`location` 标签：

> Residual spatial/location contribution conditional on the fixed graph structure.

`location_x_<feature>` 标签：

> Location–feature interaction for the named original variable, conditional on the fixed graph structure.

二者都不是纯地理效应、不是因果、不是邻接矩阵贡献。location 及交互可能与三张图已编码的空间信息重叠。

---

## 14. 结构化警告

Agent 必须读 warning code，不能只看 `status=ok_with_warnings`。

| 级别 | 含义 |
| --- | --- |
| `accepted_limitation` | 已知数据限制，可继续，必须披露 |
| `review_required` | 需人工确认后才能把该图/解释当作无保留结果 |
| `critical_failure` | 停止，不得静默继续 |

具体规则：

| 情况 | 级别 | 规则 |
| --- | --- | --- |
| 规范节点顺序中缺少任何一个目标 IZ | `critical_failure` | 缺一即停 |
| IZ 代码无法与规范顺序一对一匹配 | `critical_failure` | 含重复、缺失、多余、无法对齐 |
| 非法图权重（负、NaN、Inf） | `critical_failure` | 缺一即停 |
| mobility 为 2019–2023 预平均 OD | `accepted_limitation` | 必须披露，不因此停训 |
| 排除研究区外 OD 对（已记录的区外 IZ） | `accepted_limitation` | 披露排除名单即可 |
| OD 表缺少部分 IZ–IZ 组合 | `accepted_limitation` | 允许。真实 OD 稀疏，**不要求**完整 $N \times N$ 有向边 |
| 某 IZ 在规范顺序中存在，但 mobility 入边和出边都为 0 | `review_required` | 记录孤立节点数与 IZ 代码，供人工检查；不自动删点 |
| 零方差上下文列映成 0 | `accepted_limitation` | 保留列 |
| 融合 support **列和**小于 1 | `accepted_limitation` | 由零出度列导致，不重归一 |
| GeoShapley 可加性超过数值容差 | `review_required` | 默认容差 `1e-6` |
| 校准样本量低于 $n_{\min}$ | `review_required` | `calibration_status=unavailable`，不截断 $k$ |

---

## 15. Checkpoint、校准产物与图回退

训练 checkpoint 与校准产物必须分开。

### 训练 checkpoint（参数冻结后不再写入校准量）

必须保存：

- 模型参数
- `graph_set` 与图顺序
- 各图规范哈希
- α 配置与维数
- 模型配置、随机种子、`config_id`（L、H、stride）
- 节点顺序与规范哈希
- COVID / 上下文 / 坐标 scaler
- 选中的 epoch
- 优化器状态（可选）
- 用于选择该 checkpoint 的 validation_selection NLL

不要把 $q_{95}$ 或 P90 阈值写进训练 checkpoint。

### 校准产物（checkpoint 冻结之后单独生成）

必须保存：

- checkpoint 标识或 checksum
- `calibration_status`（`available` 或 `unavailable`）
- `n_calibration`、`n_min`、`gamma`
- 若 available：$q_{95}$、$k$、P90 σ 阈值
- 若 unavailable：不写 $q_{95}$ 或阈值，并说明原因
- validation_calibration 日期范围
- 方法名称：`finite-sample corrected empirical calibration`
- 交换性限制说明（时间重叠与空间相关）

推理必须核对该校准产物属于当前加载的 checkpoint。不匹配则失败。

### 图集合兼容

| Checkpoint | 只能用于 |
| --- | --- |
| 三图 | `[geo, transport, mobility]` |
| 两图 | `[geo, transport]` |

禁止：

- 推理时从三图 checkpoint 去掉 mobility
- 推理时重归一剩下的 α
- 静默改 `graph_set`
- 用 `target_offset_days` ≠ 7 的 checkpoint 做 offset=7 的单点推理

mobility 失败：加载已训练的两图 checkpoint；否则停止并请求两图训练。

---

## 16. 输出与 Agent

### 预报表

每个 issue date × IZ 一行，不是每个 IZ 七行。未来推理无观测值。

字段：`input_start_date`, `issue_date`, `target_report_date`, `target_offset_days`, `iz_code`, `node_index`, `predicted_mu_z`, `predicted_variance_z`, `predicted_sigma_z`, `predicted_mu_original`, `predicted_variance_original`, `predicted_sigma_original`, `raw80_lower`, `raw80_upper`, `raw95_lower`, `raw95_upper`, `calibrated_lower`, `calibrated_upper`, `calibration_status`, `display_mean`, `display_lower`, `display_upper`, `display_interval_type`, `mean_clipped`, `lower_bound_clipped`, `uncertainty_flag`, `uncertainty_threshold`, `model_checkpoint`, `calibration_artefact`, `node_order_hash`

`display_interval_type` 只能是 `calibrated_95` 或 `raw_gaussian_95`。`calibration_status=unavailable` 时，`calibrated_*`、`uncertainty_flag`、`uncertainty_threshold` 为空，且 `display_interval_type=raw_gaussian_95`。

### 其他表

- 图融合：α、`graph_set`、校验状态、fallback、checkpoint、哈希
- Embedding：IZ、node_index、8 维、诊断、哈希（诊断用，不作 GeoShapley player）
- GeoShapley：每个目标 IZ × 每个分量一行（6 个主效应 + `location` + 6 个 `location_x_*`）；含 `component`（`main` / `location` / `interaction`）、`phi_0`、`reconstructed_prediction`、`additivity_error`；`explanation_scope=target_iz_local`
- 测试：观测、μ、v、σ、残差、AE、raw80/raw95/校准覆盖、旗标
- Provenance：数据版本、图文件、哈希、特征、CRS、S1 外层划分、validation 内部划分日期与样本数、scaler、配置、L/H、checkpoint、校准产物、α、不确定定义、fallback、警告码

地图表用 IZ 码连接边界。σ 图与残差图分开。GP 可达性来自 `pt_gp_min`。

### Agent 工具

均返回 `status`, `outputs`, `warnings`, `provenance`。

1. `validate_inputs()`
2. `load_temporal_dataset()`
3. `build_graph_supports()`
4. `train_model()`
5. `load_checkpoint()`
6. `forecast_single_target()`
7. `evaluate_test_period()`
8. `explain_target_iz_with_geoshapley()`
9. `export_map_ready_results()`

顺序：加载 S1 → 校验节点与三图 → 校验六变量与坐标 → 披露警告码 → 加载兼容 H=7 且 graph_set 匹配的 checkpoint **以及**匹配的校准产物，或请求训练 → 每个 IZ 一个 $t+7$ 分布 → 可选 GeoShapley → 导出地图表。

---

## 17. 默认超参

写入 `configs/model.yaml`，不是层内魔法数。

| 项 | 默认 |
| --- | --- |
| embedding $E$ | 8 |
| 上下文层数 | 2 |
| DCRNN 层数 | 2 |
| 扩散阶 $K$ | 2 |
| 隐维 $R$ | 64 |
| dropout | 0.1 |
| variance $\varepsilon$ | `1e-6` |
| raw 80% | $\pm 1.2815515655\,\sigma$ |
| raw 95% | $\pm 1.9599639845\,\sigma$ |
| 校准 | finite-sample corrected empirical calibration；$q_{95}$，γ = 0.05；$n_{\min}=20$；仅 validation_calibration |
| 不确定阈值 | `uncertainty_flag`：仅当校准 available 时，validation_calibration 上 σ 的配置分位（默认 0.90） |
| validation 内部切分 | 按 target_date 时间顺序 50/50 |
| GeoShapley 参考 | 研究区中位数 |
| location CRS | EPSG:27700 |

---

## 18. 代码布局（实现时）

新建 `src/model/`：

| 文件 | 板块 |
| --- | --- |
| `node_order.py` | 第 4 节 |
| `dataset.py` | 第 1 节 |
| `graphs.py` | 第 5、14 节 |
| `diffusion.py` | 第 6、7 节 |
| `context.py` | 第 2、3、8 节 |
| `dcrnn_encoder.py` | 第 9 节 |
| `heads.py` | 第 10、11 节 |
| `train.py` | 训练 checkpoint |
| `evaluate.py` | 第 11、12 节与校准产物 |
| `geoshapley.py` | 第 13 节 |
| `export.py` / `tools.py` | 第 16 节 |

修改：`src/agent.py`、`pyproject.toml`、`configs/model.yaml`。

不改：`src/forecast.py` 的窗口生成、`src/graph/*.py` 的图构建、`data/raw`。

---

## 19. 必测项

数据：S1 只加载不重建；目标 `[B, N, 1]`；target 为 $t+7$；外层分区 target 不重叠；lookback 可跨前一段；节点顺序失败即停；COVID scaler 仅训练期；上下文 scaler 跨 IZ；零方差映 0。

验证内部：selection/calibration 按时间切分；calibration 不更新参数；provenance 记录日期与样本数。

图：geo 可对称；transport/mobility 保持有向；正反向方向；$T$ 行和与 $S$ 列和；$T$ 零行对应 $S$ 零列；无 NaN/Inf；α 为正且和为 1；一套 α；正反向不合并；融合不重归一；fallback 显式且 checkpoint 兼容。

嵌入：`[N, 8]`；局部与图路径；GraphConv 维数 $S(XW)$；有限；塌缩诊断；零 embedding 对照。

概率：μ/v/σ 形状；正且有限；$\sigma^2 = v$；手算 NLL；mask；μ 反变换加 mean；$v$ 乘 $s^2$；σ 乘 $|s|$；mean 不加到 v/σ；经验 $q_{95}$ 顺序统计量；$n < n_{\min}$ 时 calibration unavailable 且不截断 $k$；raw 80% 与 raw 95%；阈值仅来自 available 的 validation_calibration；无 MC Dropout 分解；不得把校准称为 coverage guarantee。

哈希与 N：规范哈希相等；legacy 迁移；$N$ 来自 `node_order`。

Checkpoint：图集合不匹配失败；α 维不匹配失败；缺两图 checkpoint 时不得删 mobility；校准产物必须匹配 checkpoint。

GeoShapley：输出 $\phi_0$、六个 $\phi_j$、$\phi_{\mathrm{location}}$、六个 $\phi_{\mathrm{location},j}$；空联盟与全集用等式约束而非大权重；$\phi_0=f_i(\varnothing)$ 且分解严格重构观测预测；不输出 embedding 维；联盟成员用观测值；缺席成员用参考值；其他 IZ 行不变；每个联盟新拷贝；全程冻结 scaler；重算整图 embedding；一个 $\mu_{t+7}$；`target_iz_local`；无坐标则无 location 也无交互；不与字段级混表；不得把无交互的 7-player Shapley 称为 GeoShapley。

输出：每个 issue × IZ 一行；无 H 步行；地图用 IZ 码；`display_interval_type` 为 `calibrated_95` 或 `raw_gaussian_95`；未来无观测；残差与 σ 分开；缺少目标 IZ 或节点码无法对齐为 `critical_failure`；OD 稀疏不为失败；mobility 零入出度为 `review_required`。

---

## 20. 对外解释口径

1. 共享 α 是原型参数共享，不是机制发现。
2. α 是预测效用，不是因果。
3. 上下文效应含图传播；GeoShapley 仍解释原始变量，因为扰动后会重算 embedding。
4. GeoShapley 输出变量主效应、位置主效应、以及位置–变量交互；都是模型解释，不是因果。
5. location 及 `location_x_*` 与固定图结构可能重叠。
6. mobility 是 2019–2023 静态平均 OD，不是疫情实时流动。
7. 高斯不确定是初始近似；覆盖率看未截断 raw 80%/95% 区间。$q_{95}$ 是 finite-sample corrected empirical calibration，因时间重叠与空间相关，不是交换性假设下的正式覆盖保证。样本不足则校准 unavailable。
