[English](README.md) | **中文**

<div align="center">

# Marketing Attribution & Budget Optimization

**从宏观 MMM 到微观多触点归因，再到预算优化的不确定性量化**

*MMM · 多触点归因 · 预算优化 · block bootstrap 置信区间*

<a href="https://github.com/MeaFew/attributor/actions"><img src="https://github.com/MeaFew/attributor/workflows/CI/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff">
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">

**中文** | <a href="./README.md">English</a>

</div>

---

## 核心结论

> **预算优化的「0% 提升」不是无解，而是诚实结论**。饱和响应模型表明当前分配已近局部最优（revenue 提升点估计 ≈ **0.0%**）；但 block bootstrap 揭示真正的决策风险——95% 置信区间是朴素逐行重抽样的 **8.9 倍宽**，单点估计会让决策者基于虚假精确感砍预算。本项目不止输出「该花多少」，更回答「这个数字有多可信」。

<p align="center">
  <img src="reports/images/uncertainty_summary.svg" width="900" alt="Attributor block bootstrap 与朴素 bootstrap 不确定性对比">
</p>

<div align="center">

**每渠道最优预算的 95% 置信区间（单品牌 913 行 · block_size=7 天 · N=200）**

| 渠道 | 点估计 | block 95% CI | block/naive 宽度比 |
|------|-------:|--------------|:------------------:|
| **meta_facebook** | \$634,590 | [\$632,850, \$635,031] | **9.1x** |
| **meta_instagram** | \$240,627 | [\$238,886, \$241,068] | **9.1x** |
| **google_pmax** | \$80,043 | [\$78,174, \$80,413] | **9.3x** |
| **google_video** | \$17,542 | [\$16,374, \$18,335] | **7.4x** |
| google_paid_search | \$19,276 | [\$18,703, \$23,764] | 14.6x |
| **google_display** ⚠️ | \$7,061 | [\$706, \$7,499] | **17.8x** |
| **google_shopping** ⚠️ | \$2,438 | [\$244, \$7,320] | 3.1x |
| meta_other | \$531 | [\$52, \$1,561] | 1.0x |

> ⚠️ = 置信区间逼近 0、系数符号不稳定的渠道——**现有数据不支持对其下强结论**。

**Ridge MMM holdout R² = 0.42**（OLS in-sample 0.54 / holdout 0.44），Revenue 提升 CI = [0.00%, 0.08%]。

<details>
<summary><b>📊 查看预算置信区间森林图（block bootstrap 的代表性成果图）</b></summary>

<img src="reports/images/budget_ci.png" alt="预算置信区间森林图：block vs naive 95% CI 对比" width="820">

</details>

</div>

> **为什么是 8.9 倍**：MMM 残差强自相关（OLS Durbin-Watson = **0.90**，无自相关应为 2.0）。朴素 case-resample 逐行重抽样「假装样本独立」，系统性低估不确定性、CI 偏窄，给决策者虚假精确感。block bootstrap 把时间序列切成 7 天整块、有放回重抽整块并保持块内时序，保留自相关结构，得到诚实（更宽）的区间。`tests/test_uncertainty.py::TestBlockVsNaive` 直接证明：在 AR(1)=0.85 序列上，naive 重抽样把 lag-1 自相关从 0.85 破坏到 ≈0，block 重抽样保留 ≈0.7。

<details>
<summary><b>🔧 核心模块详解（5 个模块的完整方法、结果表与诚实解读）</b></summary>

### 1. 数据预处理（`src/attributor/preprocess.py`）

```
输入: 132,759 rows x 50 cols（含大量空值与千分位逗号分隔符）
输出: 清洗后的 Parquet (日粒度)
关键操作:
  - 千分位逗号去除 + Float64 强制转换（解决 Polars 自动推断为 String 的问题）
  - CTR、CPM、ROAS 衍生指标计算
  - Adstock 衰减特征构造: x_t + 0.5*x_{t-1} + 0.25*x_{t-3} + 0.125*x_{t-7}
  - 时间特征提取（year/month/day_of_week/is_weekend）
```

### 2. 营销组合建模（`src/attributor/mmm_model.py`）

#### 防泄漏与正则化说明（重要）

- **按时间切分而非随机切分**：MMM 是日粒度时间序列，早期版本在全量数据上拟合并报 R²，本质是重代入（in-sample）拟合。现按日期留出尾部 20% 作为 holdout，同时报 in-sample R² 与 holdout R²/MAE，二者之差即过拟合程度。
- **Ridge/Lasso 标准化 + CV 选 alpha**：早期版本在**未标准化**的特征上施加 alpha=1.0/0.1 的小惩罚，导致 Ridge/Lasso 系数与 OLS 几乎完全相同（三模型实为一模型）。现先 `StandardScaler` 标准化，再用 `TimeSeriesSplit` 在对数网格（Ridge logspace(-3,3)、Lasso logspace(-3,1)）上选 alpha，使正则化真正生效、三模型显著有别。

#### Benchmark

> Conjura MMM Dataset 为 2024 年 6 月发布的学术数据集，暂无官方竞赛 Baseline。以下为 MMM 领域公认的标准建模基准：

| 参照系 | R^2 | Adj. R^2 | 说明 |
|--------|-----|---------|------|
| **MMM 领域基准（Ridge）** | 0.70–0.85 | — | 品牌级、含完整促销/价格/竞争数据的 MMM（参考文献：Hanssens et al., *Market Response Models*, 2nd ed.） |
| **naive 均值预测** | ~0.0 | — | 用历史 revenue 均值作为预测 |
| **单变量（最大渠道）** | ~0.35 | — | 仅用 spend 最高的单一渠道回归 |
| **本项目 OLS（in-sample / holdout）** | 0.542 / 0.440 | 0.536 | 单品牌全渠道线性回归（auto-selected brand=5488b0，按时间留出 183 天 holdout） |
| **本项目 Ridge（in-sample / holdout）** | 0.342 / 0.419 | — | L2 正则化，标准化 + CV 选 alpha=1000 |
| **本项目 Lasso（in-sample / holdout）** | 0.542 / 0.440 | — | L1 正则化，标准化 + CV 选 alpha=10 |

> R^2 ~ 0.54（in-sample）/ 0.44（holdout）反映了跨品牌聚合层面 MMM 的典型挑战：无价格/促销/竞品数据时，仅用渠道 spend 解释 revenue 变异的天然上限；holdout 比 in-sample 低约 0.10，是诚实的泛化估计。品牌级细分模型可达 0.70–0.85。

#### 模型结果

| 模型 | R² (in-sample) | R² (holdout) | MAE (holdout) | 最优正则化参数 |
|------|-----|---------|---------|---------------|
| OLS | 0.542 | 0.440 | 1,816,109 | — |
| **Ridge** | 0.342 | 0.419 | 1,812,313 | alpha = 1000（CV） |
| Lasso | 0.542 | 0.440 | 1,816,124 | alpha = 10（CV） |

> Ridge 在 CV 选出的大 alpha 下显著收缩系数（in-sample R² 降至 0.342，但 holdout R² 0.419 与 OLS 的 0.440 接近），说明强正则化牺牲少量偏差换取更稳定的系数——这对预算优化外推尤其重要。Lasso 选出 alpha=10 但未将任何系数压到 0（spend 变量均有解释力）。Durbin-Watson 等诊断量由模型自动输出，参见 `data/processed/models/mmm_results.json`。

### 3. 多触点归因（`src/attributor/multi_touch_attribution.py`）

使用真实 **Criteo Attribution Modeling for Bidding Dataset**（30 天实时流量，约 1,650 万条展示、4.5 万次转化）。将 impression 级数据按 `uid` 聚合成用户旅程，取 Top 10 campaigns 作为独立渠道，其余 665 个 campaign 归入 `other` 桶，对比 5 种归因模型 + 移除效应分析：

| 渠道 | First-Touch | Last-Touch | Linear | Time-Decay | **Shapley** | **Removal Eff.** |
|------|:-----------:|:----------:|:------:|:----------:|:-----------:|:----------:|
| campaign_10341182 | 4.2% | 5.1% | 4.2% | 5.0% | **4.2%** | **16.9%** |
| campaign_9100693 | 3.2% | 4.7% | 5.3% | 4.4% | **3.3%** | **19.4%** |
| campaign_15184511 | 3.3% | 3.2% | 2.7% | 3.2% | **3.2%** | **16.1%** |
| campaign_30801593 | 3.0% | 3.0% | 3.2% | 2.9% | **2.9%** | **1.1%** |
| campaign_32368244 | 2.4% | 2.9% | 2.4% | 2.8% | **2.4%** | **13.3%** |
| campaign_15398570 | 2.2% | 2.2% | 1.7% | 2.2% | **2.3%** | **5.8%** |
| campaign_29427842 | 1.6% | 1.7% | 1.6% | 1.7% | **1.9%** | **6.5%** |
| campaign_2869134 | 1.9% | 2.9% | 3.5% | 2.7% | **1.9%** | **12.0%** |
| campaign_31772643 | 1.2% | 1.1% | 0.8% | 1.1% | **1.3%** | **5.4%** |
| campaign_28351001 | 1.3% | 1.1% | 0.8% | 1.1% | **1.3%** | **3.5%** |
| **other** | **75.7%** | **72.0%** | **73.7%** | **73.0%** | **75.5%** | **0.0%** |

> 注：campaign ID 为 Criteo 数据中的匿名化广告活动 ID。`other` 聚合了除 Top 10 外的 665 个长尾 campaign，因此占据绝大部分展示与转化。

**关键发现：**

- **规则类模型（First/Last/Linear/Time-Decay）**结论高度一致：由于 `other` 桶覆盖了绝大多数展示，各规则模型都将其归因份额排在 72%–76% 之间。
- **Shapley Value** 在真实数据上仍提供了最稳定的分配，Top 3 渠道（campaign_10341182、campaign_9100693、campaign_15184511）合计贡献约 11%，与规则类模型趋势一致。
- **移除效应分析（Removal Effect）** 对 `other` 桶不敏感（移除 `other` 后剩余样本极少，导致其份额被压到 0%），但对头部单一 campaign 非常敏感——campaign_9100693 和 campaign_10341182 的移除效应分别达到 19.4% 和 16.9%，说明这两个 campaign 对整体转化率的边际影响最大。
- **方法启示**：真实数据下归因模型之间的差异比模拟数据更小（因为 `other` 桶占据主导），但 Shapley 与 Removal Effect 仍能有效识别头部高影响 campaign。

### 4. 预算优化（`src/attributor/budget_optimizer.py`）

以 Ridge MMM 的系数作为渠道**饱和响应函数**的渐近上限，在总预算约束下用 SLSQP 求解最优分配：

**响应模型（饱和，而非线性）**：

```
revenue_i = coef_i · spend_i^gamma / (spend_i^gamma + tau_i^gamma)
```

其中 `coef_i` 取自 Ridge 弹性系数，`gamma=1.5` 为 Hill 斜率，`tau_i`（半饱和点）锚定在渠道当前日均 spend——这样模型在观测运营点附近与线性弹性一致，但随 spend 远超 `tau_i` 时收益递减、趋于 `coef_i` 上限。

| 场景 | 总预算 | 预测 Revenue（当前 → 优化） | 提升幅度 |
|------|--------|-------------|---------|
| 当前分配（Baseline） | 100% | $2,082,150 → — | — |
| **重新优化分配** | 100% | $2,082,150 → $2,082,159 | **≈ 0.0%** |
| 预算 +10% + 优化 | 110% | $2,082,150 → $2,082,159 | ≈ 0.0% |
| 预算 +20% + 优化 | 120% | $2,082,150 → $2,082,157 | ≈ 0.0% |
| 预算 -10% + 优化 | 90% | $2,082,150 → $2,082,160 | ≈ 0.0% |

> **诚实的业务启示**：早期版本用**线性**响应函数，在总预算约束下最优解退化为平凡贪心——把预算挪向最高弹性渠道并外推到训练范围之外，得出 +1.8% 的虚高提升（线性模型隐含无限规模回报，本身无法支撑预算建议）。改用饱和响应后，在该品牌的当前运营点（各渠道已接近各自 `tau`）重新分配几乎无额外收益（≈0%），增量预算的边际收益也被饱和函数天然抑制——这正是预算优化应有的、诚实的结论：**当前分配已接近局部最优**。若要释放优化空间，需引入更强的特征（价格、促销、竞品）或在更细粒度（子渠道/时段）上建模。

### 5. 预算优化的不确定性（`src/attributor/budget_uncertainty.py`）

> 单点估计会害死人。第四节的优化器输出「google_video 应花 \$17,542」这样一个**单点**——CMO 拿它去砍预算时，隐含的假设是「我精确地知道最优分配」。但 Ridge 系数本身有噪声、训练样本只是历史的一小段，单点估计掩盖了真实的决策风险。本节给每个渠道的「最优 spend」与「revenue 提升」配 **95% 置信区间**，把项目从「会优化」升级为「懂决策风险」。

#### 方法：block bootstrap（为什么不是朴素 bootstrap）

朴素 case-resample（逐行有放回重抽样）假设样本独立同分布。但 MMM 残差**强自相关**——本项目 OLS 的 Durbin-Watson = **0.90**（无自相关应为 2.0）。在强自相关下逐行重抽样等于「假装样本独立」，会**系统性低估不确定性、CI 偏窄**，给决策者虚假的精确感。

解决方案是 **block bootstrap**：把时间序列切成连续的 block（默认 7 天一块），**有放回重抽样整块、块内保持原始时间顺序**地拼成新训练集——保留块内自相关结构，得到诚实的不确定性区间。

```
朴素 case-resample          block bootstrap
┌─────────────────┐         ┌──┬──┬──┬──┬──┐   原序列切成连续块
│ 打乱时序        │         └──┴──┴──┴──┴──┘
│ 假装 i.i.d.     │   vs.   ┌──┐┌──┐┌──┐┌──┐   有放回抽整块
│ → CI 偏窄       │         │  ││  ││  ││  │   块内保序
│ → 虚假精确感    │         └──┘└──┘└──┘└──┘
└─────────────────┘         → CI 诚实（更宽）
```

> 这不是「为了显得更严谨而加复杂度」——是朴素法在这里**根本是错的**。`tests/test_uncertainty.py::TestBlockVsNaive` 直接证明：在 AR(1)=0.85 的自相关序列上，block 重抽样后 lag-1 自相关 ≈ 0.7（保留），naive 重抽样后坍缩到 ≈ 0（破坏）。CI 宽度上 block 均值区间是 naive 的数倍宽——这就是被自相关放大的、真实存在的不确定性。

#### 诚实的解读：哪些结论可信、哪些不可信

- **估计稳定的渠道**（CI 窄、相对点估计比例小）：`meta_facebook`、`meta_instagram`、`google_pmax`——这三者当前 spend 量级大（数十万级）、且已接近各自饱和点，bootstrap 重抽样后最优分配几乎不动。**对这三个渠道，单点估计可以作为决策依据。**
- **估计不确定的渠道**（CI 极宽、下界逼近 0）：`google_display`（CI [\$706, \$7,499]，宽 \$6,793）、`google_shopping`（CI [\$244, \$7,320]）。**诚实地说：现有数据不支持对这两个渠道下强结论。** 它们的 Ridge 系数符号本身就不稳定（display 的 OLS 系数 p=0.31 不显著），优化器在「该砍」和「该加」之间反复横跳，CI 自然横跨整个允许区间。拿单点估计去砍这两个渠道的预算是危险的——需要更多数据（更长时间窗 / 价格促销特征）或更细分粒度才能定论。
- **block vs naive 的倍数**说明朴素法在这些渠道上会把 CI 低估 7–18 倍。最后两列的 `block/naive` 比值越高，说明该渠道的不确定性越是被自相关主导——朴素法在那里错的越离谱。

> 这是本项目最想传达的「懂决策风险」的深度：优化器不只输出一个数字，而是告诉你「这个数字有多可信」。CI 宽不代表模型失败，它诚实地反映了数据的信息量——宽 CI 本身就是一个有价值的决策输入（「这个渠道的结论不确定，先别动它」）。

**配置**（`config.py` 集中管理，单点修改）：`BLOCK_SIZE_DAYS=7`、`N_BOOTSTRAP=200`、`BOOTSTRAP_CI_LEVEL=0.95`、`BOOTSTRAP_RANDOM_SEED=42`。复用而非复制 `mmm_model.fit_ridge` / `prepare_features` / `chronological_split` 与 `budget_optimizer.optimize_budget` / `extract_params` / `load_mmm_results`。

</details>

---

## 快速开始

```bash
git clone https://github.com/MeaFew/attributor.git
cd attributor

# 1. 创建并激活 Python 3.11 虚拟环境
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. 安装锁定依赖、项目包和开发工具
make setup
# Windows 无 GNU Make：python -m pip install -r requirements.lock
#                    python -m pip install -e ".[dev]"

# 3. 下载 MMM 数据集（GitHub Releases，约 31MB）
bash download_data.sh

# 4.（可选但推荐）下载真实归因数据集 Criteo Attribution Modeling for Bidding Dataset
#    约 623MB，下载后放到 data/raw/criteo_attribution_dataset.tsv.gz
#    官方：https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/

make all          # 清洗 → MMM → 归因 → 优化
                  # 归因步骤自动判断：有 Criteo 原始数据则走真实 preprocess_criteo，
                  # 否则 fallback 到 generate_touchpoints 合成数据（与 run_all.py 行为一致）
python -m attributor.budget_uncertainty  # 复现 README 的 block-bootstrap 区间
# Windows (无 GNU Make): python run_all.py          # 运行完整管线：清洗 → MMM → 归因 → 优化
make dashboard    # 启动 Streamlit 交互看板
make verify       # 本地质量门（lint + format + test + audit）
```

<details>
<summary><b>📂 项目概览 / 技术架构 / 业务问题</b></summary>

本系统基于 figshare 发布的「Conjura Multi-Region MMM Dataset」（覆盖约 100 个电商品牌、19 个地区、2019–2024 年共 132,759 条日粒度记录），构建了一套从**宏观营销组合建模（MMM）**到**微观用户旅程归因**再到**预算约束优化**的完整分析链路。

核心解决的业务问题：

- **渠道 ROI 量化困难**：多个渠道同时投放时，如何剥离各渠道对转化的真实贡献？
- **归因模型选择无依据**：First-touch、Last-touch、Shapley Value、移除效应分析 (Removal Effect) 等方法结论差异巨大，如何系统比较？
- **预算分配凭经验**：在总预算约束下，如何科学重新分配各渠道 spend 以最大化 revenue？

### 技术架构

```mermaid
flowchart LR
    A[Raw CSV<br/>132K rows x 50 cols] --> B[Polars ETL]
    B --> C[Parquet]
    C --> D[MMM Modeling<br/>OLS / Ridge / Lasso]
    C --> E[User Journey<br/>Criteo 16.5M]
    E --> F[6 Attribution Models]
    D --> G[Budget Optimizer<br/>scipy SLSQP]
    F --> G
    G --> H[Streamlit Dashboard]
```

| 层级 | 技术选型 | 设计理由 |
|------|---------|---------|
| 数据清洗 | **Polars** | 向量化执行 + 惰性求值，处理 132K 行毫秒级 |
| 存储 | Parquet | 列式压缩，高效读写 |
| 宏观建模 | **statsmodels** + **scikit-learn** | OLS 提供完整统计推断（p-value、置信区间）；Ridge/Lasso 处理渠道间共线性 |
| 微观归因 | 自研 6 种模型 | 覆盖规则类（First/Last/Linear/Time-decay）与博弈论类（Shapley/移除效应分析），便于横向对比 |
| 预算优化 | **scipy.optimize** SLSQP | 支持等式约束（总预算不变）与不等式约束（单渠道下限），收敛稳定 |
| 交付 | **Streamlit** + **Plotly** | 三页交互看板：MMM 概览 / 归因对比 / 预算模拟器 |

</details>

<details>
<summary><b>📁 项目结构</b></summary>

```
attributor/
├── src/attributor/
│   ├── preprocess.py              # Polars ETL：缺失值、千分位处理、adstock、衍生指标
│   ├── mmm_model.py               # OLS + Ridge + Lasso，VIF / Durbin-Watson / 残差诊断
│   ├── generate_touchpoints.py    # 基于真实渠道结构模拟 50K 用户旅程（fallback）
│   ├── preprocess_criteo.py       # 将 Criteo impression 数据聚合为用户旅程
│   ├── multi_touch_attribution.py # 6 种归因模型：First / Last / Linear / Time-decay / Shapley / Removal Effect
│   ├── budget_optimizer.py        # scipy.optimize SLSQP 预算约束优化（饱和 Hill 响应）
│   └── budget_uncertainty.py      # block bootstrap 给最优 spend 配 95% 置信区间（不确定性量化）
├── notebooks/
│   └── 01_eda.ipynb               # 探索性数据分析
├── dashboard/
│   └── app.py                     # Streamlit 三页交互看板
├── tests/
│   ├── test_algorithms.py         # 算法单元测试：6 归因模型（含 Shapley 数学正确性）+ MMM 三模型 + 预算优化器
│   ├── test_core_modules.py       # 核心模块扩展测试：MMM 拟合 / Shapley / SLSQP 优化器边界场景
│   ├── test_preprocess.py         # 数据清洗单元测试
│   ├── test_mmm.py                # 模型输出格式与统计量测试
│   ├── test_uncertainty.py        # block bootstrap 不确定性：切块逻辑 / CI 覆盖率 / block vs naive 对比
│   └── test_attribution.py        # 归因归一化与边界条件测试
├── data/
│   ├── raw/                       # Conjura MMM dataset（figshare）
│   └── processed/                 # 清洗后 Parquet
├── reports/
│   └── images/                    # 生成的图表
├── Makefile                       # 工作流编排
├── requirements.lock              # 可复现的锁定依赖
└── .github/workflows/ci.yml       # GitHub Actions：lint + test + docker-build
```

</details>

<details>
<summary><b>⚠️ 局限与生产化思考</b></summary>

| 局限 | 当前方案 | 生产化路径 |
|------|---------|-----------|
| 多触点归因使用 Criteo 真实数据，但 campaign 已聚合 | 取 Top 10 campaigns + `other` 桶（665 个 campaign），便于 Shapley 计算 | 直接接入 CDP/Segment/Tealium 获取完整、未聚合的用户旅程；或使用更多计算资源处理全部 campaign |
| MMM 为日粒度 | 原始数据的日粒度已提供一定的时间分辨率  | 引入 hour-of-day 或 daypart 特征进一步精细化 |
| 无竞争环境变量 | 模型假设市场份额不变 | 引入竞品 spend 数据（如 Pathmatics、Sensor Tower） |
| 单节点执行 | 本地 Parquet | 迁移至 Snowflake/BigQuery + dbt 管线编排 |
| 预算优化为静态 | 一次性求解，未考虑动态预算调整 | 强化学习（PPO / MADDPG）实现实时预算竞价 |

</details>

---

## 相关项目

| 项目 | 仓库 | 简介 |
|------|------|------|
| 电商用户行为分析 | [MeaFew/shoplytics](https://github.com/MeaFew/shoplytics) | 2,900万条真实用户行为数据，10大分析模块 |
| 信用风险评分 | [MeaFew/riskscore](https://github.com/MeaFew/riskscore) | WOE/IV + XGBoost/LightGBM + SHAP 可解释性 |
| 多元时序预测 | [MeaFew/foresight](https://github.com/MeaFew/foresight) | LSTM / Transformer / XGBoost 时序预测对比 |
| 图神经网络反欺诈 | [MeaFew/graphguard](https://github.com/MeaFew/graphguard) | 图神经网络非法交易检测 |

---

## 许可证

代码采用 MIT License。数据集来源于 figshare 公开发布的 Conjura MMM Dataset，遵循其使用条款。
