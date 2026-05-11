<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE 短期电力负荷预测</h1>
<h3>实时系统 + 部署漂移研究 + Drift-Weighted 集成框架</h3>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/release-v1.6-brightgreen" alt="v1.6">
  <a href="https://huggingface.co/spaces/jeffliulab/predict-power">
    <img src="https://img.shields.io/badge/🤗%20实时演示-HF%20Space-yellow" alt="Live demo">
  </a>
  <a href="https://github.com/jeffliulab/new-england-real-time-power-predict-data">
    <img src="https://img.shields.io/badge/📦%20数据仓库-cron--刷新-blue" alt="Data repo">
  </a>
</p>

<p>
  多模态 CNN-Transformer + Chronos-Bolt-mini 零样本融合，预测 ISO 新英格兰 8 个负荷区下一天 24 小时的逐区电力需求。在 2022 年末两天测试切片上 baseline MAPE 5.24%，集成后 4.21%。已在 HuggingFace Space 公开部署，配套每日刷新的 7 日滚动回测；以这套部署作为案例，研究 BTM 太阳能装机增长引发的部署漂移，并在论文 <a href="docs/paper.pdf">docs/paper.pdf</a> 中提出推断时的 drift-weighted 集成方案。
</p>

<p>
  <strong>v1.6 新增内容</strong>：三窗口漂移轨迹（2022-05 / 2025-05 / 2026-04-05）、按小时 MAPE 分解（高 BTM 区域中午 MAPE 是非中午的 <strong>约 9 倍</strong>）、duck-curve 深度对比（WCMA +17pp vs ME +6.6pp）、以及新方法 <strong>drift-weighted ensemble re-weighting</strong>——在 14 天滚动 validation 上重拟合逐区 α，完全不重训模型。
</p>

<p>
  <strong>🌐 实时 demo</strong>：<a href="https://huggingface.co/spaces/jeffliulab/predict-power">huggingface.co/spaces/jeffliulab/predict-power</a> — 真实 ISO-NE 逐区负荷 + 真实 HRRR 天气，每日更新。<br>
  <strong>📄 工作坊论文</strong>：<a href="docs/paper.pdf">docs/paper.pdf</a>（LaTeX 源码私有保留，等 arXiv 上传后开放）。<br>
  <strong>📦 自动数据仓库</strong>：<a href="https://github.com/jeffliulab/new-england-real-time-power-predict-data">jeffliulab/new-england-real-time-power-predict-data</a> —— GitHub Actions cron 每天 04:00 UTC 重建滚动回测。
</p>

</div>

---

## 项目亮点

- **训练好的 baseline**（1.75 M 参数）：单 encoder 的 CNN-Transformer，融合 7 通道 HRRR 天气栅格 + 24 小时逐区负荷历史 + 44 维 calendar one-hot → 在 2022 末两日测试切片上 **5.24% MAPE**（[模型代码](space/models/cnn_transformer_baseline.py)）。
- **基础模型集成** ⭐：trained baseline 与 **Chronos-Bolt** ([Amazon, Apache-2.0](https://huggingface.co/amazon/chronos-bolt-mini)，**零样本**直接使用，不做 fine-tune) 的逐区加权晚融合 → 用 205 M 参数的 `chronos-bolt-base` 取得 **4.33% MAPE**，用 21 M 参数的 `chronos-bolt-mini`（线上部署版）取得 **4.21%**。逐区权重 $\alpha_z$ 在 2022 年 14 天验证窗口上调出来后写死。
- **实时部署** ⭐：公开 HuggingFace Space，每次点击拉**实时 HRRR analyses + 实时 HRRR forecasts + 实时 ISO-NE 逐区负荷**。严格回测纪律：T 时刻预测只能用 T 时刻已发布的数据。背后辅助数据仓库每天用 GitHub Actions cron 重新跑 7 日滚动回测，把 JSON push 回去给 Space 启动时拉。
- **两窗口部署漂移研究**（论文 §4）：W1 = 2025-05 vs W2 = 2026-04/05，trained baseline 一年内从 **17.97% → 25.17%** MAPE，而不依赖天气的 Chronos 零样本只从 **9.62% → 13.45%**。逐区分解显示漂移集中在 MA + RI 区域，与各州屋顶 / 用户侧 (BTM) 太阳能装机增长一致。
- **几何注意力诊断**：4 张图 + 4 项方向正确性检查。意外发现：8 个预测头中有 7 个共享一张全局注意力图，并不会按预测的 zone 局部化。论文附录里压缩到 1 段 + 2 张图。

---

## 三仓库架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  github.com/jeffliulab/real-time-power-predict      （本仓库）      │
│  代码 source-of-truth: 训练 / 模型 / 推理 / scripts/ / 注意力诊断   │
│  论文源码 / Space 应用代码                                          │
│  ─ HF Space 在每次 push 时从 main 分支同步                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │ actions/checkout
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ github.com/jeffliulab/new-england-real-time-power-predict-data      │
│ （公开的、纯自动化数据仓库）                                        │
│ ─ .github/workflows/refresh.yml 每天 04:00 UTC 跑                   │
│ ─ 调用本仓库的 scripts/build_rolling_backtest.py，8 路并行抓 HRRR  │
│   + ISO-NE 5-min zonal CSV + Chronos-Bolt-mini                      │
│ ─ 输出 data/backtest_rolling_7d.json + iso_ne_30d.csv +             │
│   last_built.json，仅在内容变化时 commit                            │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HF Space 启动时 HTTPS GET
                         │ raw.githubusercontent.com/.../main/data/*
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ huggingface.co/spaces/jeffliulab/predict-power                      │
│ ─ Real-time 标签：实时 HRRR + 实时 ISO-NE 逐区，跑 ensemble 出图    │
│ ─ Backtest 标签：从数据仓库拉来的 7 日缓存（瞬时显示）              │
│ ─ About 标签：随 cron 每日刷新的动态 MAPE 摘要                      │
└─────────────────────────────────────────────────────────────────────┘
```

为什么拆三个仓库：HuggingFace 在 `real-time-power-predict` 收到 push 时会自动同步 Space，意味着如果回测数据每天 commit 进主仓库，就会触发每天一次 ~200 KB 数据的全量 Docker 重建。拆开后主仓库 git log 干净（只有真实代码改动），第三方也能直接 `curl` 看到最新回测数据，无需 clone。

---

## 任务定义

给定时间 `t` 之前的 24 小时窗口，预测 **ISO-NE 全部 8 个负荷区** 在 `t+1 … t+24` 的小时级 MWh 需求。

| 输入/输出 | 形状 |
|-----------|--------|
| **天气输入** | `(B, S+24, 450, 449, 7)` — 训练时用 HRRR f00 analyses，部署时未来窗口换成 HRRR f01..f24 forecasts |
| **能源输入** | `(B, S, 8)` — 8 个 zone 的小时级 MWh (ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST) |
| **日历** | `(B, S+24, 44)` — 一天中第几小时 (24) + 一周第几天 (7) + 月份 (12) + 美国节假日 flag (1) one-hot |
| **输出** | `(B, 24, 8)` — 8 个 zone 接下来 24 小时的预测 MWh |
| **指标** | 物理 MWh 空间下的 MAPE，跨 horizon 和 zone 平均；超过 (forecast-day, zone) 对的 1000 次 bootstrap 给 95% CI |
| **测试切片** | 2022 年最后两天（2022-12-30, 2022-12-31），加上 2025+ 的滚动 7 日实时窗口 |

---

## 数据与归一化

| 属性 | 值 |
|---|---|
| **负荷数据源** | ISO-NE 公共 Energy/Load/Demand 报告门户 zone 级小时档案；线上部署用 5 分钟 `fiveminuteestimatedzonalload` CSV（论文附录 B 描述了 cookie-prime trick）|
| **天气数据源** | NOAA HRRR f00 analyses，AWS S3 `s3://noaa-hrrr-bdp-pds/`，通过 `herbie-data` 库取 |
| **区域** | 新英格兰，40.5–47.5°N, –74 至 –66°W，Delaunay 重心插值后 450 × 449（约 1.6 km/格）|
| **覆盖范围** | 训练语料 2019–2023 小时级；线上部署 2025+ |
| **训练（baseline）** | 2019–2020（约 17K 样本）|
| **验证** | 2021（约 8K 样本）|
| **测试（训练窗口内自评）** | 2022 年最后 2 天 |
| **测试（线上部署）** | 滚动 7 日窗口（论文里用了 2025-05 和 2026-04/05）|

**四步归一化**（[`training/data_preparation/dataset.py`](training/data_preparation/dataset.py)）—— 所有模型变体共享：

1. `compute_norm_stats()` —— 从训练集随机抽 500 个样本算 z-score 统计；缓存到 `runs/<model>/norm_stats.pt`，并打包到每个 checkpoint。
2. 训练时：天气和负荷输入都做 z-score；目标也 z-score；MSE 在 z-空间计算。
3. MAPE 在 **物理 MWh 空间** 计算，预测要先反归一化。
4. 评测包装器在加载 checkpoint 时重新应用 `norm_stats`，部署管道吐出来的就是真实 MWh，与训练时评测同尺度。

---

## 模型架构

### Baseline ([`cnn_transformer_baseline.py`](space/models/cnn_transformer_baseline.py)，1.75 M 参数)

```
天气 (B, S+24, 450, 449, 7) → 共享 ResBlock CNN → 8×8 空间 token 网格 (P=64, D=128)
    +
表格 token（每小时 1 个）：Linear(demand+calendar → D)
    +
空间位置嵌入 × 时间位置嵌入 × 表格类型嵌入
    ↓
单层 4 stack Transformer encoder（对 3120 个 token 做 self-attention）
    ↓
切出 24 个 future-tabular tokens → MLP(128→64→8) → (B, 24, 8) MWh
```

### 基础模型集成 ([`space/model_utils.py`](space/model_utils.py))

trained baseline 与 **Chronos-Bolt-mini** ([Ansari 等 2024](https://arxiv.org/abs/2403.07815)，Amazon，Apache-2.0) 的逐区加权晚融合，**零样本**使用，不做 fine-tune。Chronos 那条路把每个 zone 当作独立的单变量序列处理：输入 720 小时（4 周）逐区历史，输出对应 zone 的 24 小时中位数分位数预测。**不看天气、不看 calendar、不知道 zone 标签**。

$$
\hat y_z = \alpha_z \cdot \hat y_z^{\text{baseline}} + (1 - \alpha_z) \cdot \hat y_z^{\text{Chronos}}
$$

逐区权重 $\alpha_z$ 在 2022-12-16 → 12-29 这 14 天验证窗口上 grid search 出来，部署时写死。$\alpha_z = 0$ 的 zone（CT, SEMA, NEMA_BOST）部署时直接丢掉 baseline；$\alpha_z$ 高的 zone（VT = 0.80）以 baseline 为主。

### 几何注意力诊断 ([`scripts/attention_maps.py`](scripts/attention_maps.py))

把 trained baseline 的 encoder attention 切出来，暴露 future-tabular → history-spatial 的权重，reshape 到 8×8 空间网格，再用 cartopy 叠到新英格兰底图上。包含 4 项方向正确性检查（slice 方向、行优先 reshape、罗盘方向、东 vs 西的聚合一致性）。5 个诊断日覆盖 mild / heat-wave / cold-snap / extreme / holiday 五种气候。

---

## 实验结果

### 训练窗口内自评（2022-12-30 → 2022-12-31）

| 变体 | 参数量 | 总体 MAPE | ME | NH | VT | CT | RI | SEMA | WCMA | NEMA_BOST |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline ⭐ | 1.75 M | **5.24 %** | 2.31 | 3.69 | 5.95 | 7.28 | 5.27 | 5.44 | 5.87 | 6.09 |
| **Ensemble (baseline + Chronos-Bolt-base)** ⭐ | 1.75 M + 205 M | **4.33 %** | — | — | — | — | — | — | — | — |
| **Ensemble (baseline + Chronos-Bolt-mini，部署版)** | 1.75 M + 21 M | **4.21 %** | — | — | — | — | — | — | — | — |

### 两窗口部署漂移研究（论文 §4）

跨两个滚动 7 日窗口的总体 MAPE [95% bootstrap CI]，加上三个 naive baseline 作 floor：

| 模型 | W1: 2025-05-01..07 | W2: 2026-04-28..05-04 |
|---|---|---|
| Baseline (CNN-Transformer + HRRR) | 17.97 % [13.65–22.84] | **25.17 %** [18.98–32.12] |
| Chronos-Bolt-mini (零样本) | 9.62 % [8.31–11.07] | **13.45 %** [10.75–16.39] |
| Ensemble (per-zone α) | 11.09 % [9.04–13.64] | **14.51 %** [11.23–18.07] |
| Persistence-1d | 9.97 % [8.23–11.78] | 15.34 % [11.79–19.35] |
| Persistence-7d | 12.80 % [10.44–15.08] | 15.50 % [12.64–18.68] |
| Climatological mean (4-week) | 13.20 % [10.90–15.80] | 15.77 % [12.69–19.23] |

W1 → W2 baseline 跳跃 (+7.2 pp) 超过窗口内 bootstrap CI 区间；不依赖天气的 Chronos 漂移得少；逐区分解把漂移定位到 MA + RI 几个 zone（论文图 3）。

### 部署管道等价性验证

把部署后的管道在 **2022-12-30**（训练窗口内）跑一遍，与 cluster 当时存下来的预测对齐，差距 ≤ 0.13 个百分点：cluster MAPE 6.54% [5.02–7.92] vs live 6.41% [4.99–7.69]，配对差 -0.13 pp [-0.23, +0.00]。逐元素差最大 94 MW（cluster 预测在最差点的 3.6%）。所以部署管道与训练时评测等价；后续线上数字测的是真实输入分布漂移，不是 bug。

---

## 实时部署

HF Space 在 [`jeffliulab/predict-power`](https://huggingface.co/spaces/jeffliulab/predict-power) 提供三个标签页：

1. **Real-time forecast** —— 每次点击拉 24 小时 HRRR f00 analyses + 来自最近一个长 cycle (00/06/12/18 UTC) 的 24 小时 HRRR forecast + 最近 24 个连续小时的 ISO-NE 5 分钟 zonal load。跑 baseline + Chronos-Bolt-mini，输出逐区加权 ensemble。冷启动约 3–5 分钟（HRRR fetch + Chronos load）；同一 session 后续点击约 10–30 秒。
2. **Backtest (last 7 days)** —— 最近 7 个完全公开日子的每日预测，严格回测纪律。逐区 × 逐模型 MAPE 表 + 总 MAPE 条形图。
3. **About** —— 随 cron 每日刷新的动态 MAPE 摘要。

最新数字由辅助数据仓库的 GitHub Actions cron 每日刷新：

```bash
curl -s https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main/data/last_built.json
```

---

## v1.6 release 内容

v1.6 在 v1.5 基础上加入：三窗口部署漂移轨迹（2022-05 / 2025-05 / 2026-04-05）、按小时 MAPE 分解 + duck-curve 深度证据（直接证据指向 BTM 太阳能机制）、drift-weighted 集成方法 + 3 个窗口的基准对比、Related Work 章节（覆盖 drift detection 和 foundation TS model 文献）、5 条面向部署系统的操作建议。所有制品钉死在 `v1.6` git tag。

| 制品 | 位置 | 类型 |
|---|---|---|
| 训练好的 baseline (1.75 M 参数, 5.24% 测试 MAPE) | https://huggingface.co/jeffliulab/predict-power-baseline | 模型 |
| 实时 demo (Real-time + 7 日滚动回测) | https://huggingface.co/spaces/jeffliulab/predict-power | 演示 |
| 每日刷新的回测数据 (cron, GitHub Actions) | https://github.com/jeffliulab/new-england-real-time-power-predict-data | 数据 |
| 工作坊论文 PDF (v1.6 内容) | [docs/paper.pdf](docs/paper.pdf) | 论文 |
| Checkpoint 仓库内拷贝 | `pretrained_models/baseline/best.pt`, `space/checkpoints/best.pt` | 模型 |
| 训练用 demand CSV (2019-2022 小时级 per-zone) | `pretrained_models/baseline/dump/demand_2019_2022_hourly.csv` | 数据 |
| 多年漂移 sweep + drift-weighted benchmark (脚本) | `scripts/experiments/{historical_drift_sweep, drift_weighted_ensemble}.py` | 代码 |
| Hour-of-day + load curve + paired CI + horizon 分析脚本 | `scripts/figures/render_*.py` | 代码 |

论文 LaTeX 源码、中间 JSON、原始图等留在私有工作目录（公开 release 不包含论文源码，直到 arXiv 上传）。以上制品都能从本仓库 + auxiliary 数据仓库重现。

---

## 项目结构

```
real-time-power-predict/
├── README.md / README_zh.md
├── CLAUDE.md                                # 仓库专用操作约定
├── docs/
│   └── paper.pdf                            # 工作坊论文 PDF（公开唯一论文制品；LaTeX 源私有）
├── space/                                   # HF Space 源码 (push 后自动同步到 HF)
├── training/                                # 训练入口 + dataset 模块
├── inference/                               # 命令行离线推理
├── pretrained_models/                       # HF 风格 model card + checkpoint 镜像
├── runs/                                    # 训练运行产物
├── models/                                  # encoder-decoder 源码 (lineage 留底，论文里没用)
├── tests/                                   # smoke test
└── scripts/
    ├── data_preparation/                    # 公开 ISO-NE + HRRR 离线 fetcher (训练数据构建)
    ├── baselines/                           # naive baselines (论文 §4)
    ├── experiments/                         # historical_drift_sweep.py
    ├── validation/                          # reproduce_dec30_2022.py + future_weather_shift_quantify.py
    ├── figures/                             # bootstrap_mape.py + 所有论文图渲染
    ├── attention_maps.py                    # 注意力诊断
    ├── build_rolling_backtest.py            # cron 调用的 7 日滚动回测构建器
    └── build_rolling_backtest.requirements.txt
```

---

## 快速开始

### 本地跑实时 demo（CPU 即可，无需 GPU）

```bash
cd space
pip install -r requirements.txt
sudo apt-get install libeccodes-dev libeccodes-tools     # cfgrib/HRRR 用
python app.py
# Gradio 会在 http://localhost:7860 启动
```

Space 启动时会从公开数据仓库加载 `BACKTEST`；如果机器无法联网，把 `space/assets/backtest_fallback.json` 放在 `app.py` 旁边，会自动用 bundled snapshot。

### 模型管道 smoke 测试

```bash
python -m tests.smoke_test
```

### 本地复现滚动回测 cron

```bash
pip install -r scripts/build_rolling_backtest.requirements.txt
python scripts/build_rolling_backtest.py --output-dir /tmp/backtest --parallel 8
```

### 复现论文里的多窗口实验

```bash
python scripts/experiments/historical_drift_sweep.py
# 3 个窗口（W0=2022-05、W1=2025-05、W2=2026-04/05），每个 ~75 分钟
# （HRRR 缓存暖；HRRR 缓存空时首跑 ~3–5 小时）
# 输出多年漂移 JSON，供论文图表使用
```

### 复现 drift-weighted 集成基准

```bash
python scripts/experiments/drift_weighted_ensemble.py
# 需要先跑完多窗口 sweep。在每个窗口的 14 天滚动 validation 上跑 baseline+Chronos，
# 按 0.05 步长 grid search 逐区 α，应用到测试窗口，
# 报告 frozen / drift-weighted / oracle 三模式的 per-window MAPE。
```

### 复现部署管道等价性验证

```bash
python scripts/validation/reproduce_dec30_2022.py
python scripts/validation/augment_validation_with_ci.py
# 期望 MAPE：live 6.41% vs cluster 6.54%，差 ≤ 0.13 pp
```

### 重新训练 baseline

```bash
python training/train.py --model cnn_transformer_baseline \
    --epochs 14 --batch_size 16 --lr 3e-4
# 需要 NVIDIA GPU (训练时用的是 A100-80GB / 40GB)
```

---

## 可复现性

- **随机种子**：训练管道没设 seed，所以 5.24% / 4.21% 这些数字在重新训练时不能逐字节复现。本 README 和论文里所有数字都钉死在 `pretrained_models/baseline/best.pt` 这个 checkpoint（与 `space/checkpoints/best.pt` 等价），随 `v1.6` tag 一起发布。
- **Datasheet**：见工作坊论文附录 C。
- **公开数据源**：ISO-NE Energy/Load/Demand 报告门户 + NOAA HRRR S3 镜像都是免认证公开的；实时 demo 和 cron 直接调用。
- **数字 → 源数据可追溯**：论文每个 MAPE 数字都对应到 `scripts/` 下某个脚本生成的 JSON。drift-weighted 方法（`scripts/experiments/drift_weighted_ensemble.py`）使用确定性 0.05 步长 grid search，输出可逐字节复现；bootstrap CI（`scripts/figures/bootstrap_mape.py`）使用固定 seed。

---

## 致谢

- **公开数据**：ISO New England (5 分钟 zonal load) + NOAA HRRR (3 km mesoscale 天气)
- **基础模型**：Chronos-Bolt by Amazon (Apache-2.0)
- **算力**：训练用 NVIDIA A100-80GB / 40GB / P100；服务用 HuggingFace Spaces cpu-basic
- **作者**：Pang Liu, Independent Researcher, [`jeff.pang.liu@gmail.com`](mailto:jeff.pang.liu@gmail.com)
