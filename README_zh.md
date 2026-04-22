<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE 日前电力需求预测</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow" alt="Status">
</p>

<p>
  针对 ISO New England 电网 8 个 load zone 的 <strong>24 小时逐时需求预测</strong>，多模态 CNN-Transformer 融合天气地图 + 历史电量 + 日历特征。
</p>

</div>

---

## 任务

给定截止到时刻 `t` 的历史窗口，预测 `t+1 … t+24` 全部 8 个 load zone 的 MWh 需求。输入同时包含高维天气地图（`450×449×7`）和表格化的时序数据（过去需求 + 日历特征）。按真实电网运营流程假设未来天气已有准确预报。

完整任务说明见 [docs/assignment.pdf](docs/assignment.pdf)。

| 成分 | 细节 |
|-----------|--------|
| **天气输入** | `(B, S+24, 450, 449, 7)`，逐时 HRRR 风格新英格兰再分析张量 |
| **电量输入** | `(B, S, 8)`，8 个 zone 的 MWh（ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST）|
| **日历特征** | hour/dow/month one-hot + holiday 标记（44 维），全部 `S+24` 时间步都有 |
| **输出** | `(B, 24, 8)`，未来 24 小时逐 zone 的 MWh 预测 |
| **评价指标** | 24 小时 × 8 zone 的平均 MAPE |

---

## 当前进度

| 部分 | 分数 | 截止 | 状态 |
|---|---|---|---|
| Part 1 — baseline CNN-Transformer patch 架构 | 40 | 4/15 | ✅ 已提交，2022 年末 2 天切片测试 MAPE **5.24 %**，独立验证通过（[runs/cnn_transformer/](runs/cnn_transformer/)）|
| Part 2 — 超越 baseline 的架构搜索 | 30 | **4/22** | 🔨 代码就绪：`cnn_encoder_decoder`（编码器-解码器 + cross-attention，详见 [docs/part2_report.md](docs/part2_report.md)），待 HPC 训练 |
| Part 3 — 模型诊断 OR 独立研究 | 30 | 5/1 | ⏳ 未开始（初步方案：Track A 地理注意力图）|
| 报告 + 展示 | — | 5/1 / 5/4 | ⏳ 未开始 |

详细工作计划见 [docs/progress.md](docs/progress.md)；课程集群上 Part 1 提交的位置见 [docs/hpc-evaluation-structure.md](docs/hpc-evaluation-structure.md)。

---

## Part 1 — Baseline（混合 CNN-Transformer，1.75M 参数）

架构严格按作业 Figure 2：

1. **空间 token** — 每张 `(450, 449, 7)` 天气快照过一个 ResBlock CNN（5 段 stride-2 下采样 + adaptive pool → `G×G` 网格，默认 `G=8`），产生 `P = G²` 个维度为 `D = 128` 的 token。
2. **历史表格 token** — 每个历史小时：`Linear(demand + 44-d calendar → D)`。
3. **未来表格 token** — 24 个预测小时：`Linear(learned_demand_mask + calendar → D)`。
4. **序列装配** — 每小时把 `P` 个空间 token + 1 个表格 token 拼在一起。加 learnable spatial pos-embed（时间上共享）+ temporal pos-embed（该小时内所有 token 共享）+ 表格类型 embedding。展平后长度 `(S+24)·(P+1) = 48·65 = 3120`。
5. **Transformer encoder** — pre-norm，4 层，4 头，GELU MLP。
6. **预测头** — 取 24 个未来表格 token → `MLP(D → D/2 → 8)`。

源码：[models/cnn_transformer.py](models/cnn_transformer.py)

### 训练配置

| 项 | 取值 |
|---------|-------|
| 优化器 | AdamW, lr=1e-3, wd=1e-4 |
| 学习率调度 | CosineAnnealingLR |
| 损失 | MSE（在 z-score 标准化目标上）|
| 历史长度 `S` | 24 小时 |
| 网格 `G` | 8 → P = 64 空间 token |
| epoch | 15（24h 时限里跑完 14 个，A100-40GB）|
| 训练 / 验证 | 2019–2020 / 2021 |

### 训练曲线

![training curves](runs/cnn_transformer/figures/training_curves.png)

本地 2021 验证最好 val MAPE 6.92%（epoch 12）。课程 held-out 评测脚本在 2022 年末 2 天上测到 5.24%。

---

## 项目结构

```
real-time-power-predict/
├── README.md                        # 本文件（英文）
├── README_zh.md                     # 中文版
├── .gitignore
├── docs/
│   ├── assignment.pdf               # 课程作业说明
│   ├── progress.md                  # 当前进度与剩余工作
│   └── hpc-evaluation-structure.md  # HPC 评测目录说明
├── models/
│   ├── __init__.py                  # 模型注册表
│   └── cnn_transformer.py           # Part 1 baseline
├── training/
│   ├── train.py                     # 训练入口
│   └── data_preparation/
│       └── dataset.py               # 数据集 + LRU 天气缓存
├── evaluation/
│   └── pangliu/                     # 课程评测 wrapper（get_model + adapt_inputs）
│       └── model.py
├── runs/
│   └── cnn_transformer/             # Part 1 产出
│       ├── config.json
│       ├── norm_stats.pt
│       ├── logs/training_log.csv
│       └── figures/training_curves.png
│       # checkpoint（best.pt, latest.pt）没入 git，按需从 HPC 拉回
└── scripts/
    └── train.slurm                  # Tufts HPC gpu 分区 SLURM 脚本
```

---

## 常用操作

### 同步代码到集群，开训

```bash
# 本地 — 把仓库 rsync 到 HPC
rsync -avz --exclude=__pycache__ --exclude=.git ./ tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/

# 集群 — sbatch 提交训练任务
ssh tufts-login
cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
sbatch scripts/train.slurm --epochs 15 --train_years 2019 2020 --val_years 2021
```

### 把结果拉回本地

```bash
rsync -avz --exclude=checkpoints tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/ ./runs/
```

### 运行课程评测

提交放在 `/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part1-models/<team>/`，详见 [docs/hpc-evaluation-structure.md](docs/hpc-evaluation-structure.md)。

```bash
# HPC 上
cd /cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation
sbatch -J part1-models/pangliu test_run.sh 2   # 2 = 测试天数
```

---

## 数据

所有数据都在 Tufts HPC 的 `/cluster/tufts/c26sp1cs0137/data/assignment3_data/`（约 278 GB）。训练脚本默认指向这个路径。

| 项 | 形状/类型 | 覆盖范围 |
|---|---|---|
| 天气张量 | `(450, 449, 7)` per hour | 2019–2023 逐时 |
| 电量 CSV | 8 个 zone，逐时 MWh | 2019–2023，UTC 时区 |
| Held-out 测试集 | 同上 | 2024 |

时间戳全部对齐到 UTC 以避免夏令时/数据源时区漂移。用训练集里随机 500 个样本估 z-score 统计量做标准化（缓存在 `runs/cnn_transformer/norm_stats.pt`）。

---

## 致谢

- **算力**：Tufts Research Technology HPC（NVIDIA A100-40GB）
- **课程**：Tufts CS 137 — Deep Neural Networks, Spring 2026
