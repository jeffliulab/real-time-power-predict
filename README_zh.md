<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE 日前电力需求预测</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-训练进行中-yellow" alt="Status">
</p>

<p>
  针对 ISO New England 电网 8 个 load zone 的 <strong>24 小时逐时需求预测</strong>，多模态 CNN-Transformer 融合天气地图 + 历史电量 + 日历特征。
</p>

<p>
  <em>HF Live demo 和 model weights 等 Part 2 训练完成后部署。</em>
</p>

</div>

---

## Highlights

- **两套架构端到端对比**：
  - **Part 1 baseline** — single-encoder CNN-Transformer（1.75 M 参数），2022 末 2 天测试 MAPE **5.24 %** ✅ 已提交
  - **Part 2** — encoder-decoder + cross-attention（~2.29 M 参数），🚂 在 Tufts HPC 链式训练中
- **完整 pipeline**：数据准备 → 训练 → 评测 → 推理 → 实时 demo（Part 3）
- **可信赖的评测**：独立复现 TA 评测 harness，MAPE 数字字节级一致
- **24h SLURM 限制下的韧性训练**：6 个 job 用 `--dependency=afterany` 串接力，跨多个 wallclock 窗口自动续训

---

## 目录

- [Highlights](#highlights)
- [任务定义](#任务定义)
- [数据与归一化](#数据与归一化)
- [架构](#架构)
- [结果](#结果)
- [项目结构](#项目结构)
- [Quick Start](#quick-start)
- [当前状态](#当前状态)
- [致谢](#致谢)

---

## 任务定义

给定截止到时刻 `t` 的历史窗口，预测 `t+1 … t+24` **全部 8 个 load zone** 的 MWh 需求。输入同时包含高维天气地图（`450×449×7`）和表格化时序数据（过去需求 + 日历特征）。按真实电网运营流程假设未来天气已有准确预报。

完整任务说明见 [docs/assignment.pdf](docs/assignment.pdf)。

| 成分 | 细节 |
|-----------|--------|
| **天气输入** | `(B, S+24, 450, 449, 7)`，逐时 HRRR 风格新英格兰再分析张量 |
| **电量输入** | `(B, S, 8)`，8 zone 的 MWh（ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST）|
| **日历特征** | hour/dow/month one-hot + holiday 标记（44 维），全部 `S+24` 时间步都有 |
| **输出** | `(B, 24, 8)`，未来 24 小时逐 zone 的 MWh 预测 |
| **指标** | 24 小时 × 8 zone 的平均 MAPE（物理 MWh 空间）|

---

## 数据与归一化

| 项 | 取值 |
|---|---|
| **数据源** | ISO-NE zonal load CSV + 7 通道 HRRR 风格天气张量（Tufts HPC）|
| **覆盖范围** | 新英格兰，450×449 网格，~3 km 分辨率 |
| **时间** | 2019 – 2023 逐时 |
| **训练** | 2019+2020（baseline）/ 2019-2021（Part 2）|
| **验证** | 2021（baseline）/ 2022（Part 2）|
| **测试（TA）** | 2022 年末 2 天 |

**归一化链路**（baseline 和 Part 2 共用同一套）：

1. `compute_norm_stats()`（[dataset.py](training/data_preparation/dataset.py)）— 用训练集 500 个随机样本估 z-score（缓存到 `runs/<model>/norm_stats.pt`）
2. 训练时：输入**和**目标都归一化 → MSE loss 在 z-score 空间
3. MAPE 在**物理 MWh 空间**计算（先反归一化）
4. 评测 wrapper（[part1-baseline](evaluation/part1-baseline/model.py) / [part2-encoder-decoder](evaluation/part2-encoder-decoder/model.py)）从 ckpt 读 `norm_stats` 并反归一化，TA 评测器看到的就是原始 MWh

---

## 架构

### Part 1 — Baseline（[cnn_transformer_baseline.py](models/cnn_transformer_baseline.py)，1.75 M 参数）

混合 CNN-Transformer，严格按作业 Figure 2：

```
天气 (B,S+24,450,449,7) → Shared ResBlock CNN → 8×8 spatial token grid (P=64, D=128)
    +
Tabular tokens（每小时 1 个）：Linear(demand+calendar → D)
    +
Spatial pos-embed × Temporal pos-embed × Tabular type-embed
    ↓
单一 4 层 Transformer encoder（self-attention 跨 3120 个 token）
    ↓
切出 24 个未来 tabular token → MLP(128→64→8) → (B, 24, 8) MWh
```

### Part 2 — Encoder-Decoder（[cnn_encoder_decoder.py](models/cnn_encoder_decoder.py)，2.29 M 参数）

日前预测本质是**类翻译任务**：已知的未来协变量当 query，过去观测当 memory。架构因此重组：

```
              Encoder（4 层 self-attn，仅历史 token）
              S × (P+1) = 1560 token →  mem_hist
                        ↓
24 个 decoder query（从 future calendar 嵌入初始化）
              ↓
              Decoder（2 层：self-attn → cross-attn → MLP）
                        ↓
              MLP(128→64→8) → (B, 24, 8) MWh
```

**为什么 encoder-decoder**。Baseline 单编码器把过去观测和未来 query 搅在同一个 self-attention 池里。专门的 decoder 把"未来=query、过去=memory"的 inductive bias 写进架构里。Encoder attention 开销下降 ~4×（1560² vs 3120²），省下来的算力投入更多 epoch / 更大 grid。

**可选 ablation**（`--use_future_weather_xattn`）：每个 decoder block 加一条 cross-attention 分支，对**未来**天气 spatial token（24 × 64 = 1536 KV）做 attention。恢复和 baseline 信息对等。

完整报告：[docs/part2_report.md](docs/part2_report.md)。

---

## 结果

### 测试 MAPE — 2022 末 2 天（TA 定义切片）

| 模型 | 参数量 | Overall | ME | NH | VT | CT | RI | SEMA | WCMA | NEMA_BOST |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Part 1)** | 1.75 M | **5.24 %** | 2.31 | 3.69 | 5.95 | 7.28 | 5.27 | 5.44 | 5.87 | 6.09 |
| Part 2 ED（epoch-6 快照） | 2.29 M | 6.82 % | 3.22 | 5.67 | 5.85 | 9.56 | 7.45 | 7.22 | 7.38 | 8.24 |
| Part 2 ED（最终）| 2.29 M | _待填_ | | | | | | | | |
| Part 2 ED + future-weather xattn | 2.42 M | _待填_ | | | | | | | | |

Part 2 数字等链式训练完成后填入（见 [当前状态](#当前状态)）。

### 训练曲线（val MAPE，越低越好）

| Epoch | Baseline (val 2021) | Part 2 ED (val 2022) |
|---|---|---|
| 0 | 11.22 % | 10.08 % |
| 4 | 9.16 % | 8.70 % |
| 6 | 8.76 % | **8.63 %** ✓ |
| 13 (baseline 最终) | **6.92 %** | _待填_ |

Part 2 在同 epoch 略胜 baseline，尽管 (a) val 集更难（2022 比 2021 天气更乱）、(b) 信息劣势（默认 `use_future_weather_xattn=False`）。

---

## 项目结构

```
real-time-power-predict/
├── README.md / README_zh.md         # 英文 / 中文总览
├── CLAUDE.md                        # repo 级 Claude 操作规则
├── .gitignore
├── docs/
│   ├── assignment.pdf               # 课程作业说明
│   ├── progress.md                  # 工作计划与进度
│   ├── part2_report.md              # Part 2 技术报告
│   ├── part3_references.md          # Part 3 阅读列表与方向
│   └── hpc-evaluation-structure.md  # Tufts HPC 评测目录布局
├── models/
│   ├── __init__.py                  # Registry: create_model, MODEL_DEFAULTS
│   ├── cnn_transformer_baseline.py  # Part 1 baseline (1.75M)
│   └── cnn_encoder_decoder.py       # Part 2 encoder-decoder (2.29M)
├── training/
│   ├── train.py                     # 训练入口（两种模型共用）
│   └── data_preparation/dataset.py  # 数据集 + LRU 天气缓存 + z-score 归一化
├── evaluation/
│   ├── part1-baseline/              # Part 1 评测 wrapper（→ HPC part1-models/pangliu/）
│   └── part2-encoder-decoder/       # Part 2 评测 wrapper（→ HPC part2-models/pangliu/）
├── inference/
│   └── predict.py                   # CLI 推理（离线预测）
├── space/                           # HF Spaces 实时 demo（Part 3 工作）
│   ├── app.py                       # Gradio UI
│   ├── model_utils.py               # 自包含的模型加载
│   ├── iso_ne_fetch.py              # ISO Express 实时数据 fetcher（占位）
│   ├── requirements.txt             # Space 依赖
│   └── README.md                    # 部署说明
├── tests/
│   └── smoke_test.py                # 参数量 + 前向+反向健康检查
├── runs/
│   ├── model_registry.json          # 所有训练模型的中心索引
│   ├── cnn_transformer_baseline/    # Part 1 产出
│   └── cnn_encoder_decoder/         # Part 2 产出（训练完）
└── scripts/
    ├── train.slurm                          # Part 1 训练 SLURM
    ├── train_cnn_encoder_decoder.slurm      # Part 2 训练 SLURM（24h，任意 GPU）
    ├── self_eval.py + self_eval.slurm       # 模型无关 MAPE 评测器
    ├── self_test.slurm + smoke_test.slurm   # 健康检查
    ├── cuda_probe.slurm                     # GPU module 兼容性 probe
    ├── setup_conda_env.slurm + fix_conda_env.slurm  # 建 cs137 conda env
    └── hf_upload.py                          # 推 ckpt 到 HF Hub
```

---

## Quick Start

### 1. 同步代码到 HPC + 开训

```bash
# 本地 → HPC
rsync -avz --exclude=__pycache__ --exclude=.git --exclude=runs --exclude=data \
    ./ tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/

# 提交链（3 × 24h，任何 GPU，--resume from latest.pt）
ssh tufts-login
cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
sbatch scripts/train_cnn_encoder_decoder.slurm \
    --resume runs/cnn_encoder_decoder/checkpoints/latest.pt
```

### 2. 拉结果回本地

```bash
rsync -avz --exclude=checkpoints \
    tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/ \
    ./runs/
# 需要时拉特定 ckpt
rsync -avz tufts-login:/cluster/.../checkpoints/best.pt \
    ./runs/cnn_encoder_decoder/checkpoints/
```

### 3. 跑 TA 评测器（独立验证）

```bash
ssh tufts-login
cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
sbatch scripts/self_eval.slurm runs/cnn_encoder_decoder/checkpoints/best.pt 2
```

提交位置（per Piazza 4/16）：`/cluster/.../evaluation/part2-models/pangliu/`，详见 [docs/hpc-evaluation-structure.md](docs/hpc-evaluation-structure.md)。

### 4. 本地 smoke test（CPU 即可）

```bash
python -m tests.smoke_test
```

### 5. CLI 推理（用保存的 sample）

```bash
python -m inference.predict \
    --checkpoint runs/cnn_encoder_decoder/checkpoints/best.pt \
    --sample tests/sample_input.pt
```

---

## 当前状态

| Part | 分数 | 截止 | 状态 |
|---|---|---|---|
| Part 1 — Baseline CNN-Transformer | 40 | 4/15 | ✅ 已提交，test MAPE **5.24 %**，独立验证通过 |
| Part 2 — 架构搜索（encoder-decoder）| 30 | 4/22 | 🚂 训练中（链 `36804770 → 36804771 → 36804772`；ablation 链 `36804839 → 36804840 → 36804841` 排队等前者）|
| Part 3 — 注意力地图 + 实时 demo | 30 | 5/1 | 📋 已规划（[docs/part3_references.md](docs/part3_references.md), [space/](space/)）|
| 报告 + 展示 | — | 5/1 / 5/4 | ⏳ 未开始 |

---

## 致谢

- **算力**：Tufts Research Technology HPC（NVIDIA A100-80GB / 40GB / P100）
- **课程**：Tufts CS 137 — Deep Neural Networks, Spring 2026
- **姊妹项目**：[real_time_weather_forecasting](https://github.com/jeffliulab/real_time_weather_forecasting)（作业 2）— 共用 HRRR 数据管道和项目结构范式
