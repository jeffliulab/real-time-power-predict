# Part 2 — 架构搜索：编码器-解码器 CNN-Transformer

> **状态：架构与代码就绪，训练结果待填入**（训练完成后在"实验结果"章节填 MAPE 数字并附训练曲线）。

## 1. 任务与动机

Part 1 的 baseline 是**单一 self-attention 编码器**：历史 + 未来所有时间步的空间 token 和表格 token 被拼成一个 3120-token 的扁平序列（`(S+24)·(P+1) = 48·65`），4 层 pre-norm Transformer 自注意力，最后从 24 个未来 tabular 位置切出状态过 MLP 预测。这个架构在 2022 年末 2 天测试上取得了 5.24 % MAPE，作为 baseline 扎实。但我们认为它有两个**结构性缺陷**：

1. **角色混淆**。在日前预测里，未来协变量（天气 + 日历）的真实身份是**查询（query）**，历史观测是**被查的记忆（memory）**。把它们塞进一个 joint encoder 等于让 query 和 memory 相互做 self-attention，理论上不错但实际上让 attention 头要在"分辨 token 是过去还是未来"上浪费表达力。经典的 seq2seq 把两者明确分开更干净。
2. **算力配比浪费**。S=24 历史 + 24 未来，序列中 50 % 是未来。编码器每层 3120² 的 self-attention 里，未来-未来 attention 没有信息流入（因为未来的真实 demand 没给），却占了 1/4 的注意力预算。我们用这部分算力换一个明确的 decoder 更划算。

Part 2 的目标：提出并训练一个**编码器-解码器架构**（`cnn_encoder_decoder`），在相同的参数预算、相同的数据切分、相同的 24 h SLURM 训练窗口下**超越 baseline 的 5.24 % 测试 MAPE**。

## 2. 架构

### 2.1 总览

![arch](figures/part2_arch.png)

```
  天气地图 (B, S+24, 450, 449, 7)        日历 (B, S+24, 44)
            │                                       │
            ▼                                       │
       Shared WeatherCNN                            │
     (ResBlock ×5 stride-2 + AdaptiveAvgPool 8×8)   │
            │                                       │
       (B, S+24, 64, 128) 空间 token                │
          │          │                              │
      hist part   future part                       │
          │          │                              │
          ▼          ▼                              │
   ┌─────────────┐   │                              │
   │  Encoder    │   │                              │
   │ hist tokens │   │                              │
   │ S×(P+1) =   │   │                              │
   │ 1560 token  │   │                              │
   │ 4× self-attn│   │                              │
   └──────┬──────┘   │                              │
          │          │                              │
       mem_hist      │ future spatial (optional)    │
          │          │ mem_future                   │
          └────┬─────┘                              │
               │                                    ▼
               ▼                       future_cal → future_tabular_embed
         ┌──────────────┐                           │
         │   Decoder    │◀────24 queries (seeded)◀──┘
         │ 2 layers:    │
         │  self-attn   │
         │  cross-attn  │
         │  (optional   │
         │   xattn→fut  │
         │   weather)   │
         │  MLP         │
         └──────┬───────┘
                ▼
           MLP head (128→64→8)
                │
                ▼
           (B, 24, 8) MWh
```

### 2.2 关键设计决策

**共享 WeatherCNN**：历史和未来天气地图都过**同一套 CNN**、在一次 batched 前向里算完，然后按时间切分。这样可以保证 BatchNorm 统计量不漂移（同批归一化），也省掉 ~50 % CNN 前向开销。Baseline 里已经这么做，我们照搬。

**Encoder — 仅历史**：序列长度 1560（24 历史小时 × 65 token/小时）。4 层 pre-norm self-attention，每层 `1560² ≈ 2.43 M` 次 attention 对比，比 baseline 的 `3120² ≈ 9.73 M` 便宜 ~4×。省下来的算力留给：(a) 更多训练 epoch（18 vs baseline 14），(b) 可选的 future-weather cross-attention 分支。

**Decoder queries — 从 future calendar 初始化**：最关键的实现细节。如果把 decoder 的 24 个 query 定义为纯随机的 `nn.Parameter`（DETR 原始做法），在时间序列任务上经常 collapse — 因为 queries 之间区分度太低，训练早期 attention 会平均化到所有 memory 上。我们**用 future calendar 特征 + `demand_mask` 过 `future_tabular_embed`** 作为 query 的种子，再叠加一个可学习 offset 和 temporal positional embedding。这样每个 query 从一开始就带有"这是 t+k 小时的日历签名"的信息，cross-attention 自然知道该去 mem_hist 的哪个区域找。

**Decoder block 结构**：pre-norm 的 `self-attn(queries) → cross-attn(queries, mem_hist) → [optional cross-attn(queries, mem_future_weather)] → MLP`。每个子层都有残差连接。2 层足够 — decoder query 数只有 24，深 decoder 帮助不大且容易过拟合。

**可选的第二条 cross-attention**（`use_future_weather_xattn=True`）：让 decoder query 除了 cross-attend 历史 memory，还能 cross-attend 未来天气的 spatial tokens（24 × 64 = 1536 个 KV）。假设是对城市密集 zone（CT、NEMA_BOST，baseline 下误差分别 7.28 %、6.09 %），预测误差主要来自**天气时序没对上**（例如气温骤变的具体小时），让 query 直接看未来天气地图应该有帮助。默认关闭，先保证基础 v1 能跑通再加。

**参数量**（手算估计，HPC smoke-test 确认）：
- 默认配置 `n_encoder_layers=4, n_decoder_layers=2`：约 **2.28 M**（比 baseline 的 1.75M 多约 30 %，因为多出 2 个 cross-attention decoder block）。
- 为了更严格的"相同参数预算"对比，可选用 `n_encoder_layers=3, n_decoder_layers=2`（约 2.08 M）或 `n_encoder_layers=3, n_decoder_layers=1`（约 1.82 M，最接近 baseline）。
- Trade-off：4+2 容量最大、架构最"完整"（decoder 够深才能做真正的多轮 attention refinement）；3+1 参数最持平但 decoder 浅，对复杂模式建模能力弱。**推荐默认先跑 4+2**，如果结果显著超 baseline 就接受容量上的差异并在报告里解释；如果跑出来效果和 baseline 持平或更差再调。

### 2.3 训练配置

| 项 | Baseline | Part 2 (ED) |
|---|---|---|
| 优化器 | AdamW lr=1e-3 wd=1e-4 | 同 |
| 调度 | Cosine | Cosine + 500 步线性 warmup |
| 损失 | MSE（归一化空间） | 同 |
| Batch | 4 | 4 |
| Epochs 计划 | 15 | 18 |
| Epochs 完成 | 14（24 h 时限） | 目标 18 |
| Train 年份 | 2019 + 2020 | 2019 + 2020 + 2021 |
| Val 年份 | 2021 | 2022 |
| Grad clip | 1.0 | 1.0 |
| Early stop patience | 无 | 5 |
| 硬件 | A100-40GB × 1 | A100-40GB × 1 |

两处配置改动有意图：
1. **加入 warmup**：decoder 的 cross-attention 在训练早期很容易往奇怪方向拉；500 步线性 warmup 是 Transformer 经典做法，DETR 用 ~1000 步，我们 500 步是权衡。
2. **Train/val 切分**：Baseline 用 2019-2020 训、2021 验。现在拓展到 2019-2020-2021 训、2022 验。TA 测试集是 2022 末 2 天 — 所以验证集和测试集在同年但不同时段，是稍微更接近"真实部署"的切分。

## 3. 理论基础与参考文献

### 3.1 为什么 encoder-decoder 更适合日前预测

**Temporal Fusion Transformer**（Lim et al., IJF 2021）[1] 最早把 encoder-decoder + variable selection + interpretable attention 明确引入多步预测。TFT 把输入分成三类：静态协变量、已知未来协变量、观测的过去协变量 — 这个分类和我们的数据结构一致（zone ID 算静态、未来天气+日历算已知未来、历史 demand 算观测过去）。TFT 在 electricity/traffic 等 benchmark 上系统性超过单 encoder 的 Transformer baseline，主要收益来自**明确把"已知未来"当 decoder 的初值**。我们的设计本质是 TFT 的一个简化实现，配合 Earthformer 的空间 token 化。

**Earthformer**（Gao et al., NeurIPS 2022）[2] 在地球系统预测里用**cuboid attention + decoder queries**，decoder query 从目标时段的坐标嵌入里初始化，跟我们从 future calendar 初始化 query 的思路同源。

**PatchTST**（Nie et al., ICLR 2023）[3] 证明了时间序列上"先 patch 再 transformer"比"直接按时间步 transformer"更好 — 这个 insight 其实 Part 1 的 baseline 已经吸收了（CNN 把 weather 地图 patchify 成 spatial token）。Part 2 主要在时序维度上做进一步的 encoder/decoder 切分。

### 3.2 其他被考虑但未采用的方向

**Factorized space-time attention**（TimeSformer，Bertasius et al., ICML 2021 [4]；Earthformer cuboid）：把 3120-token 的 joint attention 拆成 per-timestep spatial + per-position temporal，FLOPs 便宜 ~15×。我们没选这条路是因为**算力不是 baseline 的真正瓶颈**（24 h 跑 14 epoch 已经是平台期），把算力优化当作 Part 2 的 thesis 叙事上偏弱。如果 v1 还有预算，可以作为 Part 3 的对照实验。

**iTransformer**（Liu et al., ICLR 2024 [5]）：把 attention 从时间维度翻转到 channel（zone）维度。对多变量长序列有效，但我们的"变量"是异构的（7 个天气 channel + 8 个 zone + 44 个日历 bit），硬 invert 语义不通，pass。

**Frequency-domain attention**（FEDformer，Zhou et al., ICML 2022 [6]）：用 FFT 基函数替代 self-attention。电力负荷有强烈 24 h/7 d 周期，理论上 match。但实现代价高（~1 天），且文献报告的提升更多在长 horizon（96+ h），24 h 日前 horizon 上收益有限。

**Foundation models（Chronos、TimesFM、Aurora、ClimaX）**[7,8,9,10]：需要预训练 checkpoint + HuggingFace/PyTorch Lightning 环境，24 h 窗口内接入风险过高。Part 3 或后续 work 再考虑。

**Quantile / pinball loss + hierarchical reconciliation**（NBEATSx [11]、probabilistic load forecasting 文献）：能改善 per-zone 不均问题，但改动点多，留给 Part 3 的独立研究。

### 3.3 完整参考文献

[1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.* International Journal of Forecasting, 37(4), 1748–1764.

[2] Gao, Z., et al. (2022). *Earthformer: Exploring space-time transformers for Earth system forecasting.* NeurIPS.

[3] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A time series is worth 64 words: Long-term forecasting with transformers.* ICLR. (PatchTST)

[4] Bertasius, G., Wang, H., & Torresani, L. (2021). *Is space-time attention all you need for video understanding?* ICML. (TimeSformer)

[5] Liu, Y., et al. (2024). *iTransformer: Inverted transformers are effective for time series forecasting.* ICLR.

[6] Zhou, T., et al. (2022). *FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting.* ICML.

[7] Ansari, A. F., et al. (2024). *Chronos: Learning the language of time series.* arXiv:2403.07815.

[8] Das, A., et al. (2024). *A decoder-only foundation model for time-series forecasting.* ICML. (TimesFM)

[9] Bodnar, C., et al. (2024). *Aurora: A foundation model of the atmosphere.* Microsoft Research.

[10] Nguyen, T., et al. (2023). *ClimaX: A foundation model for weather and climate.* ICML.

[11] Olivares, K. G., Challu, C., Marcjasz, G., Weron, R., & Dubrawski, A. (2023). *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx.* International Journal of Forecasting, 39(2), 884–900.

**扩展阅读**（Part 3 报告的 related work 会展开）：Informer [12]、Autoformer [13]、Crossformer、TimeMixer、TSMixer、TiDE、SparseTSF、Time-LLM、Moirai、Lag-Llama、TimeGPT、Pangu-Weather、GraphCast、MetNet-3、FourCastNet、Token Merging、TokenLearner、DLinear、NHITS、NBEATS、Perceiver IO、Swin Transformer 等。

[12] Zhou, H., et al. (2021). *Informer: Beyond efficient transformer for long sequence time-series forecasting.* AAAI.

[13] Wu, H., et al. (2021). *Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting.* NeurIPS.

## 4. 实验结果

> _此节待训练完成后填写。_ 计划表格如下。

### 4.1 测试集：2022 年末 2 天（和 Part 1 一致）

| 指标 | Baseline (Part 1) | Part 2 (ED) | Δ |
|---|---|---|---|
| Overall MAPE | 5.24 % | **TBD** | — |
| ME | 2.31 % | TBD | — |
| NH | 3.69 % | TBD | — |
| VT | 5.95 % | TBD | — |
| CT | 7.28 % ← 最难 | TBD | — |
| RI | 5.27 % | TBD | — |
| SEMA | 5.44 % | TBD | — |
| WCMA | 5.87 % | TBD | — |
| NEMA_BOST | 6.09 % ← 次难 | TBD | — |

### 4.2 参数量与算力

| 项 | Baseline | Part 2 (ED) |
|---|---|---|
| 总参数量 | 1.75 M | TBD |
| Encoder attention FLOPs/layer | `3120² × D` | `1560² × D` (≈25 %) |
| Decoder attention FLOPs/layer | — | `24² + 24·1560 × D`（很便宜）|
| Epochs completed in 24 h | 14 | TBD |
| 每 epoch 时间 | ~100 min | TBD（预期 ~70 min）|

### 4.3 训练曲线

_填入 `runs/cnn_encoder_decoder/figures/training_curves.png`。_

### 4.4 讨论

（待训练完成后填写）：
- Overall MAPE 是否超越 5.24 %，幅度；
- CT / NEMA_BOST 的改善幅度，是否符合 "encoder-decoder 帮助未来 query 定位更准" 的假设；
- 如果开了 `use_future_weather_xattn=True`，对比有无的差异；
- 训练曲线形状：收敛速度是否变快，过拟合迹象等。

## 5. 提交

- **Canonical 位置**：`/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part2-models/pangliu/`
- **内容**：`model.py`（此仓库 `evaluation/part2-models/pangliu/model.py`）+ `best.pt`（训练产出）+ `config.json`（训练产出）+ `models/__init__.py` + `models/cnn_encoder_decoder.py` + `models/cnn_transformer.py`（被 import）
- **Self-test**：`sbatch scripts/self_eval.slurm part2-models/pangliu 2`（`self_eval.py` 已经 model-agnostic，从 ckpt args 里自动解析 model 名）。

## 6. 局限与后续

- 没调超参（lr、embed_dim、layer 数）— 一天时间不够做网格搜索。
- 没做 ensemble — seed 固定的单次训练，方差未估。
- 没系统地对比 factorized attention / iTransformer / FEDformer — 留给 Part 3。
- 如果总 MAPE 没超 5.24 %，考虑立刻开 `use_future_weather_xattn=True` 二训。
