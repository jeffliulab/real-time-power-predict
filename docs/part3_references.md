# Part 3 — Independent Study References

Curated reading list and starting points for the model-diagnosis /
independent-study deliverable.

## Track A — Geographic Attention Maps (preferred)

Extract attention weights from the trained encoder-decoder model and
project them back to the New England geographic grid to ask:

- Which 8×8 spatial cells does each load zone "look at" most?
- Does attention shift across forecast hours (e.g. urban vs. rural)?
- Do extreme weather days produce sharper attention?

### Key papers

1. **Attention Is All You Need** (Vaswani et al., NeurIPS 2017) — base
   self/cross-attention formulation. Reference for `nn.MultiheadAttention.forward(need_weights=True)`.
2. **Vision Transformer (ViT)** (Dosovitskiy et al., ICLR 2021) — patch
   tokenization. Our spatial token grid is the same idea applied to
   weather rasters.
3. **DINO / DINOv2** (Caron et al., ICCV 2021; Oquab et al., 2024) —
   show that self-supervised attention maps emerge structurally; our
   supervised attention should also be structurally meaningful.
4. **Earthformer** (Gao et al., NeurIPS 2022) — visualizes cuboid
   attention for precipitation nowcasting; decoder cross-attn → spatial
   memory pattern is canonical.
5. **Climate transformer interpretability** — Pathak et al. FourCastNet
   (NVIDIA 2022), Lam et al. GraphCast (Science 2023). Both visualize
   spatial attention on global weather.

### Implementation notes

- Modify `EnergyTransformerBlock.forward` and `CrossAttentionBlock.forward`
  to optionally return attention weights.
- For decoder cross-attn: weights have shape (B, n_heads, 24, 1560).
  Reshape (1560 = S * (P+1) = 24 * 65) and slice off tabular tokens to
  get (B, n_heads, 24, 24, 64) — 24 future hours × 24 history hours × 64
  spatial cells. Average over heads + history for a per-future-hour
  spatial heatmap.
- Reshape 64 spatial cells → 8×8 grid → overlay on the assignment's
  Figure 1 ISO-NE map.

## Track B — Independent Study (alternative)

### B1. Weather-extreme robustness

How does MAPE differ between:
- Normal days (40-60th percentile of daily mean temp)
- Heat waves (top 10% daily mean temp)
- Cold snaps (bottom 10%)

Hypothesis: model trained on average weather underperforms on extremes.

**References**:
- Hobeichi et al., "Robustness of climate impact forecasts" (Nature
  Comm. 2022)
- Vermeulen et al., "Heatwave demand surge in ISO-NE" (PNAS Energy 2023)

### B2. Calendar-feature ablation

Drop holiday flag → measure MAPE delta on holidays (Christmas Day,
Thanksgiving). Quantify how much the model relies on the explicit
holiday signal vs. learning patterns from past years.

### B3. Pretrained weather encoder transfer

Replace our trained-from-scratch ResBlock CNN with frozen features from:
- ClimaX (Nguyen et al., ICML 2023) — climate-pretrained ViT
- Aurora (Bodnar et al., 2024) — 1.3B atmospheric foundation model
- Same-resolution SimCLR/SatMAE pretrained on satellite imagery

Measure data-efficiency improvement (train on 6 months instead of 3
years; compare MAPE).

**References**:
- Nguyen et al., "ClimaX" (arXiv:2301.10343)
- Bodnar et al., "Aurora" (Microsoft Research 2024)
- Cong et al., "SatMAE" (NeurIPS 2022)

## Track C — Real-time deployment study

Beyond the demo, characterize:
- Inference latency on CPU (HF Spaces) vs GPU
- HRRR data fetch time (NOAA AWS S3 latency from various regions)
- Cold-start time on HF Spaces

Compare: would batched offline forecasting (run once at midnight UTC)
be more useful than on-demand interactive forecasting? What's the
operational latency budget for grid operators?

## Recommended Track Plan

**Primary**: Track A — geographic attention maps (~3-4 days work,
publishable visualization, tightly couples to Part 1+2 code).

**Secondary** (if time permits): Track B1 — extreme-weather robustness
analysis (~1-2 days, runs eval over saved val predictions, no retraining).

**Skip**: Track B3 (pretrained encoder transfer) — too much
infrastructure setup for the time budget.
