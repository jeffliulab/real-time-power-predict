"""
Encoder-Decoder CNN-Transformer for multi-zone energy demand forecasting.

Architecture (Part 2 variant of Part 1's cnn_transformer):
    Shared WeatherCNN encodes historical + future weather into spatial tokens.

    Encoder: history-only sequence
        S * (P + 1) tokens ([P spatial + 1 tabular] per historical hour)
        -> n_encoder_layers of pre-norm self-attention
        -> memory `mem_hist`

    Decoder: 24 future tabular queries seeded from future calendar features
        -> n_decoder_layers of (self-attn over queries, cross-attn to mem_hist,
           optional second cross-attn to future-weather spatial tokens, MLP)
        -> MLP head -> (B, 24, n_zones)

The forward() signature matches the Part 1 model exactly so the dataset
loader and training loop are reused verbatim.
"""

import torch
import torch.nn as nn

from .cnn_transformer_baseline import ResBlock2d, WeatherCNN, EnergyTransformerBlock


class CrossAttentionBlock(nn.Module):
    """Pre-norm decoder block: self-attn over queries -> cross-attn to memory -> MLP.

    A second optional cross-attention to a secondary memory (e.g. future weather)
    is inserted between the history cross-attn and the MLP when `mem2` is passed.
    """

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1,
                 has_second_cross=False):
        super().__init__()
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_q1 = nn.LayerNorm(embed_dim)
        self.norm_k1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.has_second_cross = has_second_cross
        if has_second_cross:
            self.norm_q2 = nn.LayerNorm(embed_dim)
            self.norm_k2 = nn.LayerNorm(embed_dim)
            self.cross_attn2 = nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout, batch_first=True
            )

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mem1, mem2=None):
        # Self-attention over queries
        h = self.norm_self(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention to primary memory (history)
        q = self.norm_q1(x)
        k = self.norm_k1(mem1)
        h, _ = self.cross_attn(q, k, k)
        x = x + h

        # Optional second cross-attention (e.g. future weather)
        if self.has_second_cross and mem2 is not None:
            q = self.norm_q2(x)
            k = self.norm_k2(mem2)
            h, _ = self.cross_attn2(q, k, k)
            x = x + h

        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CNNEncoderDecoderForecaster(nn.Module):
    """
    Encoder-Decoder hybrid CNN-Transformer for 24 h day-ahead demand forecasting.

    Parameters
    ----------
    n_weather_channels : int
        Number of weather input channels (7).
    n_zones : int
        Number of energy load zones (8).
    cal_dim : int
        Dimension of calendar features (44).
    history_len : int
        Number of historical timesteps S.
    embed_dim : int
        Transformer embedding dimension D.
    grid_size : int
        CNN output spatial grid side length G; P = G^2 spatial tokens per hour.
    n_encoder_layers : int
        Number of encoder self-attention layers over history tokens.
    n_decoder_layers : int
        Number of decoder cross-attention layers over future queries.
    n_heads : int
        Attention heads.
    mlp_ratio : float
        FFN hidden ratio.
    dropout : float
    use_future_weather_xattn : bool
        If True, each decoder layer adds a second cross-attention to the future
        weather spatial tokens (24 * P additional keys/values).
    """

    def __init__(self, n_weather_channels=7, n_zones=8, cal_dim=44,
                 history_len=24, embed_dim=128, grid_size=8,
                 n_encoder_layers=4, n_decoder_layers=2,
                 n_heads=4, mlp_ratio=4.0, dropout=0.1,
                 use_future_weather_xattn=False):
        super().__init__()
        self.n_zones = n_zones
        self.cal_dim = cal_dim
        self.history_len = history_len
        self.future_len = 24
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.n_spatial = grid_size * grid_size
        self.tokens_per_step = self.n_spatial + 1
        self.total_steps = history_len + self.future_len
        self.use_future_weather_xattn = use_future_weather_xattn

        # --- Shared weather CNN ---
        self.weather_cnn = WeatherCNN(n_weather_channels, embed_dim, grid_size)

        # --- Tabular embeddings ---
        self.hist_tabular_embed = nn.Linear(n_zones + cal_dim, embed_dim)
        self.future_tabular_embed = nn.Linear(n_zones + cal_dim, embed_dim)
        self.demand_mask = nn.Parameter(torch.zeros(n_zones))

        # --- Positional embeddings ---
        # Spatial positions (0..P-1), shared across all timesteps
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.n_spatial, embed_dim) * 0.02
        )
        # Temporal positions: encoder uses 0..S-1, decoder uses S..S+23
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, self.total_steps, embed_dim) * 0.02
        )
        # Tabular-type embedding (added only to tabular tokens)
        self.tabular_type_embed = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        # Learnable query embedding added to decoder queries (one per future hour)
        self.decoder_query_embed = nn.Parameter(
            torch.randn(1, self.future_len, embed_dim) * 0.02
        )

        self.pos_drop = nn.Dropout(dropout)

        # --- Encoder stack (self-attention over history tokens) ---
        self.encoder = nn.ModuleList([
            EnergyTransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # --- Decoder stack ---
        self.decoder = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim, n_heads, mlp_ratio, dropout,
                has_second_cross=use_future_weather_xattn,
            )
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, n_zones),
        )

    def _encode_weather(self, hist_weather, future_weather):
        """Run the shared WeatherCNN once on concatenated history + future frames.

        Keeps BatchNorm statistics aligned with the baseline's encoding recipe.

        Returns
        -------
        hist_spatial   : (B, S, P, D)
        future_spatial : (B, 24, P, D)
        """
        B, S, H, W, C = hist_weather.shape
        T = S + self.future_len
        all_weather = torch.cat([hist_weather, future_weather], dim=1)  # (B, T, H, W, C)
        flat = all_weather.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)
        tokens = self.weather_cnn(flat)  # (B*T, P, D)
        tokens = tokens.reshape(B, T, self.n_spatial, self.embed_dim)
        return tokens[:, :S], tokens[:, S:]

    def forward(self, hist_weather, hist_energy, hist_cal,
                future_weather, future_cal):
        """
        Parameters
        ----------
        hist_weather   : (B, S, 450, 449, 7)
        hist_energy    : (B, S, n_zones)
        hist_cal       : (B, S, cal_dim)
        future_weather : (B, 24, 450, 449, 7)
        future_cal     : (B, 24, cal_dim)

        Returns
        -------
        (B, 24, n_zones)
        """
        B = hist_weather.shape[0]
        S = self.history_len

        # --- Shared CNN over all weather frames ---
        hist_spatial, future_spatial = self._encode_weather(hist_weather, future_weather)

        # Add spatial positional embedding to all spatial tokens
        hist_spatial = hist_spatial + self.spatial_pos_embed.unsqueeze(0)
        future_spatial = future_spatial + self.spatial_pos_embed.unsqueeze(0)

        # --- Encoder: history tokens (64 spatial + 1 tabular per hour) ---
        hist_tab_in = torch.cat([hist_energy, hist_cal], dim=-1)  # (B, S, n_zones + cal_dim)
        hist_tab = self.hist_tabular_embed(hist_tab_in).unsqueeze(2)  # (B, S, 1, D)
        hist_tab = hist_tab + self.tabular_type_embed                  # mark as tabular

        # (B, S, P+1, D)
        hist_tokens = torch.cat([hist_spatial, hist_tab], dim=2)

        # Temporal positions 0..S-1 broadcast over the P+1 tokens within each hour
        hist_temporal = self.temporal_pos_embed[:, :S, :].unsqueeze(2)  # (1, S, 1, D)
        hist_tokens = hist_tokens + hist_temporal

        # Flatten to (B, S*(P+1), D)
        enc_seq = hist_tokens.reshape(B, S * self.tokens_per_step, self.embed_dim)
        enc_seq = self.pos_drop(enc_seq)

        for blk in self.encoder:
            enc_seq = blk(enc_seq)
        mem_hist = self.encoder_norm(enc_seq)  # (B, S*(P+1), D)

        # --- Decoder queries seeded from future calendar ---
        # Replace unknown future demand with the learnable demand_mask vector.
        future_demand = self.demand_mask.unsqueeze(0).unsqueeze(0).expand(
            B, self.future_len, -1
        )
        future_tab_in = torch.cat([future_demand, future_cal], dim=-1)
        queries = self.future_tabular_embed(future_tab_in)  # (B, 24, D)

        # Add decoder-specific temporal pos (S..S+23) + learnable query embed
        queries = queries + self.temporal_pos_embed[:, S:S + self.future_len, :]
        queries = queries + self.decoder_query_embed
        queries = self.pos_drop(queries)

        # Optional second memory: future spatial tokens with their own temporal pos
        mem_future = None
        if self.use_future_weather_xattn:
            fut_temporal = self.temporal_pos_embed[:, S:S + self.future_len, :].unsqueeze(2)
            fut_tokens = future_spatial + fut_temporal
            mem_future = fut_tokens.reshape(
                B, self.future_len * self.n_spatial, self.embed_dim
            )

        for blk in self.decoder:
            queries = blk(queries, mem_hist, mem_future)
        queries = self.decoder_norm(queries)

        return self.head(queries)  # (B, 24, n_zones)
