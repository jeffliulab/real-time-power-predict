"""
Hybrid CNN-Transformer for multi-zone energy demand forecasting.

Architecture (per assignment spec):
    1. CNN downsamples each weather map (450,449,7) -> (G,G,D), flatten to P spatial tokens
    2. Tabular tokens: Linear embed of (demand + calendar) per timestep
    3. Unified sequence of (S+24) * (P+1) tokens with spatial + temporal pos embeddings
    4. Transformer Encoder
    5. Slice future 24 tabular tokens -> MLP -> (B, 24, n_zones)
"""

import math
import torch
import torch.nn as nn


class ResBlock2d(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class WeatherCNN(nn.Module):
    """
    CNN that downsamples a weather map into a grid of spatial tokens.

    Input:  (B, 7, 450, 449)
    Output: (B, P, embed_dim)  where P = grid_size^2
    """

    def __init__(self, in_channels=7, embed_dim=128, grid_size=8):
        super().__init__()
        self.grid_size = grid_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock2d(32, 64, stride=2),
            ResBlock2d(64, 128, stride=2),
            ResBlock2d(128, 128, stride=2),
            ResBlock2d(128, embed_dim, stride=2),
        )
        # 450/2/2/2/2/2 ≈ 14, adaptive pool to exact grid_size
        self.pool = nn.AdaptiveAvgPool2d(grid_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.encoder(x)
        x = self.pool(x)           # (B, embed_dim, G, G)
        B, D, G1, G2 = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, P, D)


class EnergyTransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class CNNTransformerBaselineForecaster(nn.Module):
    """
    Hybrid CNN-Transformer for day-ahead energy demand forecasting.

    Parameters
    ----------
    n_weather_channels : int
        Number of weather input channels (7).
    n_zones : int
        Number of energy load zones (8).
    cal_dim : int
        Dimension of calendar features (44).
    history_len : int
        Number of historical timesteps (S).
    embed_dim : int
        Transformer embedding dimension.
    grid_size : int
        CNN output spatial grid side length (G). P = G^2 spatial tokens.
    n_layers : int
        Number of Transformer encoder layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_weather_channels=7, n_zones=8, cal_dim=44,
                 history_len=24, embed_dim=128, grid_size=8,
                 n_layers=4, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.n_zones = n_zones
        self.cal_dim = cal_dim
        self.history_len = history_len
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.n_spatial = grid_size * grid_size  # P
        self.future_len = 24
        self.total_steps = history_len + self.future_len
        self.tokens_per_step = self.n_spatial + 1  # P spatial + 1 tabular

        self.weather_cnn = WeatherCNN(n_weather_channels, embed_dim, grid_size)

        # Tabular embedding: demand (n_zones) + calendar (cal_dim) -> embed_dim
        self.hist_tabular_embed = nn.Linear(n_zones + cal_dim, embed_dim)
        self.future_tabular_embed = nn.Linear(n_zones + cal_dim, embed_dim)
        # Learnable mask vector for missing future demand
        self.demand_mask = nn.Parameter(torch.zeros(n_zones))

        # Positional embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.n_spatial, embed_dim) * 0.02
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, self.total_steps, embed_dim) * 0.02
        )
        # Learnable type token to distinguish spatial vs tabular
        self.tabular_type_embed = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.Sequential(*[
            EnergyTransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head: from each future tabular token -> n_zones
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, n_zones),
        )

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
        predictions : (B, 24, n_zones)
        """
        B = hist_weather.shape[0]
        S = self.history_len
        device = hist_weather.device

        # --- CNN: process all weather maps in parallel ---
        # Combine historical + future weather into one batch for CNN
        all_weather = torch.cat([hist_weather, future_weather], dim=1)  # (B, S+24, H, W, C)
        BT = B * (S + self.future_len)
        # Reshape to (B*T, C, H, W) for CNN
        all_weather_flat = all_weather.reshape(BT, 450, 449, 7).permute(0, 3, 1, 2)
        spatial_tokens = self.weather_cnn(all_weather_flat)  # (B*T, P, D)
        spatial_tokens = spatial_tokens.reshape(B, S + self.future_len, self.n_spatial, self.embed_dim)

        # Add spatial positional embedding to all spatial tokens
        spatial_tokens = spatial_tokens + self.spatial_pos_embed.unsqueeze(0)

        # --- Tabular tokens ---
        hist_tab_input = torch.cat([hist_energy, hist_cal], dim=-1)  # (B, S, n_zones+cal_dim)
        hist_tab_tokens = self.hist_tabular_embed(hist_tab_input).unsqueeze(2)  # (B, S, 1, D)

        future_demand_masked = self.demand_mask.unsqueeze(0).unsqueeze(0).expand(B, self.future_len, -1)
        future_tab_input = torch.cat([future_demand_masked, future_cal], dim=-1)
        future_tab_tokens = self.future_tabular_embed(future_tab_input).unsqueeze(2)  # (B, 24, 1, D)

        # Add tabular type embedding
        all_tab_tokens = torch.cat([hist_tab_tokens, future_tab_tokens], dim=1)  # (B, S+24, 1, D)
        all_tab_tokens = all_tab_tokens + self.tabular_type_embed

        # --- Assemble unified sequence ---
        # Per timestep: [P spatial tokens, 1 tabular token]
        # Shape: (B, S+24, P+1, D)
        all_tokens = torch.cat([spatial_tokens, all_tab_tokens], dim=2)

        # Add temporal positional embedding (broadcast over tokens within each step)
        temporal_pe = self.temporal_pos_embed.unsqueeze(2)  # (1, S+24, 1, D)
        all_tokens = all_tokens + temporal_pe

        # Flatten to (B, (S+24)*(P+1), D)
        seq = all_tokens.reshape(B, self.total_steps * self.tokens_per_step, self.embed_dim)
        seq = self.pos_drop(seq)

        # --- Transformer Encoder ---
        seq = self.blocks(seq)
        seq = self.norm(seq)

        # --- Extract future tabular tokens ---
        # Each timestep has (P+1) tokens; tabular token is the last one in each group.
        # Future timesteps are at positions S, S+1, ..., S+23
        future_tab_indices = []
        for t in range(S, S + self.future_len):
            tab_pos = t * self.tokens_per_step + self.n_spatial  # last token in group
            future_tab_indices.append(tab_pos)

        future_tab_indices = torch.tensor(future_tab_indices, device=device)
        future_states = seq[:, future_tab_indices, :]  # (B, 24, D)

        # --- Prediction ---
        predictions = self.head(future_states)  # (B, 24, n_zones)
        return predictions
