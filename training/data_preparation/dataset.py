"""
Energy demand forecasting dataset.

Loads hourly weather tensors and zonal energy demand CSVs, constructs
sliding-window samples for a seq2seq CNN-Transformer model.

Weather tensors are cached in memory per-year for efficient random access.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from collections import OrderedDict

US_HOLIDAYS = {
    (1, 1), (1, 20), (2, 17), (5, 26), (6, 19),
    (7, 4), (9, 1), (10, 13), (11, 11), (11, 27), (12, 25),
}

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
N_ZONES = len(ZONE_COLS)
FUTURE_LEN = 24
CAL_DIM = 24 + 7 + 12 + 1  # hour(24) + dow(7) + month(12) + holiday(1) = 44


def extract_calendar_features(timestamps):
    """
    Extract one-hot calendar features from an array of timestamps.

    Parameters
    ----------
    timestamps : np.ndarray of datetime64 or pd.DatetimeIndex, shape (T,)

    Returns
    -------
    np.ndarray, shape (T, 44), float32
    """
    dti = pd.DatetimeIndex(timestamps)
    T = len(dti)
    features = np.zeros((T, CAL_DIM), dtype=np.float32)

    hours = dti.hour
    dows = dti.dayofweek
    months = dti.month - 1

    features[np.arange(T), hours] = 1.0
    features[np.arange(T), 24 + dows] = 1.0
    features[np.arange(T), 31 + months] = 1.0

    for i, dt in enumerate(dti):
        if (dt.month, dt.day) in US_HOLIDAYS:
            features[i, 43] = 1.0

    return features


class WeatherCache:
    """
    LRU cache for weather tensors indexed by global time index.

    Keeps up to max_tensors tensors in memory.
    """

    def __init__(self, weather_dir, timestamps, max_tensors=2000):
        self.weather_dir = Path(weather_dir)
        self.timestamps = timestamps
        self.max_tensors = max_tensors
        self._cache = OrderedDict()

    def _get_path(self, t_idx):
        dt = pd.Timestamp(self.timestamps[t_idx])
        return self.weather_dir / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

    def get(self, t_idx):
        """Load and cache a weather tensor. Returns (450, 449, 7) or None."""
        if t_idx in self._cache:
            self._cache.move_to_end(t_idx)
            return self._cache[t_idx]

        path = self._get_path(t_idx)
        if not path.exists():
            return None
        x = torch.load(path, weights_only=True).float()
        if torch.isnan(x).any():
            return None

        self._cache[t_idx] = x
        if len(self._cache) > self.max_tensors:
            self._cache.popitem(last=False)
        return x

    def get_seq(self, start_idx, length):
        """Load a contiguous sequence. Returns (length, 450, 449, 7) or None."""
        frames = []
        for i in range(start_idx, start_idx + length):
            x = self.get(i)
            if x is None:
                return None
            frames.append(x)
        return torch.stack(frames)


class EnergyForecastDataset(Dataset):
    """
    Sliding-window dataset for energy demand forecasting.

    Each sample provides:
        - Historical weather tensors   (S, 450, 449, 7)
        - Historical energy demand     (S, n_zones)
        - Historical calendar features (S, cal_dim)
        - Future weather tensors       (24, 450, 449, 7)
        - Future calendar features     (24, cal_dim)
        - Target energy demand         (24, n_zones)

    Weather tensors are cached in an LRU cache (shared across samples) to
    avoid redundant disk reads for overlapping windows.
    """

    def __init__(self, data_root, years, history_len=24,
                 normalize=True, norm_stats=None, cache_size=2000):
        self.data_root = Path(data_root)
        self.weather_dir = self.data_root / "weather_data"
        self.energy_dir = self.data_root / "energy_demand_data"
        self.history_len = history_len
        self.normalize = normalize
        self.norm_stats = norm_stats

        self._load_energy_data()
        self._build_index(years)
        self.weather_cache = WeatherCache(
            self.weather_dir, self.timestamps, max_tensors=cache_size
        )

    def _load_energy_data(self):
        """Load and concatenate all energy demand CSVs."""
        dfs = []
        for csv_path in sorted(self.energy_dir.glob("target_energy_zonal_*.csv")):
            dfs.append(pd.read_csv(csv_path, parse_dates=["timestamp_utc"]))
        self.energy_df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)

        self.timestamps = self.energy_df["timestamp_utc"].values
        self.energy_values = self.energy_df[ZONE_COLS].values.astype(np.float32)
        self.hours_epoch = self.timestamps.astype("datetime64[h]").astype(np.int64)
        self.calendar_features = extract_calendar_features(self.timestamps)

    def _build_index(self, years):
        """Build list of valid sample indices."""
        ts_years = self.timestamps.astype("datetime64[Y]").astype(int) + 1970
        year_mask = np.isin(ts_years, years)
        candidates = np.where(year_mask)[0]

        self.samples = []
        for t_idx in candidates:
            hist_start = t_idx - self.history_len
            future_end = t_idx + FUTURE_LEN
            if hist_start < 0 or future_end > len(self.timestamps):
                continue
            self.samples.append(t_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_idx = self.samples[idx]
        hist_start = t_idx - self.history_len

        hist_slice = slice(hist_start, t_idx)
        future_slice = slice(t_idx, t_idx + FUTURE_LEN)

        hist_weather = self.weather_cache.get_seq(hist_start, self.history_len)
        if hist_weather is None:
            return None

        future_weather = self.weather_cache.get_seq(t_idx, FUTURE_LEN)
        if future_weather is None:
            return None

        hist_energy = torch.from_numpy(self.energy_values[hist_slice].copy())
        target_energy = torch.from_numpy(self.energy_values[future_slice].copy())

        hist_cal = torch.from_numpy(self.calendar_features[hist_slice].copy())
        future_cal = torch.from_numpy(self.calendar_features[future_slice].copy())

        if self.normalize and self.norm_stats is not None:
            ns = self.norm_stats
            hist_weather = (hist_weather - ns["weather_mean"]) / (ns["weather_std"] + 1e-7)
            future_weather = (future_weather - ns["weather_mean"]) / (ns["weather_std"] + 1e-7)
            hist_energy = (hist_energy - ns["energy_mean"]) / (ns["energy_std"] + 1e-7)
            target_energy = (target_energy - ns["energy_mean"]) / (ns["energy_std"] + 1e-7)

        return (hist_weather, hist_energy, hist_cal,
                future_weather, future_cal, target_energy)


class BlockShuffleSampler(torch.utils.data.Sampler):
    """
    Sampler that shuffles blocks of consecutive indices.

    Within each block, samples are sequential (cache-friendly for sliding
    windows). Blocks themselves are shuffled for epoch-level randomness.
    """

    def __init__(self, dataset_len, block_size=96):
        self.dataset_len = dataset_len
        self.block_size = block_size

    def __iter__(self):
        indices = list(range(self.dataset_len))
        blocks = [indices[i:i + self.block_size]
                  for i in range(0, len(indices), self.block_size)]
        np.random.shuffle(blocks)
        return iter([idx for block in blocks for idx in block])

    def __len__(self):
        return self.dataset_len


def collate_skip_none(batch):
    """Custom collate that filters out None samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def compute_norm_stats(data_root, years, history_len=24, n_samples=500, seed=42):
    """
    Compute normalization statistics from a subsample of the training data.

    Returns dict with:
        weather_mean/std: (1, 1, 1, 7) for broadcasting over (T, H, W, C)
        energy_mean/std:  (1, n_zones) for broadcasting over (T, n_zones)
    """
    ds = EnergyForecastDataset(data_root, years, history_len=history_len,
                               normalize=False, cache_size=2000)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    indices.sort()

    weather_sum = torch.zeros(7)
    weather_sq_sum = torch.zeros(7)
    energy_sum = torch.zeros(N_ZONES)
    energy_sq_sum = torch.zeros(N_ZONES)
    count = 0

    for i in indices:
        sample = ds[i]
        if sample is None:
            continue
        hist_weather, hist_energy, _, future_weather, _, target_energy = sample

        all_weather = torch.cat([hist_weather, future_weather], dim=0)
        w_mean = all_weather.mean(dim=(0, 1, 2))
        weather_sum += w_mean
        weather_sq_sum += (all_weather ** 2).mean(dim=(0, 1, 2))

        all_energy = torch.cat([hist_energy, target_energy], dim=0)
        e_mean = all_energy.mean(dim=0)
        energy_sum += e_mean
        energy_sq_sum += (all_energy ** 2).mean(dim=0)
        count += 1

        if (count % 100) == 0:
            print(f"  Norm stats: processed {count}/{min(n_samples, len(ds))} samples",
                  flush=True)

    weather_mean = weather_sum / count
    weather_std = torch.sqrt(weather_sq_sum / count - weather_mean ** 2)
    energy_mean = energy_sum / count
    energy_std = torch.sqrt(energy_sq_sum / count - energy_mean ** 2)

    return {
        "weather_mean": weather_mean.reshape(1, 1, 1, -1),
        "weather_std": weather_std.reshape(1, 1, 1, -1),
        "energy_mean": energy_mean.reshape(1, -1),
        "energy_std": energy_std.reshape(1, -1),
    }


def get_dataloaders(data_root, batch_size=4, history_len=24,
                    num_workers=0, train_years=None, val_years=None,
                    output_dir=None):
    """
    Build train and validation DataLoaders with normalization.

    num_workers defaults to 0 (single process) so the weather cache
    is shared across all samples efficiently.

    Returns: train_loader, val_loader, norm_stats
    """
    if train_years is None:
        train_years = [2019, 2020, 2021]
    if val_years is None:
        val_years = [2022]

    candidate_paths = []
    if output_dir is not None:
        candidate_paths.append(Path(output_dir) / "norm_stats.pt")
    candidate_paths.append(Path(data_root) / "norm_stats.pt")
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate_paths.append(project_root / "norm_stats.pt")

    norm_stats = None
    for p in candidate_paths:
        if p.exists():
            norm_stats = torch.load(p, weights_only=True)
            print(f"Loaded normalization stats from {p}")
            break

    if norm_stats is None:
        print("Computing normalization statistics ...")
        norm_stats = compute_norm_stats(data_root, train_years,
                                        history_len=history_len)
        for p in candidate_paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(norm_stats, p)
                print(f"Saved normalization stats to {p}")
                break
            except (PermissionError, OSError):
                continue

    train_ds = EnergyForecastDataset(data_root, train_years,
                                     history_len=history_len,
                                     norm_stats=norm_stats)
    val_ds = EnergyForecastDataset(data_root, val_years,
                                   history_len=history_len,
                                   norm_stats=norm_stats)

    loader_kwargs = dict(collate_fn=collate_skip_none, pin_memory=True)

    train_sampler = BlockShuffleSampler(len(train_ds), block_size=96)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, drop_last=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, **loader_kwargs)

    return train_loader, val_loader, norm_stats
