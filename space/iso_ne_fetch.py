"""
High-level ISO-NE per-zone demand fetcher for the Space.

Wraps the low-level fetcher in ``iso_ne_zonal.py`` with:

  - In-memory cache (5-minute TTL) so repeated clicks within a few
    minutes don't refetch from ISO-NE
  - Optional bundled CSV fallback for offline / API-down scenarios
  - Optional integration with a long-history CSV pulled from the data
    repo at Space startup (used to seed Chronos context without
    re-fetching 30 days of ISO-NE on every click)

Public API kept stable so ``app.py`` can swap from the old EIA-based
implementation without further changes:

  - ``ZONE_COLS``                          : list of 8 zone names
  - ``fetch_recent_demand_mwh(end_dt)``    : (24, 8) MWh + source label
  - ``fetch_long_history_mwh(end_dt, hours=720)`` : (hours, 8) MWh + label
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from iso_ne_zonal import ZONE_COLS, fetch_range, fetch_recent_hours

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_CSV = ASSETS_DIR / "sample_demand_2022.csv"
SAMPLE_CSV_LONG = ASSETS_DIR / "sample_demand_2022_long.csv"

# In-memory cache: { ("recent", end_hour) | ("long", end_hour, hours) -> (ts, np.ndarray) }
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 300

# Path of the data-repo 30-day CSV (refreshed daily by GitHub Actions in
# new-england-real-time-power-predict-data; downloaded by app.py at
# startup and saved to /tmp). When present, fetch_long_history_mwh
# uses it as the base and splices in the last 1-2 days from live API.
DATA_REPO_30D_CSV_PATH = Path(os.environ.get(
    "DATA_REPO_30D_CSV_PATH", "/tmp/iso_ne_30d.csv"))


def _hour_floor_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0, tzinfo=None)


def _cache_get(key: tuple) -> Optional[np.ndarray]:
    cached = _CACHE.get(key)
    if cached is None:
        return None
    ts, arr = cached
    if (datetime.now(timezone.utc) - ts).total_seconds() < _CACHE_TTL_SECONDS:
        return arr.copy()
    return None


def _cache_put(key: tuple, arr: np.ndarray) -> None:
    _CACHE[(key)] = (datetime.now(timezone.utc), arr.copy())


def _load_sample_recent() -> np.ndarray:
    df = pd.read_csv(SAMPLE_CSV)
    arr = df[ZONE_COLS].tail(24).to_numpy(dtype=np.float32)
    if arr.shape != (24, 8):
        raise RuntimeError(
            f"Bundled sample_demand_2022.csv has wrong shape {arr.shape}")
    return arr


def _load_sample_long(hours: int) -> np.ndarray:
    if SAMPLE_CSV_LONG.exists():
        df = pd.read_csv(SAMPLE_CSV_LONG)
        arr = df[ZONE_COLS].tail(hours).to_numpy(dtype=np.float32)
        if arr.shape == (hours, 8):
            return arr
    short = _load_sample_recent()
    return np.tile(short, (hours // 24 + 1, 1))[:hours].astype(np.float32)


def fetch_recent_demand_mwh(end_dt: Optional[datetime] = None
                              ) -> tuple[np.ndarray, str]:
    """Return ``(24, 8)`` MWh for the most recent 24 contiguous hours
    ending at ``end_dt`` (or now). Source label is one of:
      - ``"live (ISO-NE 5-min zonal -> hourly)"``
      - ``"cached"``
      - ``"sample-2022"``
    """
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    end_dt = _hour_floor_utc(end_dt)
    cache_key = ("recent", end_dt)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached, "cached"
    try:
        arr, latest = fetch_recent_hours(end_dt, hours=24)
        _cache_put(cache_key, arr)
        lag_hours = (end_dt - latest).total_seconds() / 3600
        label = f"live (ISO-NE 5-min zonal, latest hour {latest.isoformat()}, "
        label += f"lag {lag_hours:.0f}h)" if lag_hours > 0 else f"live (ISO-NE 5-min zonal)"
        return arr, label
    except Exception as e:  # noqa: BLE001
        logger.warning("ISO-NE realtime fetch failed: %s; falling back to bundled CSV", e)
        return _load_sample_recent(), "sample-2022 (ISO-NE unreachable)"


def _load_30d_base() -> Optional[pd.DataFrame]:
    """Load data-repo's pre-built 30-day per-zone CSV if available."""
    if not DATA_REPO_30D_CSV_PATH.exists():
        return None
    try:
        df = pd.read_csv(DATA_REPO_30D_CSV_PATH, parse_dates=["timestamp_utc"])
        df = df.set_index("timestamp_utc").sort_index()
        return df[ZONE_COLS]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load 30d base CSV at %s: %s",
                        DATA_REPO_30D_CSV_PATH, e)
        return None


def fetch_long_history_mwh(end_dt: Optional[datetime] = None,
                             hours: int = 720
                             ) -> tuple[np.ndarray, str]:
    """Return ``(hours, 8)`` MWh of per-zone history ending at ``end_dt - 1h``.

    Strategy:
      1. If the data repo's 30d base CSV is present, start from it.
      2. Otherwise fall back to the bundled long-history CSV.
      3. Always splice the last ~24-48 hours from the live ISO-NE API
         so the tail is fresh.
    """
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    end_dt = _hour_floor_utc(end_dt)
    cache_key = ("long", end_dt, hours)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached, "cached"

    target_end = end_dt - timedelta(hours=1)        # last hour we want
    target_start = target_end - timedelta(hours=hours - 1)

    base = _load_30d_base()
    base_label = "data-repo 30d"

    if base is None:
        long_arr = _load_sample_long(hours)
        out = long_arr
        _cache_put(cache_key, out)
        return out, "sample-2022 (no data-repo CSV)"

    # Try to splice live ISO-NE for the last 2 days for freshness
    splice_label = ""
    try:
        live = fetch_range(target_end - timedelta(days=2), target_end,
                            hourly=True)
        # Overwrite overlapping rows in `base` with `live`
        base.update(live)
        splice_label = " + live splice"
    except Exception as e:  # noqa: BLE001
        logger.info("Live splice into long history failed: %s", e)

    # Ensure we have continuous coverage; if base doesn't reach target_start,
    # fall back to bundled long CSV for the missing tail
    if base.index.min() > target_start:
        logger.info("30d base starts at %s, missing %s -> %s; padding from sample",
                     base.index.min(), target_start, base.index.min())
        sample_long = _load_sample_long(hours)
        out = sample_long
    else:
        # Slice exact window
        idx = pd.date_range(start=target_start, end=target_end, freq="1h")
        sliced = base.reindex(idx)
        if sliced.isna().any().any():
            logger.info("30d base has %d NaN rows in window; interpolating",
                         int(sliced.isna().any(axis=1).sum()))
            sliced = sliced.interpolate(method="time", limit=12).ffill().bfill()
        out = sliced[ZONE_COLS].to_numpy(dtype=np.float32)

    _cache_put(cache_key, out)
    return out, base_label + splice_label
