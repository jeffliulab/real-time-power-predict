"""
Fetch the past 24 hours of ISO-NE per-zone demand for the live demo.

Two sources, in priority order:

1. **Public genfuelmix API** at https://www.iso-ne.com/ws/wsclient
   (returns system-level demand). We split the total into the 8
   load zones using a fixed zonal-proportion vector estimated from
   the 2022 historical zonal data.

2. **Bundled CSV fallback** at `assets/sample_demand_2022.csv`.
   Used when the API is unreachable (CORS, rate limit, HF networking
   restrictions). Returns a representative 24-hour slice from
   2022-12-30/31 (the same window we use for self-eval).

Public API access for true per-zone real-time data requires an
authenticated ISO-NE account. The proportional split is a reasonable
approximation for a demo --- the model still sees real recent
ISO-NE-wide demand patterns; only the per-zone allocation is fixed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]

# Approximate zonal proportions of total ISO-NE demand,
# derived from 2022 historical zonal load reports.
# Sum is 1.0; values reflect typical share by zone.
ZONE_PROPORTIONS = np.array([
    0.064,  # ME
    0.080,  # NH
    0.045,  # VT
    0.205,  # CT
    0.070,  # RI
    0.130,  # SEMA
    0.115,  # WCMA
    0.291,  # NEMA_BOST  (largest --- Boston metro)
], dtype=np.float32)
assert abs(ZONE_PROPORTIONS.sum() - 1.0) < 1e-3

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_CSV = ASSETS_DIR / "sample_demand_2022.csv"

# In-memory cache: {timestamp_hash: (timestamp, ndarray)}
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

logger = logging.getLogger(__name__)


def _cache_key(end_dt: datetime) -> str:
    return end_dt.strftime("%Y-%m-%dT%H:00")


def _try_iso_ne_api(end_dt: datetime) -> Optional[np.ndarray]:
    """Attempt to fetch system demand from ISO-NE public endpoints.

    Returns (24, 8) array in MWh on success, or None on any failure.
    """
    try:
        url = "https://www.iso-ne.com/ws/wsclient"
        params = {
            "_nstmp_formDate": int(end_dt.timestamp() * 1000),
            "_nstmp_startDate": (end_dt - timedelta(hours=25)).strftime("%m/%d/%Y"),
            "_nstmp_endDate":   end_dt.strftime("%m/%d/%Y"),
            "_nstmp_chartName": "fuelmix",
        }
        r = requests.get(url, params=params, timeout=4)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        if "BeginDate" not in df.columns or "GenMw" not in df.columns:
            return None
        df["ts"] = pd.to_datetime(df["BeginDate"])
        hourly = df.groupby(df["ts"].dt.floor("h"))["GenMw"].sum().sort_index()
        last24 = hourly.tail(24).values.astype(np.float32)
        if len(last24) < 24:
            return None
        return _split_to_zones(last24)
    except Exception as e:  # noqa: BLE001
        logger.info("ISO-NE API fetch failed: %s", e)
        return None


def _split_to_zones(system_total: np.ndarray) -> np.ndarray:
    """system_total: (24,) -> (24, 8) using ZONE_PROPORTIONS."""
    return np.outer(system_total, ZONE_PROPORTIONS).astype(np.float32)


def _load_sample_csv() -> np.ndarray:
    """Fallback: read 24-hour slice from bundled CSV."""
    df = pd.read_csv(SAMPLE_CSV)
    arr = df[ZONE_COLS].tail(24).to_numpy(dtype=np.float32)
    if arr.shape != (24, 8):
        raise RuntimeError(f"Sample CSV has wrong shape {arr.shape}, expected (24, 8)")
    return arr


def fetch_recent_demand_mwh(end_dt: Optional[datetime] = None):
    """Fetch (24, 8) MWh array for the 24h ending at end_dt.

    Returns (array, source_label) where source_label is "live" if the
    API succeeded, "cached" if we used the in-memory cache, or
    "sample-2022" if we fell back to the bundled CSV.
    """
    if end_dt is None:
        end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    key = _cache_key(end_dt)
    cached = _CACHE.get(key)
    if cached is not None:
        ts, arr = cached
        if (datetime.now(timezone.utc) - ts).total_seconds() < _CACHE_TTL_SECONDS:
            return arr.copy(), "cached"

    arr = _try_iso_ne_api(end_dt)
    if arr is not None:
        _CACHE[key] = (datetime.now(timezone.utc), arr)
        return arr.copy(), "live"

    arr = _load_sample_csv()
    return arr, "sample-2022"


if __name__ == "__main__":
    arr, src = fetch_recent_demand_mwh()
    print(f"source: {src}")
    print(f"shape: {arr.shape}")
    print(f"per-zone first hour: {dict(zip(ZONE_COLS, arr[0]))}")
