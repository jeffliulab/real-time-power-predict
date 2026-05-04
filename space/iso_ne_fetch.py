"""
Fetch the past 24 hours of ISO-NE per-zone demand for the live demo.

Three sources, in priority order:

1. **EIA Open Data API** at https://api.eia.gov/v2/electricity/rto/region-data
   (system-level hourly demand, respondent=ISNE).  Free, requires a
   personal API key registered via https://www.eia.gov/opendata/register.php
   and exposed to the Space as the secret `EIA_API_KEY`.  We split the
   system total into the 8 ISO-NE zones using a fixed proportion
   vector estimated from 2022 zonal load reports.

2. **ISO-NE legacy `wsclient` endpoint**.  Tried as a backup; in
   practice it currently returns HTTP 500 from outside the IETF
   network, so it almost always falls through.

3. **Bundled CSV fallback** at `assets/sample_demand_2022.csv` (24 h)
   and `assets/sample_demand_2022_long.csv` (720 h).  Used when both
   live paths fail (no key configured, network down, rate-limited).

True per-zone real-time data requires an authenticated ISO Express
account. The proportional split is a reasonable demo approximation:
the model sees real recent ISO-NE-wide demand patterns; only the
per-zone allocation is fixed.
"""

from __future__ import annotations

import logging
import os
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
SAMPLE_CSV_LONG = ASSETS_DIR / "sample_demand_2022_long.csv"   # 720 h, 2022-12-02..12-31

# In-memory cache: {timestamp_hash: (timestamp, ndarray)}
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

logger = logging.getLogger(__name__)


def _cache_key(end_dt: datetime) -> str:
    return end_dt.strftime("%Y-%m-%dT%H:00")


EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def _try_eia_api(end_dt: datetime, hours: int = 24) -> Optional[np.ndarray]:
    """Fetch ISO-NE system demand from EIA Open Data.

    Requires the env var ``EIA_API_KEY`` (registered free at
    https://www.eia.gov/opendata/register.php and exposed to this
    Space as a Secret).

    Returns ``(hours, 8)`` MWh array on success, ``None`` on any failure
    (no key, HTTP error, missing rows, parse error).
    """
    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key:
        return None
    try:
        # EIA returns data on hour-ending convention; pull a generous
        # window so we can clip the freshest `hours` hours.
        start = (end_dt - timedelta(hours=hours + 6)).strftime("%Y-%m-%dT%H")
        end = end_dt.strftime("%Y-%m-%dT%H")
        params = {
            "api_key": key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ISNE",
            "facets[type][]": "D",        # 'D' = demand
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": hours + 24,
        }
        r = requests.get(EIA_API_URL, params=params, timeout=8)
        if r.status_code != 200:
            logger.info("EIA API HTTP %d: %s", r.status_code, r.text[:200])
            return None
        payload = r.json()
        rows = payload.get("response", {}).get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if "period" not in df.columns or "value" not in df.columns:
            return None
        df["ts"] = pd.to_datetime(df["period"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if len(df) < hours:
            return None
        last = df.tail(hours)["value"].to_numpy(dtype=np.float32)
        return _split_to_zones(last)
    except Exception as e:  # noqa: BLE001
        logger.info("EIA API fetch failed: %s", e)
        return None


def _try_iso_ne_api(end_dt: datetime) -> Optional[np.ndarray]:
    """Backup: ISO-NE legacy wsclient endpoint.

    Frequently returns HTTP 500 from outside their network, so this
    is mostly a fallback after EIA. Returns ``(24, 8)`` MWh or ``None``.
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

    arr = _try_eia_api(end_dt, hours=24)
    if arr is not None:
        _CACHE[key] = (datetime.now(timezone.utc), arr)
        return arr.copy(), "live (EIA)"

    arr = _try_iso_ne_api(end_dt)
    if arr is not None:
        _CACHE[key] = (datetime.now(timezone.utc), arr)
        return arr.copy(), "live (ISO-NE)"

    arr = _load_sample_csv()
    return arr, "sample-2022"


def fetch_long_history_mwh(end_dt: Optional[datetime] = None,
                            hours: int = 720):
    """Fetch a long per-zone demand history (default 720 h = 30 days) ending
    at end_dt, for use as Chronos-Bolt context.

    Strategy:
      1. Read the bundled long-history CSV (720 hourly rows from 2022-12).
      2. Splice in the 24 freshest hours from the live API / cache (so the
         tail of the history reflects recent live demand) when available.

    Returns:
      (array of shape (hours, 8), source_label).  source_label ends in
      "+live" when the tail 24 h came from the API, "+sample" otherwise.
    """
    if end_dt is None:
        end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # 1. Bundled long-history CSV (always present)
    if not SAMPLE_CSV_LONG.exists():
        # Fall back to short CSV repeated; less faithful but never crashes.
        short = _load_sample_csv()
        long_arr = np.tile(short, (hours // 24 + 1, 1))[:hours]
        return long_arr.astype(np.float32), "sample-2022-tiled"

    df = pd.read_csv(SAMPLE_CSV_LONG)
    long_arr = df[ZONE_COLS].tail(hours).to_numpy(dtype=np.float32)
    if long_arr.shape != (hours, 8):
        # Something odd; return what we have and tag.
        return long_arr.astype(np.float32), "sample-2022-short"

    # 2. Try to splice 24 freshest hours from the live API
    fresh = _try_iso_ne_api(end_dt)
    if fresh is not None and fresh.shape == (24, 8):
        long_arr[-24:] = fresh
        return long_arr, "sample-2022+live"
    return long_arr, "sample-2022"


if __name__ == "__main__":
    arr, src = fetch_recent_demand_mwh()
    print(f"recent (24 h): source={src}, shape={arr.shape}")
    long_arr, long_src = fetch_long_history_mwh()
    print(f"long ({len(long_arr)} h): source={long_src}, shape={long_arr.shape}")
