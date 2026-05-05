"""
Real-time HRRR weather fetcher for the predict-power Space.

This is the runtime counterpart to ``scripts/data_preparation/fetch_hrrr_weather.py``
(used to build the training set). It MUST produce arrays in the same
shape, channel order, and grid as training, otherwise the model sees an
out-of-distribution input. Specifically:

  - 7 channels in fixed order:
      [TMP_2m, RH_2m, UGRD_10m, VGRD_10m, GUST_surface, DSWRF_surface, APCP_1hr]
  - NE bbox: lat 40.5-47.5 N, lon -74.0 to -66.0 (West)
  - Regridded to 450 lat-rows x 449 lon-cols via xarray.interp(linear),
    NOT direct slicing of the native Lambert-Conformal grid

We fetch from the public ``noaa-hrrr-bdp-pds`` AWS S3 bucket via the
Herbie library (proven path; same as training).

Two top-level entry points:
  - ``fetch_history(end_dt, hours=24)`` returns ``(hours, 450, 449, 7)``,
    one f00 analysis per requested hour
  - ``fetch_forecast(cycle_dt, hours=24)`` returns ``(hours, 450, 449, 7)``,
    cycle_dt's f01..f{hours} forecast hours

Both paths are cached at ``/tmp/hrrr_cache/{cycle_YYYYMMDDHH}_f{NN}.npz``.
The cache survives within an HF Space uptime session and is wiped on sleep.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# === Match training pipeline EXACTLY ===
_BBOX = {"lat_min": 40.5, "lat_max": 47.5,
         "lon_min": -74.0, "lon_max": -66.0}
GRID_H = 450     # lat rows
GRID_W = 449     # lon cols
N_CHANNELS = 7

# Target lat/lon grid (geographic, not native HRRR Lambert-Conformal)
_LAT = np.linspace(_BBOX["lat_min"], _BBOX["lat_max"], GRID_H)
_LON = np.linspace(_BBOX["lon_min"], _BBOX["lon_max"], GRID_W)

# Channel definitions: (name, herbie searchString)
_CHANNELS: list[tuple[str, str]] = [
    ("TMP",      ":TMP:2 m above ground"),
    ("RH",       ":RH:2 m above ground"),
    ("UGRD",     ":UGRD:10 m above ground"),
    ("VGRD",     ":VGRD:10 m above ground"),
    ("GUST",     ":GUST:surface"),
    ("DSWRF",    ":DSWRF:surface"),
    ("APCP_1hr", ":APCP:surface:0-1 hour acc"),
]

CACHE_DIR = Path(os.environ.get("HRRR_CACHE_DIR", "/tmp/hrrr_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(cycle_dt: datetime, fxx: int) -> Path:
    return CACHE_DIR / f"{cycle_dt.strftime('%Y%m%d%H')}_f{fxx:02d}.npz"


def _hour_floor_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)


# --- regridding weights (computed lazily, then cached for the process) ---
# HRRR's native Lambert-Conformal grid is fixed across cycles, so we can
# precompute (mask, kdtree, weights, idxs) once from any sample dataset.
# Per-channel regrid is then a single matmul (~10 ms on cpu-basic).
_REGRID_CACHE: dict = {}


def _build_regrid_weights(lat2d: np.ndarray, lon2d_signed: np.ndarray):
    """Build cropping mask + 4-NN inverse-distance weights for our target grid.

    Returns dict with keys:
      - ``mask``: bool array (1059, 1799) selecting cells inside an NE
         bounding box that contains our target grid with ~1° margin
      - ``idxs``: (450*449, 4) int32 — indices into the masked source array
      - ``weights``: (450*449, 4) float32 — sums to 1 along axis=1
    """
    from scipy.spatial import cKDTree   # noqa: WPS433

    # Crop with margin so target-grid corners always have neighbors in source
    mask = ((lat2d >= _BBOX["lat_min"] - 1.5)
            & (lat2d <= _BBOX["lat_max"] + 1.5)
            & (lon2d_signed >= _BBOX["lon_min"] - 1.5)
            & (lon2d_signed <= _BBOX["lon_max"] + 1.5))
    if mask.sum() == 0:
        raise RuntimeError("Bounding-box mask is empty; HRRR grid mismatch?")

    src_pts = np.stack(
        [lat2d[mask].astype(np.float64),
         lon2d_signed[mask].astype(np.float64)],
        axis=-1)
    LL, LN = np.meshgrid(_LAT, _LON, indexing="ij")
    tgt_pts = np.stack([LL.ravel(), LN.ravel()], axis=-1)

    tree = cKDTree(src_pts)
    dists, idxs = tree.query(tgt_pts, k=4)
    # Inverse-distance weights, normalized
    inv_d = 1.0 / np.maximum(dists, 1e-9)
    w = (inv_d / inv_d.sum(axis=1, keepdims=True)).astype(np.float32)
    return {"mask": mask, "idxs": idxs.astype(np.int32), "weights": w}


def _regrid(field2d: np.ndarray, weights_pack: dict) -> np.ndarray:
    """Apply precomputed mask + weights to a (1059, 1799) HRRR field, return
    (450, 449) float32 on the regular lat/lon target grid."""
    cropped = field2d[weights_pack["mask"]].astype(np.float32)
    out = (cropped[weights_pack["idxs"]] * weights_pack["weights"]).sum(axis=1)
    return out.reshape(GRID_H, GRID_W)


def _fetch_one_via_herbie(cycle_dt: datetime, fxx: int) -> np.ndarray:
    """Fetch one (cycle, forecast-hour) pair, return (450, 449, 7) float32.

    Caller is responsible for caching; this function always hits the network.
    Raises RuntimeError on any failure.
    """
    try:
        from herbie import Herbie       # noqa: WPS433  (optional heavy dep)
    except ImportError as e:
        raise RuntimeError(
            f"hrrr_fetch.py requires herbie-data: {e}") from e

    H = Herbie(
        cycle_dt.strftime("%Y-%m-%d %H:00"),
        model="hrrr",
        product="sfc",
        fxx=fxx,
        verbose=False,
    )
    channels: list[np.ndarray] = []
    for ch_name, regex in _CHANNELS:
        try:
            # Newer Herbie (>=2024.x) renamed `searchString` to `search`
            ds = H.xarray(search=regex, verbose=False)
        except Exception as e:  # noqa: BLE001
            # APCP accumulation window varies with forecast hour:
            # f00 has no APCP, f01 has "0-1 hour acc" (matches our regex),
            # f02 has "0-2 hour acc" or "1-2 hour acc", etc. We zero-fill
            # any APCP fetch failure (the training mean is near zero in
            # MM units anyway, so post-z-score the model sees ~0).
            if ch_name == "APCP_1hr":
                logger.info("APCP_1hr unavailable at %s f%02d (%s); using zero",
                             cycle_dt, fxx,
                             type(e).__name__ if not str(e) else str(e)[:80])
                channels.append(np.zeros((GRID_H, GRID_W), dtype=np.float32))
                continue
            raise RuntimeError(
                f"Herbie xarray() failed for {ch_name} at "
                f"{cycle_dt.isoformat()} f{fxx:02d}: {e}") from e
        var = list(ds.data_vars)[0]
        arr = ds[var]
        field2d = np.squeeze(arr.values)
        if field2d.shape != (1059, 1799):
            raise RuntimeError(
                f"unexpected HRRR field shape {field2d.shape} for {ch_name}")

        # Initialize regrid weights once per process from the first dataset
        if "weights_pack" not in _REGRID_CACHE:
            lat2d = arr.coords["latitude"].values
            lon2d = arr.coords["longitude"].values
            lon2d_signed = np.where(lon2d > 180, lon2d - 360, lon2d)
            _REGRID_CACHE["weights_pack"] = _build_regrid_weights(
                lat2d, lon2d_signed)
            logger.info("Built HRRR -> NE-grid regrid weights "
                        "(one-time setup, ~0.3s)")

        regridded = _regrid(field2d, _REGRID_CACHE["weights_pack"])
        channels.append(regridded.astype(np.float32))

    tensor = np.stack(channels, axis=-1)
    if np.isnan(tensor).any():
        raise RuntimeError(
            f"NaN in regridded HRRR tensor for "
            f"{cycle_dt.isoformat()} f{fxx:02d}")
    return tensor


def _fetch_with_cache(cycle_dt: datetime, fxx: int) -> np.ndarray:
    """Fetch one (cycle, fxx) pair via cache or network."""
    p = _cache_path(cycle_dt, fxx)
    if p.exists():
        try:
            with np.load(p) as f:
                return f["weather"].astype(np.float32)
        except Exception:  # corrupt cache file, refetch
            p.unlink(missing_ok=True)
    tensor = _fetch_one_via_herbie(cycle_dt, fxx)
    # Store as float16 to halve disk usage (~2.8 MB/file vs 5.6 MB)
    np.savez_compressed(p, weather=tensor.astype(np.float16))
    return tensor


def _fetch_parallel(jobs: Sequence[tuple[datetime, int]],
                     parallel: int = 8,
                     progress: Optional[Callable[[int, int, str], None]] = None,
                     ) -> dict[tuple[datetime, int], np.ndarray]:
    """Fetch many (cycle_dt, fxx) pairs in parallel; return dict by job key."""
    if not jobs:
        return {}
    out: dict[tuple[datetime, int], np.ndarray] = {}
    if parallel <= 1:
        for i, (cdt, fxx) in enumerate(jobs):
            out[(cdt, fxx)] = _fetch_with_cache(cdt, fxx)
            if progress:
                progress(i + 1, len(jobs), f"{cdt.strftime('%Y-%m-%d %H')} f{fxx:02d}")
        return out

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_fetch_with_cache, cdt, fxx): (cdt, fxx)
                   for cdt, fxx in jobs}
        completed = 0
        for fut in as_completed(futures):
            key = futures[fut]
            out[key] = fut.result()
            completed += 1
            if progress:
                cdt, fxx = key
                progress(completed, len(jobs),
                         f"{cdt.strftime('%Y-%m-%d %H')} f{fxx:02d}")
    return out


# =====================================================================
#  Public API
# =====================================================================

def fetch_history(end_dt: datetime, hours: int = 24,
                   parallel: int = 8,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   ) -> np.ndarray:
    """Return ``(hours, 450, 449, 7)`` float32 of HRRR f00 analyses for
    the inclusive window ``[end_dt - hours, end_dt - 1h]``.

    Each requested valid-hour ``H`` uses cycle ``H`` with fxx=0 (i.e.,
    the analysis at that valid hour), matching how the training data
    was constructed.
    """
    end_dt = _hour_floor_utc(end_dt)
    valid_hours = [end_dt - timedelta(hours=hours - i) for i in range(hours)]
    jobs = [(vh, 0) for vh in valid_hours]
    fetched = _fetch_parallel(jobs, parallel=parallel, progress=progress)
    out = np.stack([fetched[(vh, 0)] for vh in valid_hours], axis=0)
    return out


# HRRR cycles with extended (0-48 h) forecasts. Other hourly cycles
# (01/02/04/05/...) only go out to f18, so we can't get 24 h from them.
LONG_CYCLE_HOURS = (0, 6, 12, 18)


def _latest_long_cycle_le(dt: datetime) -> datetime:
    """Return the most recent HRRR long cycle (00/06/12/18 UTC) <= dt."""
    dt = _hour_floor_utc(dt)
    while dt.hour not in LONG_CYCLE_HOURS:
        dt -= timedelta(hours=1)
    return dt


def fetch_forecast_for_window(target_start: datetime, hours: int = 24,
                                publication_lag_hours: int = 2,
                                parallel: int = 8,
                                progress: Optional[Callable[[int, int, str], None]] = None,
                                ) -> tuple[np.ndarray, datetime, int]:
    """Return ``(hours, 450, 449, 7)`` covering valid hours
    ``[target_start, target_start + hours - 1]``, using the most recent
    HRRR long cycle (one of 00/06/12/18 UTC) that was published before
    ``target_start`` (with ``publication_lag_hours`` margin to allow for
    cycle processing delay).

    Returns ``(weather, cycle_dt, fxx_start)`` so the caller can log
    which cycle was used.
    """
    target_start = _hour_floor_utc(target_start)
    cutoff = target_start - timedelta(hours=publication_lag_hours)
    cycle_dt = _latest_long_cycle_le(cutoff)
    fxx_start = int((target_start - cycle_dt).total_seconds() / 3600)
    jobs = [(cycle_dt, fxx) for fxx in range(fxx_start, fxx_start + hours)]
    fetched = _fetch_parallel(jobs, parallel=parallel, progress=progress)
    out = np.stack([fetched[(cycle_dt, fxx)]
                    for fxx in range(fxx_start, fxx_start + hours)], axis=0)
    return out, cycle_dt, fxx_start


def fetch_forecast(cycle_dt: datetime, hours: int = 24,
                    parallel: int = 8,
                    progress: Optional[Callable[[int, int, str], None]] = None,
                    ) -> np.ndarray:
    """Backwards-compat wrapper: fetch f01..f{hours} from a specific cycle.

    NOTE: only long cycles (00/06/12/18 UTC) reliably cover 24+ hours.
    For automatic cycle selection, prefer ``fetch_forecast_for_window``.
    """
    cycle_dt = _hour_floor_utc(cycle_dt)
    jobs = [(cycle_dt, fxx) for fxx in range(1, hours + 1)]
    fetched = _fetch_parallel(jobs, parallel=parallel, progress=progress)
    out = np.stack([fetched[(cycle_dt, fxx)] for fxx in range(1, hours + 1)],
                   axis=0)
    return out


def latest_available_cycle(target_dt: datetime,
                             max_lookback_hours: int = 4,
                             ) -> datetime:
    """Find the most recent HRRR cycle <= ``target_dt`` whose forecast
    hours appear to be on S3 (HRRR has ~1-2 hour publication lag).

    We probe by trying to instantiate Herbie for each cycle from
    ``target_dt`` backwards, succeeding when ``H.grib`` resolves.
    Returns the cycle datetime (UTC, hour-floored, naive).
    """
    target_dt = _hour_floor_utc(target_dt)
    try:
        from herbie import Herbie       # noqa: WPS433
    except ImportError as e:
        raise RuntimeError(f"herbie-data not installed: {e}") from e

    for back in range(0, max_lookback_hours + 1):
        cdt = target_dt - timedelta(hours=back)
        try:
            H = Herbie(cdt.strftime("%Y-%m-%d %H:00"),
                       model="hrrr", product="sfc", fxx=1, verbose=False)
            if H.grib is not None:
                return cdt
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(
        f"No HRRR cycle available within last {max_lookback_hours}h of "
        f"{target_dt.isoformat()}")


if __name__ == "__main__":
    # Smoke test: fetch one f00 + one f01 from yesterday's noon cycle
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    yesterday_noon = (datetime.now(timezone.utc) - timedelta(days=1)
                       ).replace(hour=12, minute=0, second=0, microsecond=0,
                                  tzinfo=None)

    print(f"Smoke test cycle: {yesterday_noon} UTC")
    arr = _fetch_with_cache(yesterday_noon, 0)
    print(f"  f00: shape={arr.shape}, dtype={arr.dtype}, "
          f"mean per channel: " + ", ".join(
              f"{name}={arr[..., i].mean():.2f}" for i, (name, _) in enumerate(_CHANNELS)))
    arr1 = _fetch_with_cache(yesterday_noon, 1)
    print(f"  f01: shape={arr1.shape}, dtype={arr1.dtype}, "
          f"mean per channel: " + ", ".join(
              f"{name}={arr1[..., i].mean():.2f}" for i, (name, _) in enumerate(_CHANNELS)))
    print(f"  cache dir: {CACHE_DIR}, n files: {len(list(CACHE_DIR.glob('*.npz')))}")
