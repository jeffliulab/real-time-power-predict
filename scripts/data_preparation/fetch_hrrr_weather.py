"""
Fetch hourly HRRR f00 (analysis) snapshots, extract 7 channels over the
New England bbox, regrid to 450 × 449, and save one .pt file per hour.

USAGE
    python fetch_hrrr_weather.py --year 2022 --out ../../data/weather_data
    # writes ../../data/weather_data/2022/X_2022010100.pt ... X_2022123123.pt

REQUIRES
    pip install -r requirements.txt   # herbie-data, xarray, cfgrib

NOTES
- HRRR archive on AWS S3:
    s3://noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2
- We use the **f00 analysis** (instantaneous state at the valid hour),
  not the forecast. This matches what the model was trained on.
- 7 channels in fixed order:
    [TMP, RH, UGRD, VGRD, GUST, DSWRF, APCP_1hr]
  - TMP    : 2-meter air temperature (K)
  - RH     : 2-meter relative humidity (%)
  - UGRD   : 10-meter U wind component (m/s)
  - VGRD   : 10-meter V wind component (m/s)
  - GUST   : surface wind gust (m/s)
  - DSWRF  : downward shortwave radiation flux (W/m^2)
  - APCP_1hr : 1-hour accumulated precipitation (kg/m^2)
- Bounding box (New England):
    lat: 40.5 -- 47.5 N  (size 7.0 deg)
    lon: -74.0 -- -66.0 (size 8.0 deg)
  Regridded to 450 lat-rows × 449 lon-cols (so ~1.6 km per cell).

VERIFY before bulk run:
1. Run with --year 2022 --month 6 --day 1 to fetch ONE hour and inspect
   the resulting .pt with `torch.load(...).shape == (450, 449, 7)`.
2. Cross-check the 7th channel (APCP_1hr) against NOAA's HRRR docs;
   accumulation periods can differ between HRRR product versions.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from herbie import Herbie
    import xarray as xr
except ImportError as e:
    print("This script needs herbie-data + xarray. Install via:\n"
          "  pip install -r requirements.txt", file=sys.stderr)
    raise

# === Geographic bounding box for NE US (matches course-staff slice) ===
_BBOX = {
    "lat_min": 40.5,
    "lat_max": 47.5,
    "lon_min": -74.0,
    "lon_max": -66.0,
}
_GRID_H = 450     # lat rows
_GRID_W = 449     # lon cols

# Target lat/lon grid (geographic, not native HRRR lambert-conformal)
_LAT = np.linspace(_BBOX["lat_min"], _BBOX["lat_max"], _GRID_H)
_LON = np.linspace(_BBOX["lon_min"], _BBOX["lon_max"], _GRID_W)


# Channel mapping: GRIB short name -> (xarray var, level)
# These match herbie's `subset(...)` API. If herbie returns
# different names, adjust here. Order matters: defines the C dim.
_CHANNELS = [
    # (label,           grib_search_string)
    ("TMP",      ":TMP:2 m above ground"),
    ("RH",       ":RH:2 m above ground"),
    ("UGRD",     ":UGRD:10 m above ground"),
    ("VGRD",     ":VGRD:10 m above ground"),
    ("GUST",     ":GUST:surface"),
    ("DSWRF",    ":DSWRF:surface"),
    ("APCP_1hr", ":APCP:surface:0-1 hour acc"),
]


def fetch_one_hour(dt: datetime) -> torch.Tensor | None:
    """Fetch HRRR f00 analysis at the given UTC hour and return a
    (450, 449, 7) float32 tensor over the NE bbox.

    Returns None on any failure (download, GRIB parse, missing variable).
    """
    try:
        H = Herbie(
            dt.strftime("%Y-%m-%d %H:00"),
            model="hrrr",
            product="sfc",        # surface fields (wrfsfcf*)
            fxx=0,                # f00 analysis
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [Herbie ctor failed] {dt}: {e}", file=sys.stderr)
        return None

    channels = []
    for ch_name, regex in _CHANNELS:
        try:
            ds = H.xarray(searchString=regex, verbose=False)
            # ds is a single-variable xarray.Dataset; pick the first data var
            varname = list(ds.data_vars)[0]
            arr = ds[varname]
            # arr is on HRRR's native lambert grid; regrid to lat/lon.
            # We rely on `latitude` and `longitude` coords (HRRR has them
            # as 2D fields). Use scipy/xarray's interp which handles 2D.
            regridded = arr.interp(
                latitude=xr.DataArray(_LAT, dims=["lat"]),
                longitude=xr.DataArray(_LON, dims=["lon"]),
                method="linear",
            ).values
            # Shape may be (1, H, W) due to time dim — squeeze
            regridded = np.squeeze(regridded)
            if regridded.shape != (_GRID_H, _GRID_W):
                # Handle (W, H) transpose if interp returned that order
                if regridded.shape == (_GRID_W, _GRID_H):
                    regridded = regridded.T
                else:
                    print(f"  [shape mismatch] {ch_name} {dt}: "
                          f"{regridded.shape}, expected ({_GRID_H}, {_GRID_W})",
                          file=sys.stderr)
                    return None
            channels.append(regridded.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            print(f"  [channel fetch failed] {ch_name} {dt}: {e}",
                  file=sys.stderr)
            return None

    # Stack into (450, 449, 7)
    tensor = np.stack(channels, axis=-1)
    if np.isnan(tensor).any():
        print(f"  [NaN in tensor] {dt}", file=sys.stderr)
        return None
    return torch.from_numpy(tensor)


def fetch_year(year: int, out_dir: Path,
               start_hour: int = 0, end_hour: int | None = None,
               skip_existing: bool = True) -> None:
    """Fetch every hour of the given year and write to <out_dir>/<year>/."""
    year_dir = out_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1) if end_hour is None else (
        datetime(year, 1, 1) + timedelta(hours=end_hour))
    total_hours = int((end - start).total_seconds() / 3600)

    print(f"\n=== Fetching HRRR f00 for {year}: "
          f"{total_hours} hours --> {year_dir} ===")

    skipped = succeeded = failed = 0
    pbar = tqdm(range(total_hours), desc=f"HRRR {year}", unit="hr")
    for i in pbar:
        if i < start_hour:
            continue
        dt = start + timedelta(hours=i)
        out_path = year_dir / f"X_{dt.strftime('%Y%m%d%H')}.pt"
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        tensor = fetch_one_hour(dt)
        if tensor is None:
            failed += 1
            continue
        torch.save(tensor, out_path)
        succeeded += 1
        pbar.set_postfix(ok=succeeded, fail=failed, skip=skipped)

    print(f"\n  succeeded={succeeded}  failed={failed}  skipped={skipped}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True,
                   help="Year to fetch (e.g. 2022)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output base dir (subdir <year>/ will be created)")
    p.add_argument("--start-hour", type=int, default=0,
                   help="Resume from this hour-of-year (for partial reruns)")
    p.add_argument("--end-hour", type=int, default=None,
                   help="Stop at this hour-of-year (for testing)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-fetch even if .pt exists already")
    args = p.parse_args()

    fetch_year(args.year, args.out,
               start_hour=args.start_hour,
               end_hour=args.end_hour,
               skip_existing=not args.no_skip_existing)


if __name__ == "__main__":
    main()
