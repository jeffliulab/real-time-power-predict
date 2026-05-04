# Dataset preparation — rebuild the ISO-NE + HRRR slice from public sources

This directory documents and provides **runnable skeleton scripts** to
reconstruct the training dataset (2019–2023) from public sources. The
canonical course-staff copy lives at
`/cluster/tufts/c26sp1cs0137/data/assignment3_data/` on Tufts HPC; if
that mirror disappears (or you want to reproduce on another cluster /
laptop), use these scripts.

## Final on-disk layout (what the model expects)

```
data/
  weather_data/
    2019/X_2019010100.pt   # torch.Tensor (450, 449, 7), float32
    2019/X_2019010101.pt
    ...
    2023/X_2023123123.pt
  energy_demand_data/
    2019.csv               # columns: timestamp_utc, ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST
    2020.csv
    ...
    2023.csv
```

- Weather tensor shape: `(450, 449, 7)` per hourly timestep.
  - 7 channels (in order): `TMP`, `RH`, `UGRD`, `VGRD`, `GUST`, `DSWRF`, `APCP_1hr`.
  - Geographic bounding box: New England, ~lat 40.5°–47.5°N, lon −74°–−66°W,
    regridded to 450 × 449 cells (~1.6 km/cell). See `_BBOX` in `fetch_hrrr_weather.py`.
- Demand CSV: 1 row per hour, 8 columns of MWh demand per ISO-NE zone.
  All timestamps are **UTC** (no DST shifts).

## Source datasets

### ISO New England demand
- **Public portal** (no authentication required for historical):
  <https://www.iso-ne.com/isoexpress/web/reports/load-and-demand>
- The "Hourly Real-Time and Day-Ahead LMPs and Demand" tables include
  per-zone hourly MWh demand for the 8 load zones for 2003–present.
  Files are CSV per month per zone.
- Our `fetch_iso_ne_demand.py` downloads these per-month CSVs, joins,
  and emits one `<year>.csv` per year matching the layout above.

### NOAA HRRR (High-Resolution Rapid Refresh)
- **Public archive** on AWS S3 (HRRRZARR project):
  <https://noaa-hrrr-bdp-pds.s3.amazonaws.com/index.html>
- Hourly grib2 files at 3 km native resolution, with 18-hour and 48-hour
  forecast horizons. We use the **f00 analysis** snapshot (instantaneous
  state at the valid time) as our hourly weather grid.
- Our `fetch_hrrr_weather.py` uses the **`herbie`** Python library
  (<https://herbie.readthedocs.io/>) which knows the NOAA archive layout,
  GRIB2 → xarray translation, and channel naming. We extract the 7
  channels named above, regrid to (450, 449) over the New England bbox,
  and save each hour as `X_YYYYMMDDHH.pt`.

## Files in this directory

| File | What it does |
|---|---|
| [`README.md`](README.md) | this document |
| [`requirements.txt`](requirements.txt) | extra Python deps (herbie-data, xarray, requests, pandas) |
| [`fetch_iso_ne_demand.py`](fetch_iso_ne_demand.py) | downloads + preprocesses 2019–2023 demand CSVs |
| [`fetch_hrrr_weather.py`](fetch_hrrr_weather.py) | downloads + regrids 2019–2023 HRRR to (450,449,7) tensors |
| [`build_dataset.py`](build_dataset.py) | aligns demand + weather, fills gaps, writes the canonical `data/` tree |

## End-to-end procedure

```bash
# 0. Install deps for data prep (separate from the main project requirements)
cd scripts/data_preparation
pip install -r requirements.txt

# 1. Demand: ~minutes, downloads ~80 MB of CSVs total
python fetch_iso_ne_demand.py --years 2019 2020 2021 2022 2023 --out ../../data/energy_demand_data

# 2. Weather: HOURS, downloads ~600 GB of HRRR analysis files,
#    extracts ~50 GB of regridded tensors. Run per year on a node
#    with >100 GB scratch space; uses herbie's S3 caching.
#    Estimated wall-time per year: 3-6 h (network-bound).
python fetch_hrrr_weather.py --year 2019 --out ../../data/weather_data
python fetch_hrrr_weather.py --year 2020 --out ../../data/weather_data
# ... repeat for 2021, 2022, 2023

# 3. Align + sanity-check
python build_dataset.py --data ../../data --years 2019 2020 2021 2022 2023
# Prints a summary table: per year, how many hours have BOTH weather + demand,
# how many are missing one or the other, and basic statistics.
```

## What the skeletons do NOT do (yet)

These scripts are **runnable skeletons**. They have full code paths
(arg-parsing, S3 fetching via `herbie`, regridding via `xarray`'s
`interp` to a target lat/lon grid, demand-CSV joins, gap-filling) but
have NOT been tested end-to-end on a multi-day window. Running the
full 2019–2023 download takes hours of network I/O and we did not
exhaust that test budget for this submission.

Things a re-implementer should validate before trusting the output:

1. **Channel naming and units.** The HRRR variable names we use
   (`TMP`, `RH`, `UGRD`, `VGRD`, `GUST`, `DSWRF`, `APCP_1hr`) match
   the GRIB2 short names from NOAA's HRRR documentation. Our model
   was trained on these exact channels in this exact order. If herbie
   returns different names (e.g., `t2m` instead of `TMP`), update the
   `_CHANNEL_MAP` in `fetch_hrrr_weather.py`.
2. **Grid resolution.** We re-grid to `(450, 449)` covering ~7° latitude
   × ~8° longitude. The original course-staff slice was at this exact
   resolution; verify the regridded output has the right corner
   coordinates (printed by `build_dataset.py`).
3. **Time alignment.** HRRR analysis files are stamped at top-of-hour UTC.
   ISO-NE demand uses Eastern Time historically; our fetch script
   converts to UTC explicitly. Cross-check by inspecting one day's
   demand CSV vs the corresponding weather files for consistent UTC
   timestamps.
4. **Normalization stats.** After rebuilding, recompute `norm_stats.pt`
   from the new dataset (don't reuse the cluster's). The model is
   sensitive to z-score statistics; a 5% drift in `weather_mean` /
   `weather_std` will visibly change MAPE.
5. **Holiday calendar.** `dataset.py` hard-codes the US federal
   holidays for 2019-2023. If you extend to 2024+, update the
   holiday set in `training/data_preparation/dataset.py`.

## Reproducibility statement

The scripts are deterministic given the same upstream archives. NOAA
HRRR's S3 bucket is content-addressable and immutable for the years
covered (2014-present). ISO-NE's demand CSVs are also archived
without revision once published. So a re-run should produce the same
weather tensors and demand values as the cluster's mirror, modulo any
floating-point regridding non-determinism.

The trained-model checkpoints in `pretrained_models/` are pinned to
the *cluster's* dataset; exact-bit reproduction of MAPE numbers
requires loading those checkpoints, not retraining on a re-fetched
dataset.
