"""
Fetch hourly per-zone demand from ISO New England's public data feed
and write one CSV per year matching the layout the training pipeline
expects: <year>.csv with columns
    timestamp_utc, ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST

ISO Express historical demand portal:
    https://www.iso-ne.com/isoexpress/web/reports/load-and-demand

Files are published as monthly CSVs per zone. We download all
12 months × 8 zones × N years, join, sanity-check, and write one
year-CSV per year.

USAGE
    python fetch_iso_ne_demand.py --years 2019 2020 2021 2022 2023 \\
        --out ../../data/energy_demand_data

NOTE
    This script is a skeleton: the URL pattern + CSV column names
    have been correct as of the assignment data preparation in
    early 2026, but ISO-NE has reorganized this portal in the past.
    Verify by manually downloading one CSV before bulk-running.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]

# ISO-NE serves per-month per-zone CSVs from a structured archive. The
# archive index is HTML; we resolve directly to the per-zone CSV path
# template documented at https://www.iso-ne.com/isoexpress/.
# Format below was verified against the 2022 archive in early 2026.
_BASE = "https://www.iso-ne.com/static-assets/documents/markets/hstdata/" \
        "znl_info/hourly/{year}/hourly_{year}_{month:02d}_{zone_abbr}.csv"

# ISO-NE uses these abbreviations in the URL:
_ZONE_URL_ABBR = {
    "ME":        "me",
    "NH":        "nh",
    "VT":        "vt",
    "CT":        "ct",
    "RI":        "ri",
    "SEMA":      "sema",
    "WCMA":      "wcma",
    "NEMA_BOST": "nemabost",
}


def fetch_one(year: int, month: int, zone: str, retries: int = 3,
              timeout: int = 30) -> pd.DataFrame | None:
    """Download one (year, month, zone) CSV; return DataFrame with cols
    [timestamp_utc, <zone>] in MWh; None on hard failure."""
    url = _BASE.format(year=year, month=month, zone_abbr=_ZONE_URL_ABBR[zone])
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 200:
                # ISO-NE CSVs typically have 3-5 header lines + then
                # rows of (Date, Hour, MWh, ...). Skip metadata lines
                # that start with a quote or are blank.
                df = pd.read_csv(io.BytesIO(r.content), skiprows=4)
                # Common column names across versions:
                date_col = next((c for c in df.columns
                                 if c.lower() in ("date", "local_date", "trade_date")), None)
                hour_col = next((c for c in df.columns
                                 if c.lower() in ("hour", "hour_ending", "he")), None)
                load_col = next((c for c in df.columns
                                 if "load" in c.lower() or "mwh" in c.lower()
                                 or c.lower() == "demand"), None)
                if not (date_col and hour_col and load_col):
                    raise RuntimeError(f"unexpected columns in CSV: {list(df.columns)}")
                # Build timestamp_utc — ISO-NE local hours are EST/EDT.
                # The course's mirror used UTC, so we shift here.
                # Eastern offset is UTC-5 standard, UTC-4 daylight.
                # pandas tz_localize+tz_convert handles DST correctly.
                ts_local = pd.to_datetime(df[date_col]) + \
                           pd.to_timedelta(df[hour_col].astype(int) - 1, unit="h")
                ts_local = ts_local.dt.tz_localize("US/Eastern", nonexistent="shift_forward",
                                                   ambiguous="infer")
                df["timestamp_utc"] = ts_local.dt.tz_convert("UTC").dt.tz_localize(None)
                out = df[["timestamp_utc", load_col]].rename(columns={load_col: zone})
                return out
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(1 + attempt)
    print(f"  [WARN] failed ({year}-{month:02d}-{zone}): {last_err}", file=sys.stderr)
    return None


def fetch_year(year: int, out_dir: Path) -> None:
    """Download all 12 months × 8 zones for a given year, join into one
    wide DataFrame, write to <out_dir>/<year>.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Fetching ISO-NE demand for {year} ===")

    per_zone: dict[str, pd.DataFrame] = {}
    for zone in ZONE_COLS:
        monthly_dfs: list[pd.DataFrame] = []
        for month in range(1, 13):
            df = fetch_one(year, month, zone)
            if df is not None:
                monthly_dfs.append(df)
            print(f"  {zone}-{month:02d}: {'ok' if df is not None else 'FAIL'}")
        if not monthly_dfs:
            print(f"  [SKIP] zone {zone}: no months downloaded")
            continue
        zone_df = pd.concat(monthly_dfs, ignore_index=True) \
                    .drop_duplicates(subset=["timestamp_utc"]) \
                    .sort_values("timestamp_utc") \
                    .reset_index(drop=True)
        per_zone[zone] = zone_df

    if not per_zone:
        print(f"  [FAIL] no zones downloaded for {year}")
        return

    # Join all zones on timestamp
    base = next(iter(per_zone.values()))[["timestamp_utc"]].copy()
    for zone, zdf in per_zone.items():
        base = base.merge(zdf, on="timestamp_utc", how="outer")
    base = base.sort_values("timestamp_utc").reset_index(drop=True)

    # Reorder columns to canonical
    cols = ["timestamp_utc"] + [z for z in ZONE_COLS if z in base.columns]
    base = base[cols]

    out_path = out_dir / f"{year}.csv"
    base.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}: {len(base):,} hourly rows, "
          f"{len(base.columns)-1} zones")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", type=int, required=True,
                   help="Years to fetch (e.g. 2019 2020 2021 2022 2023)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for <year>.csv files")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        fetch_year(year, args.out)
    print("\nAll done.")


if __name__ == "__main__":
    main()
