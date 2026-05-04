"""
Cross-validate + summarize the assembled dataset.

Given a `data/` directory populated by:
    fetch_iso_ne_demand.py  --> data/energy_demand_data/<year>.csv
    fetch_hrrr_weather.py   --> data/weather_data/<year>/X_YYYYMMDDHH.pt

This script walks both halves and prints, per year:
    - count of hourly weather tensors present
    - count of hourly demand rows present
    - count of hours with BOTH (the trainable subset)
    - count of hours with weather-only / demand-only
    - basic statistics: per-channel weather mean/std (for sanity vs
      training-time norm_stats), per-zone demand mean/std (MWh)

USAGE
    python build_dataset.py --data ../../data --years 2019 2020 2021 2022 2023

This is a SANITY pass, not a re-run of training. After running this and
confirming no major gaps, you can compute fresh norm_stats with
`scripts/self_eval.py` style code OR just retrain (the training pipeline
recomputes norm_stats on first run from 500 random training samples).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
CHANNEL_NAMES = ["TMP", "RH", "UGRD", "VGRD", "GUST", "DSWRF", "APCP_1hr"]


def summarize_year(data_root: Path, year: int) -> dict:
    """Return a per-year summary dict."""
    weather_dir = data_root / "weather_data" / str(year)
    demand_csv = data_root / "energy_demand_data" / f"{year}.csv"

    # Demand
    demand_hours = set()
    if demand_csv.exists():
        df = pd.read_csv(demand_csv, parse_dates=["timestamp_utc"])
        demand_hours = {pd.Timestamp(t).floor("h") for t in df["timestamp_utc"]}
        demand_count = len(demand_hours)
    else:
        df = None
        demand_count = 0

    # Weather
    weather_hours = set()
    if weather_dir.exists():
        for f in weather_dir.glob("X_*.pt"):
            stem = f.stem  # X_YYYYMMDDHH
            try:
                ts = pd.Timestamp(stem[2:6] + "-" + stem[6:8] + "-" +
                                  stem[8:10] + " " + stem[10:12] + ":00")
                weather_hours.add(ts)
            except Exception:
                continue
    weather_count = len(weather_hours)

    both = weather_hours & demand_hours
    weather_only = weather_hours - demand_hours
    demand_only = demand_hours - weather_hours

    summary = {
        "year": year,
        "weather_count": weather_count,
        "demand_count": demand_count,
        "both": len(both),
        "weather_only": len(weather_only),
        "demand_only": len(demand_only),
        "total_hours_in_year": 8760 + (366 - 365 if year % 4 == 0 else 0),
    }

    # Sample 100 weather files for per-channel mean / std
    if weather_hours:
        sample = sorted(weather_hours)[::max(1, weather_count // 100)][:100]
        ch_means, ch_stds = [], []
        for ts in tqdm(sample, desc=f"  weather stats {year}", leave=False):
            path = weather_dir / f"X_{ts.strftime('%Y%m%d%H')}.pt"
            try:
                x = torch.load(path, weights_only=True).float()
                if x.ndim != 3 or x.shape[-1] != 7:
                    continue
                ch_means.append(x.mean(dim=(0, 1)).numpy())
                ch_stds.append(x.std(dim=(0, 1)).numpy())
            except Exception:
                continue
        if ch_means:
            summary["weather_mean_per_channel"] = \
                np.mean(np.stack(ch_means), axis=0).tolist()
            summary["weather_std_per_channel"] = \
                np.mean(np.stack(ch_stds), axis=0).tolist()

    # Demand stats
    if df is not None and len(df) > 0:
        for zone in ZONE_COLS:
            if zone in df.columns:
                summary[f"demand_mean_{zone}"] = float(df[zone].mean())
                summary[f"demand_std_{zone}"] = float(df[zone].std())

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True,
                   help="Path to the data/ root (must contain "
                        "weather_data/ and energy_demand_data/)")
    p.add_argument("--years", nargs="+", type=int, required=True,
                   help="Years to summarize")
    args = p.parse_args()

    print(f"\nDataset root: {args.data.resolve()}")
    print("=" * 80)

    summaries: list[dict] = []
    for year in args.years:
        print(f"\n--- {year} ---")
        s = summarize_year(args.data, year)
        summaries.append(s)
        print(f"  weather hours present: {s['weather_count']:>5} / {s['total_hours_in_year']:>5}")
        print(f"  demand  hours present: {s['demand_count']:>5} / {s['total_hours_in_year']:>5}")
        print(f"  BOTH (trainable):      {s['both']:>5}")
        print(f"  weather-only:          {s['weather_only']:>5}")
        print(f"  demand-only:           {s['demand_only']:>5}")
        if "weather_mean_per_channel" in s:
            print(f"  weather mean (sampled):")
            for name, mu, sigma in zip(CHANNEL_NAMES,
                                        s["weather_mean_per_channel"],
                                        s["weather_std_per_channel"]):
                print(f"    {name:<12} mu={mu:>+10.3f}  sigma={sigma:>10.3f}")
        for zone in ZONE_COLS:
            mu = s.get(f"demand_mean_{zone}")
            sigma = s.get(f"demand_std_{zone}")
            if mu is not None:
                print(f"  {zone:<10} demand: mu={mu:>7.0f} MWh  sigma={sigma:>7.0f}")

    # Aggregate
    print("\n" + "=" * 80)
    print("Aggregate (across all years):")
    total_both = sum(s["both"] for s in summaries)
    total_w = sum(s["weather_count"] for s in summaries)
    total_d = sum(s["demand_count"] for s in summaries)
    print(f"  total trainable hours: {total_both:,}")
    print(f"  total weather hours:   {total_w:,}")
    print(f"  total demand  hours:   {total_d:,}")
    print()
    if total_both > 0 and total_w > 0:
        completeness = 100.0 * total_both / total_w
        print(f"  weather-with-matched-demand completeness: {completeness:.1f} %")


if __name__ == "__main__":
    main()
