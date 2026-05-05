"""
Real-time ISO-NE per-zone demand fetcher (no auth required).

Endpoint: https://www.iso-ne.com/transform/csv/fiveminuteestimatedzonalload
Returns 5-minute estimated load for all 8 ISO-NE zones; we roll up to
hourly (mean of 12 5-min observations) to match the model's input format.

Required trick: the endpoint returns HTTP 403 to direct curl, but accepts
the request once a session has visited a normal page first (cookie-prime
pattern, borrowed from the gridstatus.io library at
gridstatus/isone.py:_make_request).

Zone IDs (ISO-NE locational tags) -> our column names:
    4001 .Z.MAINE         -> ME
    4002 .Z.NEWHAMPSHIRE  -> NH
    4003 .Z.VERMONT       -> VT
    4004 .Z.CONNECTICUT   -> CT
    4005 .Z.RHODEISLAND   -> RI
    4006 .Z.SEMASS        -> SEMA
    4007 .Z.WCMASS        -> WCMA
    4008 .Z.NEMASSBOST    -> NEMA_BOST

Data publication delay is roughly 1 day: at 19:31 EDT today the CSV for
yesterday is fully populated; intra-day data may be missing recent hours
near the wall-clock present. The fetcher always asks for whole UTC days
and the caller is responsible for trimming to the desired range.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]

# ISO-NE locational identifiers (zone IDs in the public CSV)
_ZONE_ID_TO_COL = {
    4001: "ME",
    4002: "NH",
    4003: "VT",
    4004: "CT",
    4005: "RI",
    4006: "SEMA",
    4007: "WCMA",
    4008: "NEMA_BOST",
}

_ZONAL_URL = "https://www.iso-ne.com/transform/csv/fiveminuteestimatedzonalload"
_PRIME_URL = "https://www.iso-ne.com/isoexpress/web/reports/operations/-/tree/gen-fuel-mix"

logger = logging.getLogger(__name__)


def _new_session() -> requests.Session:
    """Return a requests.Session that has cookies primed for ISO-NE."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; predict-power/1.0; "
                      "https://github.com/jeffliulab/real-time-power-predict)",
    })
    s.get(_PRIME_URL, timeout=10)
    return s


def _parse_csv(text: str) -> pd.DataFrame:
    """Parse ISO-NE's quoted-CSV format (rows prefixed with C/H/D markers).

    Returns a DataFrame with columns: timestamp_utc, zone_id, zone_name,
    native_load_mw, btm_solar_mw.
    """
    data_rows = [line for line in text.splitlines() if line.startswith('"D"')]
    if not data_rows:
        raise RuntimeError("ISO-NE CSV had no data rows")
    parsed = list(csv.reader(data_rows, quotechar='"'))
    df = pd.DataFrame(parsed, columns=[
        "row_type", "datetime", "zone_id", "zone_name",
        "native_load_mw", "btm_solar_mw",
    ])
    # ISO-NE timestamps in the CSV are local time without TZ marker but
    # are documented as Eastern Prevailing Time. Localize then convert.
    ts_local = pd.to_datetime(df["datetime"]).dt.tz_localize(
        "US/Eastern", nonexistent="shift_forward", ambiguous="infer",
    )
    df["timestamp_utc"] = ts_local.dt.tz_convert("UTC").dt.tz_localize(None)
    df["zone_id"] = df["zone_id"].astype(int)
    df["native_load_mw"] = df["native_load_mw"].astype(float)
    return df[["timestamp_utc", "zone_id", "zone_name", "native_load_mw"]]


def fetch_one_day(date: datetime, session: Optional[requests.Session] = None,
                   timeout: int = 20) -> pd.DataFrame:
    """Fetch one calendar day of 5-minute per-zone estimated load.

    Args:
        date: any datetime; only the date portion (Eastern local) is used.
        session: optional pre-primed session for batched fetches.

    Returns:
        Wide DataFrame indexed by timestamp_utc with one column per zone
        (ME, NH, ..., NEMA_BOST), values in MWh-equivalent (5-min average MW
        which when multiplied by 5/60 hours equals MWh; we keep MW units
        and aggregate to hourly mean which numerically equals hourly MWh).
    """
    own_session = session is None
    if own_session:
        session = _new_session()
    date_str = date.strftime("%Y%m%d")
    url = f"{_ZONAL_URL}?start={date_str}&end={date_str}"
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(
            f"ISO-NE zonal fetch failed: HTTP {r.status_code} for {url}")
    if "text/csv" not in r.headers.get("Content-Type", "").lower():
        raise RuntimeError(
            f"ISO-NE zonal fetch returned non-CSV: {r.headers.get('Content-Type')}")
    long_df = _parse_csv(r.content.decode("utf8"))
    long_df["zone"] = long_df["zone_id"].map(_ZONE_ID_TO_COL)
    if long_df["zone"].isna().any():
        unknown = long_df.loc[long_df["zone"].isna(), "zone_id"].unique().tolist()
        raise RuntimeError(f"Unknown zone IDs in ISO-NE response: {unknown}")
    wide = long_df.pivot_table(
        index="timestamp_utc", columns="zone", values="native_load_mw",
        aggfunc="first")
    wide = wide[ZONE_COLS]            # canonical column order
    wide.index.name = "timestamp_utc"
    return wide


def fetch_range(start_date: datetime, end_date: datetime,
                  hourly: bool = True) -> pd.DataFrame:
    """Fetch 5-minute (or hourly-rolled) per-zone load over an inclusive
    date range [start_date, end_date].

    Args:
        start_date / end_date: datetimes; only the date portion is used.
            Both endpoints are inclusive.
        hourly: if True (default), aggregate 12 5-min bins per hour to
            the hourly mean (matches model input format). If False, return
            the raw 5-minute resolution.

    Returns:
        DataFrame with timestamp_utc index and 8 zone columns.
    """
    if start_date.tzinfo is not None:
        start_date = start_date.astimezone(timezone.utc).replace(tzinfo=None)
    if end_date.tzinfo is not None:
        end_date = end_date.astimezone(timezone.utc).replace(tzinfo=None)

    session = _new_session()
    parts = []
    cur = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        try:
            parts.append(fetch_one_day(cur, session=session))
        except Exception as e:  # noqa: BLE001
            logger.warning("ISO-NE fetch for %s failed: %s", cur.date(), e)
        cur += timedelta(days=1)

    if not parts:
        raise RuntimeError(
            f"ISO-NE fetch returned no data for range "
            f"{start_date.date()} -> {end_date.date()}")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if not hourly:
        return df

    hourly_df = df.resample("1h").mean(numeric_only=True)
    hourly_df = hourly_df[ZONE_COLS]
    return hourly_df


def fetch_recent_hours(end_dt: datetime, hours: int = 24,
                        max_lookback_days: int = 3
                        ) -> tuple[np.ndarray, datetime]:
    """Return ``(hours, 8)`` MW array of the most recent complete hours.

    ISO-NE 5-min zonal data has ~1-2 hour publication lag. This helper
    looks back from ``end_dt`` (rounded down to the hour) and finds the
    latest contiguous window of ``hours`` complete hours of per-zone data
    among the last ``max_lookback_days`` UTC dates.

    Returns:
        (array of shape (hours, 8) float32, latest_timestamp_in_window).

    Raises RuntimeError if there isn't a contiguous ``hours``-window in
    the last ``max_lookback_days``.
    """
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    end_dt = end_dt.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0, tzinfo=None)

    fetch_start = end_dt - timedelta(days=max_lookback_days)
    df = fetch_range(fetch_start, end_dt, hourly=True)
    df = df.dropna()                              # only fully-populated hours
    if len(df) < hours:
        raise RuntimeError(
            f"ISO-NE has only {len(df)} complete hourly rows in the last "
            f"{max_lookback_days} days; need {hours}.")

    # Find the latest contiguous `hours`-length stretch (1-hour gaps allowed
    # are NOT allowed here; we want strictly contiguous data).
    df = df.sort_index()
    contig_end = df.index[-1]
    contig_start = contig_end - timedelta(hours=hours - 1)
    window = df.loc[contig_start:contig_end]
    if len(window) != hours:
        raise RuntimeError(
            f"ISO-NE: last {hours} hours not contiguous "
            f"(got {len(window)} of {hours} expected, latest={contig_end}).")
    return window[ZONE_COLS].to_numpy(dtype=np.float32), contig_end.to_pydatetime()


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1))
    print(f"Fetching one day of ISO-NE per-zone load for "
          f"{yesterday.date()} (UTC)...")
    df = fetch_one_day(yesterday)
    print(f"  shape={df.shape}, columns={list(df.columns)}")
    print(f"  first row: {df.iloc[0].to_dict()}")
    print()
    print("Fetching last 24 contiguous hours...")
    arr, latest = fetch_recent_hours(datetime.now(timezone.utc), hours=24)
    print(f"  shape={arr.shape}, latest_timestamp={latest}")
    print(f"  sum_at_t0={arr.sum(axis=1)[0]:.0f} MW")
    print(f"  zone means: "
          + ", ".join(f"{z}={arr[:, i].mean():.0f}" for i, z in enumerate(ZONE_COLS)))
