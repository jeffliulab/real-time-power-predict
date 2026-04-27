"""
ISO-NE real-time data fetcher (Part 3 placeholder).

Once Part 3 is implemented, this module will:
  - Fetch the last 24 hours of zonal demand from ISO Express API
  - Return a (24, 8) tensor of MWh values matching ZONE_COLS order
  - Cache results for ~5 minutes to avoid rate-limiting

API references:
  - ISO Express web services:
    https://www.iso-ne.com/isoexpress/web-services
  - Historical zonal data (CSV archive):
    https://www.iso-ne.com/isoexpress/web/reports/load-and-demand
  - Real-time hourly load (5-minute granularity, 1-hour aggregates):
    https://www.iso-ne.com/api/genfuelmix/current.json (system-level only)

Per-zone real-time data requires authenticated ISO-NE account; the public
JSON endpoints only expose system-level totals. Options for zonal real-time:

  1. Scrape https://www.iso-ne.com/isoexpress/ HTML dashboards
  2. Use an authenticated API token (via tts-research@tufts.edu)
  3. Approximate per-zone shares from system total + most recent
     historical zonal proportions (rough but functional)

For initial demo, option 3 is fine.
"""

from datetime import datetime, timedelta

import numpy as np


def fetch_last_24h_demand_mwh():
    """
    Fetch last 24 hours of ISO-NE zonal demand.

    Returns: (24, 8) numpy array, oldest hour first, MWh.
    """
    raise NotImplementedError(
        "ISO-NE real-time fetch not yet implemented. "
        "See module docstring for API references."
    )


def fetch_calendar_features(timestamps):
    """
    Build (T, 44) calendar feature matrix matching the dataset's encoding.

    44 = 24 (hour one-hot) + 7 (dow) + 12 (month) + 1 (US holiday).
    Reuses the holidays defined in training.data_preparation.dataset.
    """
    raise NotImplementedError("see space/model_utils.py for encoding")


if __name__ == "__main__":
    print(__doc__)
