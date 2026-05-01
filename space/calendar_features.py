"""
Build the 44-d calendar one-hot used by the demand-forecasting model.

Layout (matches training/data_preparation/dataset.py):
  hour-of-day one-hot (24)
  + day-of-week one-hot (7)
  + month one-hot (12)
  + US holiday flag (1)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import numpy as np

CAL_DIM = 44

# US federal holidays for 2022-2026 (date-only, year-agnostic match below).
# Encoded as (month, day) tuples for fixed-date holidays plus a small set
# of moving holidays we hardcode by date.
_FIXED_HOLIDAYS_MD = {
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas
    (6, 19),   # Juneteenth
}
_MOVING_HOLIDAYS = {
    # MLK Day (3rd Mon Jan), Presidents' Day (3rd Mon Feb),
    # Memorial Day (last Mon May), Labor Day (1st Mon Sep),
    # Columbus (2nd Mon Oct), Thanksgiving (4th Thu Nov)
    # Pre-computed for 2022-2026.
    (2022, 1, 17), (2022, 2, 21), (2022, 5, 30), (2022, 9, 5),
    (2022, 10, 10), (2022, 11, 24),
    (2023, 1, 16), (2023, 2, 20), (2023, 5, 29), (2023, 9, 4),
    (2023, 10, 9), (2023, 11, 23),
    (2024, 1, 15), (2024, 2, 19), (2024, 5, 27), (2024, 9, 2),
    (2024, 10, 14), (2024, 11, 28),
    (2025, 1, 20), (2025, 2, 17), (2025, 5, 26), (2025, 9, 1),
    (2025, 10, 13), (2025, 11, 27),
    (2026, 1, 19), (2026, 2, 16), (2026, 5, 25), (2026, 9, 7),
    (2026, 10, 12), (2026, 11, 26),
}


def _is_holiday(dt: datetime) -> bool:
    if (dt.month, dt.day) in _FIXED_HOLIDAYS_MD:
        return True
    if (dt.year, dt.month, dt.day) in _MOVING_HOLIDAYS:
        return True
    return False


def encode_one(dt: datetime) -> np.ndarray:
    """Single (44,) calendar vector for the given timestamp."""
    v = np.zeros(CAL_DIM, dtype=np.float32)
    v[dt.hour] = 1.0                       # 0..23
    v[24 + dt.weekday()] = 1.0             # 24..30  (Mon=0)
    v[31 + dt.month - 1] = 1.0             # 31..42
    v[43] = 1.0 if _is_holiday(dt) else 0.0
    return v


def encode_range(start_dt: datetime, n_hours: int) -> np.ndarray:
    """Stack n_hours calendar vectors starting at start_dt (inclusive)."""
    return np.stack([encode_one(start_dt + timedelta(hours=i))
                     for i in range(n_hours)], axis=0)


if __name__ == "__main__":
    now = datetime(2022, 12, 25, 12)
    v = encode_one(now)
    print(f"Christmas noon 2022: hour={v[:24].argmax()}, "
          f"dow={v[24:31].argmax()}, month={v[31:43].argmax()+1}, "
          f"holiday={v[43]:.0f}")
