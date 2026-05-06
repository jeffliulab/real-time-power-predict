"""
Three naive baselines for the workshop paper's drift-quantification table.

All three take a per-zone demand history (a numpy array shaped
``(T, 8)`` of MWh values, ending at the forecast-issue time T-1 hour) and
produce a ``(24, 8)`` 24-hour forecast for hours [T, T+23], one column per
ISO-NE zone in the canonical order ME, NH, VT, CT, RI, SEMA, WCMA,
NEMA_BOST.

  persistence_1d(history)  : y_hat[h, z] = history[-24 + h, z]
  persistence_7d(history)  : y_hat[h, z] = history[-168 + h, z]
  climatological(history, start_ts, ...) : y_hat[h, z] = mean of
      history rows whose (hour-of-day, day-of-week) match the
      forecast hour h, computed over the past N weeks.

These are the standard "floor" comparators in the load-forecasting
literature; if the trained model can't beat all three, it's not earning
its parameters. We use them in §4 of the workshop paper to put the
deployed model's MAPE in context.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
HORIZON = 24


def persistence_1d(history_mwh: np.ndarray) -> np.ndarray:
    """y_hat[h, z] = history[-24 + h, z], for h in [0, 24).

    Equivalent to "yesterday at the same hour" — the simplest possible
    baseline that respects diurnal periodicity.

    Args:
        history_mwh: ``(T, 8)`` per-zone demand, T >= 24.
    Returns:
        ``(24, 8)`` float32 forecast.
    """
    if history_mwh.shape[0] < 24:
        raise ValueError(
            f"persistence_1d needs >= 24 hours of history, got {history_mwh.shape[0]}")
    return history_mwh[-24:].astype(np.float32)


def persistence_7d(history_mwh: np.ndarray) -> np.ndarray:
    """y_hat[h, z] = history[-168 + h, z], for h in [0, 24).

    "Same hour last week" — respects weekly periodicity (weekday vs
    weekend) which 1-day persistence does not.

    Args:
        history_mwh: ``(T, 8)`` per-zone demand, T >= 168.
    Returns:
        ``(24, 8)`` float32 forecast.
    """
    if history_mwh.shape[0] < 168:
        raise ValueError(
            f"persistence_7d needs >= 168 hours of history, got "
            f"{history_mwh.shape[0]}")
    return history_mwh[-168:-144].astype(np.float32)


def climatological(history_mwh: np.ndarray,
                    start_ts: datetime,
                    history_start_ts: Optional[datetime] = None,
                    weeks_back: int = 4,
                    ) -> np.ndarray:
    """y_hat[h, z] = mean of history rows whose (hour-of-day, day-of-week)
    matches the forecast hour h, drawn from the last ``weeks_back`` weeks.

    For each forecast hour h \\in [0, 24) and each zone z, average over all
    history hours whose (hour, day-of-week) tuple equals the forecast
    hour's (hour, day-of-week). The "weeks_back" knob controls how much
    history to draw on; defaults to 4 weeks (28 days * 24 = 672 hours,
    same as the Chronos context length).

    Args:
        history_mwh: ``(T, 8)`` per-zone demand. T should be >= weeks_back*168.
        start_ts: forecast issue time (UTC, naive). Hour 0 of the forecast
            corresponds to start_ts.
        history_start_ts: UTC timestamp of history_mwh[0]. If None, derived
            assuming history_mwh ends at start_ts - 1 hour.
        weeks_back: how many weeks of history to draw from. Default 4.
    Returns:
        ``(24, 8)`` float32 forecast.
    """
    n_hours = history_mwh.shape[0]
    if history_start_ts is None:
        history_start_ts = start_ts - timedelta(hours=n_hours)

    # Build (hour-of-day, day-of-week) labels for every row of history.
    hist_dt = [history_start_ts + timedelta(hours=i) for i in range(n_hours)]
    hist_keys = np.array(
        [(t.hour, t.weekday()) for t in hist_dt], dtype=np.int32)

    # Restrict to the most-recent weeks_back weeks
    cutoff = max(0, n_hours - weeks_back * 168)
    hist_keys = hist_keys[cutoff:]
    hist_values = history_mwh[cutoff:]

    out = np.zeros((HORIZON, history_mwh.shape[1]), dtype=np.float32)
    for h in range(HORIZON):
        f_dt = start_ts + timedelta(hours=h)
        f_key = (f_dt.hour, f_dt.weekday())
        mask = (hist_keys[:, 0] == f_key[0]) & (hist_keys[:, 1] == f_key[1])
        if mask.sum() == 0:
            # Fall back to "same hour" mean if no exact (hour, dow) match
            mask = hist_keys[:, 0] == f_key[0]
        if mask.sum() == 0:
            # Final fallback: zone-wise mean of all history
            out[h] = hist_values.mean(axis=0)
        else:
            out[h] = hist_values[mask].mean(axis=0)
    return out


def all_baselines(history_mwh: np.ndarray,
                   start_ts: datetime,
                   history_start_ts: Optional[datetime] = None,
                   ) -> dict[str, np.ndarray]:
    """Convenience: run all 3 naive baselines and return a name -> (24,8) dict."""
    return {
        "persistence_1d": persistence_1d(history_mwh),
        "persistence_7d": persistence_7d(history_mwh),
        "climatological": climatological(history_mwh, start_ts,
                                          history_start_ts=history_start_ts),
    }


if __name__ == "__main__":
    # Smoke test: synthesize a 30-day history with strong diurnal pattern + noise
    np.random.seed(42)
    n_history = 30 * 24
    t0 = datetime(2026, 4, 1, 0, 0, 0)
    hours_of_day = np.arange(n_history) % 24
    diurnal = 800 + 600 * np.sin((hours_of_day - 8) * np.pi / 12)
    weekly = np.where((np.arange(n_history) // 24) % 7 < 5, 100, -200)
    noise = np.random.randn(n_history) * 50
    history = (diurnal + weekly + noise)[:, None] * np.array(
        [1.0, 1.1, 0.5, 2.5, 0.7, 1.2, 1.4, 2.1])  # 8 zones, scaled
    history = history.astype(np.float32)

    start_ts = t0 + timedelta(hours=n_history)
    out = all_baselines(history, start_ts,
                         history_start_ts=t0)
    for name, pred in out.items():
        print(f"{name:18s}: shape={pred.shape}, "
              f"mean={pred.mean():.0f}, "
              f"hour0={pred[0, 0]:.0f}, hour12={pred[12, 0]:.0f}")
