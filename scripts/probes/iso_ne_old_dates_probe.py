"""
Probe whether the ISO-NE public 5-min zonal endpoint serves old dates.

Before launching the 4-6 h multi-year drift sweep that needs demand
history for May 2022, 2023, 2024, 2025 + April-May 2026, verify that
the endpoint we use today (`space/iso_ne_zonal.py:fetch_range`) returns
≥95 % complete hourly data for each of the three older years.

If the probe passes, we proceed with Step 2.5b using the same fetcher.
If any window fails, surface the failure and the user can decide
between (a) pivoting to an EIA backup, (b) dropping that year, or (c)
shortening the affected window.

Read-only, single-shot. No background, no loops.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from space.iso_ne_zonal import ZONE_COLS, fetch_range


# Probe windows. Format: (label, start ISO, end ISO).
PROBE_WINDOWS = [
    ("2023-May", "2023-05-01", "2023-05-07"),
    ("2022-May", "2022-05-01", "2022-05-07"),
    ("2024-May", "2024-05-01", "2024-05-07"),
]

EXPECTED_HOURS = 7 * 24   # 168 hourly observations per window
COMPLETENESS_THRESHOLD = 0.95   # 95 % non-null hours, per zone


def probe_window(label: str, start_iso: str, end_iso: str) -> dict:
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    print(f"\n=== Probing {label}  ({start_iso} → {end_iso}) ===", flush=True)
    try:
        df = fetch_range(start, end, hourly=True)
    except Exception as e:  # noqa: BLE001
        return {"label": label, "ok": False, "error": str(e),
                 "n_hours_returned": 0, "completeness_per_zone": {}}

    n_hours_returned = len(df)
    # Per-zone non-null counts
    completeness = {
        z: float(df[z].notna().sum() / EXPECTED_HOURS) for z in ZONE_COLS
    }
    min_compl = min(completeness.values()) if completeness else 0.0
    overall_ok = (
        n_hours_returned >= EXPECTED_HOURS * COMPLETENESS_THRESHOLD
        and min_compl >= COMPLETENESS_THRESHOLD
    )
    print(f"  Hours returned : {n_hours_returned} / {EXPECTED_HOURS} expected",
           flush=True)
    print(f"  Min per-zone completeness : {min_compl * 100:.1f} %", flush=True)
    print(f"  Per zone:", flush=True)
    for z, c in completeness.items():
        flag = "OK" if c >= COMPLETENESS_THRESHOLD else "LOW"
        print(f"    {z:11s}: {c * 100:5.1f} %  [{flag}]", flush=True)
    return {
        "label": label,
        "start": start_iso,
        "end": end_iso,
        "ok": overall_ok,
        "n_hours_returned": n_hours_returned,
        "completeness_per_zone": completeness,
        "min_completeness": min_compl,
    }


def main() -> int:
    results = [probe_window(*w) for w in PROBE_WINDOWS]
    print()
    print("=" * 60)
    print("Probe summary:")
    print("=" * 60)
    all_ok = True
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {r['label']:10s}: {status:4s}  "
               f"({r['n_hours_returned']}/{EXPECTED_HOURS} hours, "
               f"min zone completeness {r.get('min_completeness', 0) * 100:.1f} %)")
        if not r["ok"]:
            all_ok = False
            err = r.get("error")
            if err:
                print(f"    error: {err}")
    print()
    if all_ok:
        print("✅ All 3 probe windows pass — Step 2.5b can use the existing "
               "fetcher unchanged.")
        return 0
    print("❌ At least one probe window failed — surface to user before "
           "running Step 2.5b.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
