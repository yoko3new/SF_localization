"""
query_flare_events.py

This script queries C-class and above solar flare events from the HEK database
within specified monthly time ranges. It filters out flares near the solar limb
(using a configurable maximum angular distance), and saves the results to a CSV file
for use in flare localization tasks.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sunpy.net.hek import HEKClient
from sunpy.net import attrs as a
from sunpy.net.hek import attrs as hek_attrs

from config.config import (
    MONTHLY_WINDOWS,
    GOES_CLASS_THRESHOLD,
    MAX_SOLAR_DISTANCE,
    EVENT_CSV,
    goes_class_to_number
)

def flare_distance(flare):
    """Compute angular distance from solar disk center (HPC coordinates)."""
    x = flare.get('hpc_x', 0)
    y = flare.get('hpc_y', 0)
    if x is None or y is None:
        return np.inf  # Discard if position is missing
    return np.sqrt(x**2 + y**2)

def format_time(t):
    """Convert astropy Time or datetime to string (fallback-safe)."""
    if t is None:
        return ""
    try:
        return str(t.to_datetime())
    except AttributeError:
        return str(t)

def query_flare_events():
    """Query HEK for flare events, filter, and save to CSV."""
    client = HEKClient()
    all_events = []

    threshold_value = goes_class_to_number(GOES_CLASS_THRESHOLD)

    for start, end in MONTHLY_WINDOWS:
        print(f"Querying: {start} to {end}")
        result = client.search(
            a.Time(start, end),
            hek_attrs.EventType("FL")
        )
        print(f"  Found {len(result)} events from HEK")

        for r in result:
            flare_class = r.get("fl_goescls", "")
            try:
                if goes_class_to_number(flare_class) < threshold_value:
                    continue  # skip low-class flare
            except Exception:
                continue  # skip invalid flare class strings

            dist = flare_distance(r)
            if dist <= MAX_SOLAR_DISTANCE:
                all_events.append({
                    "event_id": f"event_{len(all_events):04d}",
                    "hek_id": r.get("hek_id"),
                    "start_time": format_time(r.get("event_starttime")),
                    "peak_time": format_time(r.get("event_peaktime")),
                    "end_time": format_time(r.get("event_endtime")),
                    "hpc_x": r.get("hpc_x"),
                    "hpc_y": r.get("hpc_y"),
                    "goes_class": flare_class,
                    "distance": dist,
                    "ar_noaa": r.get("ar_noaanum"),
                })

    df = pd.DataFrame(all_events)
    print(f"Total filtered events: {len(df)}")
    df.to_csv(EVENT_CSV, index=False)
    print(f"Saved to: {EVENT_CSV}")

if __name__ == "__main__":
    query_flare_events()
