"""
query_flare_events.py

This script queries solar flare events of GOES class C1.0 or above from the
Heliophysics Event Knowledgebase (HEK). It performs the following steps:

1. Loops over specified monthly time windows.
2. Queries the HEK database for flare events (EventType="FL").
3. Filters events based on:
    - GOES classification (e.g., ≥ C1.0)
    - Helioprojective Cartesian (HPC) distance from solar disk center (to exclude limb events)
4. Extracts relevant metadata (time, location, class, NOAA AR number).
5. Saves the filtered list to a CSV file for downstream use.

This script is intended to support solar flare localization studies.
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
    """
    Compute the angular distance (in arcseconds) of the flare center from the
    center of the solar disk using helioprojective Cartesian (HPC) coordinates.

    Returns:
        float: Euclidean distance in arcseconds. If coordinates are missing, returns np.inf.
    """
    x = flare.get('hpc_x', 0)
    y = flare.get('hpc_y', 0)
    if x is None or y is None:
        return np.inf
    return np.sqrt(x**2 + y**2)

def format_time(t):
    """
    Safely format time fields to ISO string.
    
    Args:
        t (datetime or astropy.time.Time): Time object

    Returns:
        str: ISO time string or empty string if unavailable
    """
    if t is None:
        return ""
    try:
        return str(t.to_datetime())
    except AttributeError:
        return str(t)

def query_flare_events():
    """
    Query the HEK database for solar flares within the given monthly time ranges,
    filter them by GOES class and spatial position, and save the result to a CSV file.
    """
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
                    continue
            except Exception:
                continue  # Skip unparseable or missing flare classes

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
