"""
resample_aia_by_filename.py

This script resamples raw AIA FITS image sequences to a target cadence (default: 1 minute)
based on the timestamps encoded in the FITS filenames. For each flare event directory,
it performs the following steps:

1. Parse timestamps from filenames (expects ISO-style `T...Z` segments).
2. Sort images by time.
3. Construct a sequence of target timestamps at 1-minute intervals.
4. For each target timestamp, select the closest available frame (without duplication).
5. Copy the selected FITS files to a resampled output directory, preserving structure.

This preprocessing step is essential to reduce temporal redundancy and unify input resolution
for downstream machine learning tasks such as solar flare localization.
"""

import sys
import os
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from sunpy.map import Map

# Load global config paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import RAW_IMAGE_DIR, RESAMPLED_DIR

# --------------------------
# Path Setup
# --------------------------
input_root = RAW_IMAGE_DIR        # Directory containing original FITS files (grouped per event/channel)
output_root = RESAMPLED_DIR       # Directory to save resampled results

event_dirs = glob(os.path.join(input_root, "*", "*"))  # e.g., /raw/event_0001/94A
processed = []

print("Resampling AIA data based on filenames:")

# --------------------------
# Main Resampling Loop
# --------------------------
for event_dir in tqdm(event_dirs):
    fits_files = sorted(glob(os.path.join(event_dir, "*.fits")))
    if not fits_files:
        continue

    timestamps = []
    for f in fits_files:
        basename = os.path.basename(f)
        try:
            # Extract timestamp from filename (e.g., 2013-01-15T074303Z)
            parts = basename.split(".")
            time_part = next((p for p in parts if "T" in p and p.endswith("Z")), None)
            if not time_part:
                raise ValueError("No timestamp-like component found in filename.")
            dt = datetime.strptime(time_part, "%Y-%m-%dT%H%M%SZ")
            timestamps.append((f, dt))
        except Exception as e:
            print(f"[!] Skipped file (parse error): {basename} - {e}")
            continue

    if not timestamps:
        continue

    # Sort files by timestamp
    timestamps.sort(key=lambda x: x[1])
    files_sorted, times_sorted = zip(*timestamps)

    # Create list of target timestamps (1-minute interval)
    start_time = times_sorted[0].replace(second=0, microsecond=0)
    end_time = (times_sorted[-1] + timedelta(minutes=1)).replace(second=0, microsecond=0)
    minutes_total = int((end_time - start_time).total_seconds() // 60)
    target_times = [start_time + timedelta(minutes=i) for i in range(minutes_total + 1)]

    # Select closest file to each target time (no duplicates)
    selected_files = set()
    selected = []
    for target in target_times:
        closest = min(timestamps, key=lambda x: abs(x[1] - target))
        if closest[0] not in selected_files:
            selected.append(closest[0])
            selected_files.add(closest[0])

    # Save selected files to output directory (preserving relative path)
    rel_path = os.path.relpath(event_dir, input_root)
    output_dir = os.path.join(output_root, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    for f in selected:
        filename = os.path.basename(f)
        os.system(f"cp '{f}' '{os.path.join(output_dir, filename)}'")

    processed.append(event_dir)

# --------------------------
# Final Summary
# --------------------------
print(f"\nFinished. Total processed directories: {len(processed)}")
