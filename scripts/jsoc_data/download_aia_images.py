import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
from datetime import timedelta
import astropy.units as u
from sunpy.net import attrs as a
from sunpy.net.jsoc import JSOCClient
import logging
import argparse

from config.config import WAVELENGTHS, DELTA_MINUTES, AIA_DOWNLOAD_RESOLUTION

# -----------------------
# Argument Parser
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Start index of flare events")
parser.add_argument("--end", type=int, default=None, help="End index of flare events")
args = parser.parse_args()

# -----------------------
# Logging Setup
# -----------------------
log_path = "/home/kyang30/flare_pipeline/logs/jsoc_download.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console)
logger.info(f"🔥 Running event batch: {args.start} to {args.end}")

# -----------------------
# Setup
# -----------------------
client = JSOCClient()
event_csv = "/data/kyang30/flare_localization/events.csv"
output_base = "/data/kyang30/flare_localization/aia"
os.makedirs(output_base, exist_ok=True)

# -----------------------
# Read and Slice Events
# -----------------------
events = pd.read_csv(event_csv)
events = events.iloc[args.start:args.end]

records = []

# -----------------------
# Loop over Events
# -----------------------
for i, row in events.iterrows():
    peak_time = pd.to_datetime(row["peak_time"])
    t_start = peak_time - timedelta(minutes=DELTA_MINUTES)
    t_end = peak_time + timedelta(minutes=DELTA_MINUTES)

    event_id = row["event_id"]
    record = {
        "event_id": event_id,
        "peak_time": peak_time,
    }

    for wl in WAVELENGTHS:
        logger.info(f"[{event_id}] Querying {wl}Å from {t_start} to {t_end} via JSOC")

        try:
            result = client.search(
                a.Time(t_start, t_end),
                a.jsoc.Series("aia.lev1_euv_12s"),
                a.Wavelength(wl * u.angstrom),
                a.jsoc.Segment("image"),
                a.jsoc.Notify("kyang30@student.gsu.edu")
            )

            if len(result) == 0:
                logger.warning(f"  ⚠️  No data found for {wl}Å @ {peak_time}")
                record[f"{wl}A_count"] = 0
                continue

            save_dir = os.path.join(output_base, f"{event_id}", f"{wl}A")
            os.makedirs(save_dir, exist_ok=True)

            success = False
            for attempt in range(3):
                try:
                    client.fetch(result, path=os.path.join(save_dir, "{file}"))
                    logger.info(f"  ✅ Downloaded {len(result)} files for {wl}Å")
                    record[f"{wl}A_count"] = len(result)
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"  ⚠️  Retry {attempt+1} for {wl}Å @ {peak_time}: {e}")
                    time.sleep(5)

            if not success:
                logger.error(f"  ❌ Final failure for {wl}Å @ {peak_time}")
                record[f"{wl}A_count"] = -1

        except Exception as e:
            logger.error(f"  ❌ Error during JSOC search for {wl}Å @ {peak_time}: {e}")
            record[f"{wl}A_count"] = -1

    records.append(record)

# -----------------------
# Save Summary
# -----------------------
summary_csv = f"/data/kyang30/flare_localization/aia_download_summary_{args.start}_{args.end}.csv"
pd.DataFrame(records).to_csv(summary_csv, index=False)
logger.info(f"📄 Saved download summary to {summary_csv}")
