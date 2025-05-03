import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
from datetime import timedelta
from sunpy.net import Fido, attrs as a
import astropy.units as u

from config.config import WAVELENGTHS, DELTA_MINUTES, AIA_DOWNLOAD_RESOLUTION

event_csv = "/data/kyang30/flare_localization/events.csv"
output_base = "/data/kyang30/flare_localization/aia_multiband_data"
os.makedirs(output_base, exist_ok=True)

events = pd.read_csv(event_csv)
records = []

for i, row in events.iterrows():
    peak_time = pd.to_datetime(row["peak_time"])
    t_start = peak_time - timedelta(minutes=DELTA_MINUTES)
    t_end = peak_time + timedelta(minutes=DELTA_MINUTES)

    record = {
        "event_id": i,
        "peak_time": peak_time,
    }

    for wl in WAVELENGTHS:
        print(f"[{i}] Querying {wl}Å from {t_start} to {t_end}")

        try:
            result = Fido.search(
                a.Time(t_start, t_end),
                a.Instrument("AIA"),
                a.Wavelength(wl * u.angstrom),
                a.Sample(AIA_DOWNLOAD_RESOLUTION * u.second)
            )

            # ➤ 如果没有数据返回
            if len(result) == 0:
                print(f"  ⚠️  No data found for {wl}Å @ {peak_time}")
                record[f"{wl}A_count"] = 0
                continue

            save_dir = os.path.join(output_base, f"event_{i:04d}", f"{wl}A")
            os.makedirs(save_dir, exist_ok=True)

            # ➤ 尝试下载最多 3 次
            success = False
            for attempt in range(3):
                try:
                    Fido.fetch(result, path=os.path.join(save_dir, "{file}"), overwrite=True)
                    print(f"  ✅ Downloaded {len(result[0])} files for {wl}Å")
                    record[f"{wl}A_count"] = len(result[0])
                    success = True
                    break
                except Exception as e:
                    print(f"  ⚠️  Retry {attempt+1} for {wl}Å @ {peak_time}: {e}")
                    time.sleep(5)

            if not success:
                print(f"  ❌ Final failure for {wl}Å @ {peak_time}")
                record[f"{wl}A_count"] = -1

        except Exception as e:
            print(f"  ❌ Error during search for {wl}Å @ {peak_time}: {e}")
            record[f"{wl}A_count"] = -1

    records.append(record)

# ➤ 保存统计结果
summary_csv = "/data/kyang30/flare_localization/aia_download_summary.csv"
pd.DataFrame(records).to_csv(summary_csv, index=False)
print(f"📄 Saved download summary to {summary_csv}")