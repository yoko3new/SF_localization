import os
import sys
import time
from glob import glob
from sunpy.map import Map
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
from astropy.coordinates import SkyCoord
import astropy.units as u
from skimage.transform import resize

# Add parent directory to sys.path to find config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

RESAMPLED_DIR = config.RESAMPLED_DIR
ALIGNED_DIR = config.ALIGNED_DIR
REPORT_PATH = "align_report.csv"
DETAILED_LOG_PATH = "align_detailed_log.csv"
HEK_COORD_CSV = config.HEK_COORD_CSV  # assume this is a CSV with event_id, hpc_x, hpc_y
MAX_REPROJECT_TIME = 10  # seconds
CROP_SIZE = 512

hek_df = pd.read_csv(HEK_COORD_CSV).set_index("event_id")

from astropy.io import fits

def save_aligned_map(map_obj, output_path):
    """Save a SunPy Map to FITS with COMPRESSED_IMAGE as HDU[1]."""
    primary_hdu = fits.PrimaryHDU()  # Empty primary HDU
    comp_hdu = fits.CompImageHDU(data=map_obj.data, header=map_obj.meta, name="COMPRESSED_IMAGE")
    hdul = fits.HDUList([primary_hdu, comp_hdu])
    hdul.writeto(output_path, overwrite=True)

def safe_reproject(m, reference_map):
    try:
        start = time.time()
        aligned_map = m.reproject_to(reference_map.wcs)
        if time.time() - start > MAX_REPROJECT_TIME:
            raise TimeoutError("reproject exceeded time limit")
        return aligned_map, "aligned", ""
    except Exception as e:
        return None, "align_failed", str(e)


def crop_and_resize(m, x, y, size=512):
    data = m.data
    half = size // 2
    x, y = int(x), int(y)
    cropped = np.zeros((size, size))

    x_start = max(x - half, 0)
    y_start = max(y - half, 0)
    x_end = min(x + half, data.shape[1])
    y_end = min(y + half, data.shape[0])

    crop = data[y_start:y_end, x_start:x_end]
    cropped[:crop.shape[0], :crop.shape[1]] = crop
    resized = resize(cropped, (512, 512), anti_aliasing=True)
    return Map(resized, m.meta)


def align_event_channel(fits_files, output_dir, event_id, channel):
    os.makedirs(output_dir, exist_ok=True)
    maps = []
    valid_files = []
    logs = []

    for f in sorted(fits_files):
        try:
            m = Map(f)
            if event_id in hek_df.index:
                x_arcsec = hek_df.loc[event_id, "hpc_x"]
                y_arcsec = hek_df.loc[event_id, "hpc_y"]
                sc = SkyCoord(x_arcsec * u.arcsec, y_arcsec * u.arcsec, frame=m.coordinate_frame)
                px, py = m.wcs.world_to_pixel(sc)
                m = crop_and_resize(m, px, py, size=CROP_SIZE)
            maps.append(m)
            valid_files.append(f)
        except Exception as e:
            logs.append((event_id, channel, f, "load_failed", str(e)))

    if len(maps) == 0:
        return logs

    center_index = len(maps) // 2
    reference_map = maps[center_index]

    for i, (m, fpath) in enumerate(zip(maps, valid_files)):
        aligned_map, status, msg = safe_reproject(m, reference_map)
        if aligned_map is not None:
            save_aligned_map(aligned_map, os.path.join(output_dir, f"{i:02d}.fits"))
        logs.append((event_id, channel, fpath, status, msg))

    return logs


def process_event(event_dir):
    rel_path = os.path.relpath(event_dir, RESAMPLED_DIR)
    output_event_dir = os.path.join(ALIGNED_DIR, rel_path)

    fits_94 = sorted(glob(os.path.join(event_dir, "94A", "*.fits")))
    fits_131 = sorted(glob(os.path.join(event_dir, "131A", "*.fits")))

    report_entry = {
        "event_id": rel_path,
        "fits_94_count": len(fits_94),
        "fits_131_count": len(fits_131),
        "status": "ok"
    }

    if not (20 <= len(fits_94) <= 22 and 20 <= len(fits_131) <= 22):
        report_entry["status"] = "skipped"
        return report_entry, []

    if len(fits_94) > 21:
        mid = len(fits_94) // 2
        fits_94 = fits_94[mid - 10: mid + 11]
    if len(fits_131) > 21:
        mid = len(fits_131) // 2
        fits_131 = fits_131[mid - 10: mid + 11]

    logs = []
    try:
        logs += align_event_channel(fits_94, os.path.join(output_event_dir, "94A"), rel_path, "94A")
        logs += align_event_channel(fits_131, os.path.join(output_event_dir, "131A"), rel_path, "131A")
    except Exception as e:
        report_entry["status"] = f"error: {e}"

    return report_entry, logs


def align_all_events():
    event_dirs = glob(os.path.join(RESAMPLED_DIR, "*"))
    print(f"Total events: {len(event_dirs)}")

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = list(tqdm(pool.imap(process_event, event_dirs), total=len(event_dirs), desc="Aligning events"))

    report_data, detailed_logs = zip(*results)
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(REPORT_PATH, index=False)
    print(f"\n Alignment report saved to {REPORT_PATH}")

    flat_logs = [log for logs in detailed_logs for log in logs]
    detailed_df = pd.DataFrame(flat_logs, columns=["event_id", "channel", "filepath", "status", "message"])
    detailed_df.to_csv(DETAILED_LOG_PATH, index=False)
    print(f" Detailed log saved to {DETAILED_LOG_PATH}")


if __name__ == '__main__':
    align_all_events()
