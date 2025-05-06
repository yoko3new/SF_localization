import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sunpy.map import Map
from astropy.coordinates import SkyCoord
from astropy.io import fits
from skimage.transform import resize
import astropy.units as u
from sunpy.coordinates import get_earth, Helioprojective
import time

# Load config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

event_id = "event_0011"
channel = "94A"
CROP_SIZE = 512
MAX_REPROJECT_TIME = 10

RESAMPLED_DIR = config.RESAMPLED_DIR
ALIGNED_DIR = config.ALIGNED_DIR
HEK_COORD_CSV = config.HEK_COORD_CSV

hek_df = pd.read_csv(HEK_COORD_CSV).set_index("event_id")

def get_flare_pixel_coords(m, x_arcsec, y_arcsec):
    obstime = m.date
    center = get_earth(obstime)

    with Helioprojective.assume_spherical_screen(center=center):
        sc = SkyCoord(x_arcsec * u.arcsec, y_arcsec * u.arcsec, frame=m.coordinate_frame)
        px, py = m.wcs.world_to_pixel(sc)

    return px, py

def crop_and_resize(m, px, py, size=512):
    data = m.data
    half = size // 2
    x, y = int(px), int(py)
    cropped = np.zeros((size, size))

    x_start = max(x - half, 0)
    y_start = max(y - half, 0)
    x_end = min(x + half, data.shape[1])
    y_end = min(y + half, data.shape[0])

    crop = data[y_start:y_end, x_start:x_end]
    cropped[:crop.shape[0], :crop.shape[1]] = crop
    resized = resize(cropped, (size, size), anti_aliasing=True)
    return Map(resized, m.meta)

def save_aligned_map(map_obj, output_path):
    from astropy.io.fits import Header
    primary_hdu = fits.PrimaryHDU()
    header = Header(map_obj.meta)
    comp_hdu = fits.CompImageHDU(data=map_obj.data, header=header, name="COMPRESSED_IMAGE")
    fits.HDUList([primary_hdu, comp_hdu]).writeto(output_path, overwrite=True)

def safe_reproject(m, reference_map):
    try:
        start = time.time()
        aligned_map = m.reproject_to(reference_map.wcs)
        if time.time() - start > MAX_REPROJECT_TIME:
            raise TimeoutError("Reproject exceeded time limit")
        return aligned_map
    except Exception as e:
        print(f"[!] Reproject failed: {e}")
        return None

def debug_single_event(event_id):
    x_arcsec = hek_df.loc[event_id, "hpc_x"]
    y_arcsec = hek_df.loc[event_id, "hpc_y"]

    input_dir = os.path.join(RESAMPLED_DIR, event_id, channel)
    output_dir = os.path.join(ALIGNED_DIR, event_id, channel)
    os.makedirs(output_dir, exist_ok=True)

    fits_files = sorted(glob(os.path.join(input_dir, "*.fits")))
    if len(fits_files) == 0:
        print("❌ No FITS files found")
        return

    cropped_maps = []
    for f in fits_files:
        try:
            m = Map(f)
            px, py = get_flare_pixel_coords(m, x_arcsec, y_arcsec)
            cropped = crop_and_resize(m, px, py, size=CROP_SIZE)
            cropped_maps.append(cropped)
        except Exception as e:
            print(f"[!] Failed to load/crop {f}: {e}")

    if not cropped_maps:
        print("❌ No valid maps loaded")
        return

    reference_map = cropped_maps[len(cropped_maps) // 2]

    for i, m in enumerate(cropped_maps):
        aligned_map = safe_reproject(m, reference_map)
        if aligned_map is not None:
            output_path = os.path.join(output_dir, f"{i:02d}.fits")
            save_aligned_map(aligned_map, output_path)
            print(f"✅ Saved: {output_path} | NaN ratio: {np.isnan(aligned_map.data).sum() / aligned_map.data.size:.3f}")
        else:
            print(f"⚠️ Skipped frame {i} due to reprojection failure")

if __name__ == "__main__":
    debug_single_event("event_0011")
