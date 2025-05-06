import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sunpy.map import Map
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
import astropy.units as u

# Add parent directory to sys.path to enable config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import ALIGNED_DIR, HEK_COORD_CSV, HEATMAP_OUTPUT_DIR

def generate_heatmap(shape, center, sigma=5):
    """
    Generate a 2D Gaussian heatmap centered at the given pixel coordinates.

    Args:
        shape (tuple): Shape of the heatmap (height, width).
        center (tuple): (x, y) center location in pixel coordinates.
        sigma (float): Standard deviation of the Gaussian peak.

    Returns:
        np.ndarray: 2D array representing the heatmap.
    """
    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    xx, yy = np.meshgrid(x, y)
    cx, cy = center
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap

def generate_all_heatmaps():
    """
    Generate pixel-level heatmaps for all events using HEK-provided helioprojective coordinates.
    
    Steps:
    - Load HEK coordinates from CSV.
    - Load the reference aligned AIA FITS file for each event (frame 10, 94 Å).
    - Convert HPC coordinates to pixel coordinates using the WCS header.
    - Generate a 2D Gaussian heatmap centered at the converted pixel location.
    - Save the heatmap as .npy and record summary as a CSV file.
    """
    hek_df = pd.read_csv(HEK_COORD_CSV).set_index("event_id")
    os.makedirs(HEATMAP_OUTPUT_DIR, exist_ok=True)

    records = []
    event_dirs = sorted(glob(os.path.join(ALIGNED_DIR, "*")))

    for event_dir in tqdm(event_dirs, desc="Generating heatmaps"):
        event_id = os.path.basename(event_dir)
        if event_id not in hek_df.index:
            continue

        try:
            # Use frame 10 from the 94 Å channel as reference
            ref_path = os.path.join(event_dir, "94A", "10.fits")
            m = Map(ref_path)

            # Convert helioprojective coordinates to pixel coordinates
            x_arcsec = hek_df.loc[event_id, "hpc_x"]
            y_arcsec = hek_df.loc[event_id, "hpc_y"]
            sc = SkyCoord(x_arcsec * u.arcsec, y_arcsec * u.arcsec, frame=m.coordinate_frame)
            px, py = m.wcs.world_to_pixel(sc)

            # Generate heatmap with Gaussian centered at (px, py)
            heatmap = generate_heatmap((512, 512), (px, py), sigma=5)

            # Save heatmap
            output_dir = os.path.join(HEATMAP_OUTPUT_DIR, event_id)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "heatmap.npy")
            np.save(output_path, heatmap)

            records.append((event_id, px, py, output_path))

        except Exception as e:
            print(f"[!] Failed to process {event_id}: {e}")

    # Save summary of heatmap generation
    df = pd.DataFrame(records, columns=["event_id", "x_pixel", "y_pixel", "heatmap_path"])
    df.to_csv(os.path.join(HEATMAP_OUTPUT_DIR, "heatmap_records.csv"), index=False)
    print(f"\nGenerated {len(records)} heatmaps.")
    print(f"Saved to: {HEATMAP_OUTPUT_DIR}")

if __name__ == "__main__":
    generate_all_heatmaps()
