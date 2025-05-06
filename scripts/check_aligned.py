import os
import numpy as np
from glob import glob
from tqdm import tqdm
from sunpy.map import Map
from astropy.io import fits

# Set your aligned FITS root directory
ALIGNED_DIR = "/data/kyang30/flare_localization/aia_aligned"

def safe_map(fits_path):
    """Safely load the COMPRESSED_IMAGE HDU from a FITS file."""
    hdul = fits.open(fits_path)
    return Map(hdul[1])

def is_valid_fits(fits_path):
    """Check whether a FITS file contains usable image data."""
    try:
        m = safe_map(fits_path)
        data = m.data
        return data is not None and not np.isnan(data).all()
    except Exception:
        return False

def check_all_fits(root_dir):
    """Check all FITS files under a directory recursively."""
    fits_files = sorted(glob(os.path.join(root_dir, "**", "*.fits"), recursive=True))
    invalid_files = []

    for f in tqdm(fits_files, desc="Checking aligned FITS"):
        if not is_valid_fits(f):
            invalid_files.append(f)

    print(f"\nChecked {len(fits_files)} files. Invalid: {len(invalid_files)}")
    if invalid_files:
        print("Examples of invalid files:")
        for f in invalid_files[:10]:
            print(" -", f)

if __name__ == "__main__":
    check_all_fits(ALIGNED_DIR)