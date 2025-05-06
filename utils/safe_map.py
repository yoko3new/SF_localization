# utils/io_utils.py
from sunpy.map import Map
from astropy.io import fits

def safe_map(fits_path):
    """
    Load a SunPy Map from a FITS file, safely extracting data from COMPRESSED_IMAGE HDU.
    """
    with fits.open(fits_path) as hdul:
        return Map(hdul[1])  # Use COMPRESSED_IMAGE HDU (index 1)