# config/config.py

# dictionary for data
DATA_ROOT = "/data/kyang30/flare_localization"

RAW_IMAGE_DIR = f"{DATA_ROOT}/aia"
RESAMPLED_DIR = f"{DATA_ROOT}/aia_resampled"
ALIGNED_DIR = f"{DATA_ROOT}/aia_aligned"
HEK_COORD_CSV = "/data/kyang30/flare_localization/events.csv"
DIFF_OUTPUT_DIR = f"{DATA_ROOT}/diff_images"
HEATMAP_OUTPUT_DIR = f"{DATA_ROOT}/hek_heatmap"

# wavelength for AIA
WAVELENGTHS = [94, 131]  # unit: Å
DELTA_MINUTES = 10  # download window: 10 minutes around flare peak time
AIA_DOWNLOAD_RESOLUTION = 60  # 1min resolution

# only query flares from January of each year (2011–2014)
MONTHLY_WINDOWS = [
    ("2011-01-01", "2011-01-31"),
    ("2012-01-01", "2012-01-31"),
    ("2013-01-01", "2013-01-31"),
    ("2014-01-01", "2014-01-31"),
]

# filter
GOES_CLASS_THRESHOLD = "C1.0"  # GOES class threshold for filtering weak flares
# Maximum distance from solar disk center (in arcsec) to exclude limb flares
MAX_SOLAR_DISTANCE = 800

# data split
TRAIN_VAL_SPLIT = 0.8          # Portion of total events used for training + validation
VAL_PORTION = 0.2              # Within train+val, the portion allocated for validation

def goes_class_to_number(cls_str):
    cls_str = cls_str.strip().upper()
    scale = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    if cls_str[0] in scale:
        return float(cls_str[1:]) * scale[cls_str[0]]
    else:
        raise ValueError(f"Unknown GOES class: {cls_str}")
