# config/config.py

# dictionary for data
DATA_ROOT = "/data/kyang30/flare_localization"

RAW_IMAGE_DIR = f"{DATA_ROOT}/aia_raw"
RESAMPLED_DIR = f"{DATA_ROOT}/aia_resampled"
LABEL_DIR = f"{DATA_ROOT}/heatmap_labels"
EVENT_CSV = f"{DATA_ROOT}/events.csv"
MODEL_DIR = f"{DATA_ROOT}/models"
LOG_DIR = f"{DATA_ROOT}/logs"

# wavelength for AIA
WAVELENGTHS = [94, 131, 171, 193]  # unit：Å
DELTA_MINUTES = 10  # downlaod window: 10 minute around flare peak time

# only query flares from January of each year (2010–2014)
MONTHLY_WINDOWS = [
    ("2010-01-01", "2010-01-31"),
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
