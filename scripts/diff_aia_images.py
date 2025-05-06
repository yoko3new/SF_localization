import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sunpy.map import Map
from skimage.exposure import rescale_intensity

# ---------------- Configuration ---------------- #
# Add parent directory to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import ALIGNED_DIR, DIFF_OUTPUT_DIR

# Create output directory if it doesn't exist
os.makedirs(DIFF_OUTPUT_DIR, exist_ok=True)

# ---------------- Core Functions ---------------- #

def compute_normalized_diff_sequence(fits_files):
    """
    Compute a sequence of normalized difference images from consecutive AIA FITS frames.

    Parameters:
        fits_files (list): Sorted list of file paths to AIA FITS images.

    Returns:
        list of np.ndarray: Each element is a normalized 2D difference image.
    """
    maps = [Map(f) for f in sorted(fits_files)]
    data_seq = [m.data.astype(np.float32) for m in maps]

    diff_seq = []
    for i in range(1, len(data_seq)):
        diff = data_seq[i] - data_seq[i - 1]
        norm_diff = (diff - np.mean(diff)) / (np.std(diff) + 1e-6)
        diff_seq.append(norm_diff)

    return diff_seq


def process_event_diff(event_dir):
    """
    Process a single flare event directory to generate and save
    normalized difference images for both 94 Å and 131 Å channels.

    Parameters:
        event_dir (str): Path to the aligned FITS directory for a single event.
    """
    event_id = os.path.basename(event_dir)
    for channel in ["94A", "131A"]:
        aligned_channel_dir = os.path.join(event_dir, channel)
        fits_files = sorted(glob(os.path.join(aligned_channel_dir, "*.fits")))
        if len(fits_files) < 2:
            continue  # Not enough frames to compute difference

        diff_seq = compute_normalized_diff_sequence(fits_files)

        # Output directory for difference frames
        save_dir = os.path.join(DIFF_OUTPUT_DIR, event_id, channel)
        os.makedirs(save_dir, exist_ok=True)

        # Save difference frames as .npy files
        for i, frame in enumerate(diff_seq):
            np.save(os.path.join(save_dir, f"{i:02d}.npy"), frame)

        # Save visualization of first 3 frames for sanity check
        fig, axes = plt.subplots(1, min(3, len(diff_seq)), figsize=(12, 4))
        for j in range(min(3, len(diff_seq))):
            axes[j].imshow(rescale_intensity(diff_seq[j], in_range=(-3, 3)), cmap="gray")
            axes[j].set_title(f"Frame {j}")
            axes[j].axis("off")
        plt.suptitle(f"{event_id} - {channel} Diff Preview")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{event_id}_{channel}_sanity_check.png"))
        plt.close()


# ---------------- Main Batch Processing ---------------- #

if __name__ == "__main__":
    event_dirs = sorted(glob(os.path.join(ALIGNED_DIR, "event_*")))
    for event_dir in tqdm(event_dirs, desc="Generating diff sequences"):
        process_event_diff(event_dir)
