import os
import numpy as np
from tqdm import tqdm

# -------------------------
# Path Configuration
# -------------------------
DIFF_DIR = "/data/kyang30/flare_localization/diff_images"
EVENTS = os.listdir(DIFF_DIR)

# -------------------------
# Loop Through Each Event Directory
# -------------------------
for event_id in tqdm(EVENTS, desc="Merging difference sequences"):
    event_dir = os.path.join(DIFF_DIR, event_id)
    if not os.path.isdir(event_dir):
        continue  # Skip non-directory files

    try:
        # Load first 20 frames for both 94 Å and 131 Å channels
        diff_94 = sorted([f for f in os.listdir(os.path.join(event_dir, "94A")) if f.endswith(".npy")])[:20]
        diff_131 = sorted([f for f in os.listdir(os.path.join(event_dir, "131A")) if f.endswith(".npy")])[:20]

        # Skip if insufficient frames in either channel
        if len(diff_94) < 20 or len(diff_131) < 20:
            print(f"Skipped {event_id}: not enough frames (94A: {len(diff_94)}, 131A: {len(diff_131)})")
            continue

        # Load and stack frames into a [40, H, W] array (20 per channel)
        seq_94 = [np.load(os.path.join(event_dir, "94A", f)) for f in diff_94]
        seq_131 = [np.load(os.path.join(event_dir, "131A", f)) for f in diff_131]
        merged = np.stack(seq_94 + seq_131, axis=0)  # Shape: [40, H, W]

        # Save the merged sequence to a single .npy file
        np.save(os.path.join(event_dir, "diff.npy"), merged)

    except Exception as e:
        print(f"[ERROR] Failed to process {event_id}: {e}")
