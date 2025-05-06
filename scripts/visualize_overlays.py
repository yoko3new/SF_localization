"""
visualize_overlays.py

This script generates overlay visualizations of flare localization results
by superimposing heatmaps (Gaussian labels) onto preprocessed AIA difference image sequences.

It is primarily used for sanity-checking labeled samples in the training or test sets.
The resulting visualizations are saved to the specified output directory.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# Extend system path for local imports
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.train_utils import save_overlay

# ------------------------ Configuration ------------------------ #
EVENT_LIST_PATH = "splits/train_labeled.txt"  # Can also be test_labeled.txt
DIFF_DIR = "/data/kyang30/flare_localization/diff_images"
HEATMAP_DIR = "/data/kyang30/flare_localization/hek_heatmap"
OUTPUT_DIR = "logs/overlays"

# ------------------------ Load Event List ------------------------ #
with open(EVENT_LIST_PATH, "r") as f:
    event_ids = [line.strip() for line in f if line.strip()]

print(f"Visualizing {len(event_ids)} events...")

# ------------------------ Visualization Loop ------------------------ #
for event_id in tqdm(event_ids, desc="Generating overlays"):
    try:
        diff_path = os.path.join(DIFF_DIR, event_id, "diff.npy")
        heatmap_path = os.path.join(HEATMAP_DIR, event_id, "heatmap.npy")

        if not (os.path.exists(diff_path) and os.path.exists(heatmap_path)):
            print(f"Skipping {event_id}: missing diff.npy or heatmap.npy")
            continue

        diff_seq = np.load(diff_path)
        heatmap = np.load(heatmap_path)
        save_overlay(diff_seq, heatmap, event_id, OUTPUT_DIR)

    except Exception as e:
        print(f"Failed to visualize {event_id}: {e}")

print(f"Done. Overlays saved to: {OUTPUT_DIR}")
