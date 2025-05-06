"""
quick_test_predict.py

This script loads a trained U-Net model and quickly tests it on the first few
unlabeled flare events to verify that:

1. Input tensors have expected shape and no NaN values.
2. Model outputs are valid (no NaN).
3. The pipeline runs end-to-end without crashing.

Useful for debugging corrupted inputs or model loading issues before full-scale inference.
"""

import torch
import numpy as np
from datasets.solar_flare_dataset import get_dataloader
from models.unet import UNet

# ----------------- Configuration ----------------- #
DIFF_ROOT = "/data/kyang30/flare_localization/diff_images"
UNLABELED_LIST = "splits/pseudo_unlabeled.txt"
CHECKPOINT = "checkpoints/supervised_best.pth"
MAX_SAMPLES = 5  # Only test first 5 samples

# ----------------- Model Setup ----------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=40, out_channels=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# ----------------- Data Loader ----------------- #
loader = get_dataloader(
    event_list_file=UNLABELED_LIST,
    diff_root=DIFF_ROOT,
    heatmap_root=None,
    batch_size=1,
    shuffle=False
)

# ----------------- Inference Loop ----------------- #
with torch.no_grad():
    for i, batch in enumerate(loader):
        event_id = batch["event_id"][0]
        diff = batch["diff"]

        print(f"[{i+1}] Event: {event_id} | Input shape: {diff.shape} | NaNs in input: {torch.isnan(diff).sum().item()}")

        output = model(diff.to(device))

        if torch.isnan(output).any():
            print(f"  NaN detected in output for {event_id}")
        else:
            print(f"  Output OK")

        if i + 1 >= MAX_SAMPLES:
            break
