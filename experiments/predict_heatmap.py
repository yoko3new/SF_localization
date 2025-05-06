import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.solar_flare_dataset import SolarFlareDataset
from train_utils import save_overlay

# Set paths
DIFF_ROOT = "/data/kyang30/flare_localization/diff_images"
MODEL_PATH = "checkpoints/unet.pth"
OUTPUT_NPY_DIR = "outputs/heatmap_preds"
OUTPUT_IMG_DIR = "outputs/heatmap_vis"
EVENT_LIST_PATH = "splits/pseudo_unlabeled.txt"
SELECTED_LIST_PATH = "splits/pseudo_selected.txt"  # Path to save selected pseudo-labeled event IDs

# Inference and pseudo-label selection
def predict_heatmaps():
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    dataset = SolarFlareDataset(EVENT_LIST_PATH, diff_root=DIFF_ROOT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=21, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    selected_event_ids = []  # Store IDs of accepted pseudo-labeled events

    for batch in tqdm(dataloader, desc="Predicting"):
        event_id = batch['event_id'][0]
        diff = batch['diff'].to(device)  # [1, T, H, W]

        with torch.no_grad():
            pred = model(diff)  # [1, 1, H, W]
            heatmap = pred.squeeze().cpu().numpy()

        # Select high-confidence pseudo-labels
        peak_value = heatmap.max()
        central_crop = heatmap[254:258, 254:258]
        central_ratio = central_crop.sum() / (heatmap.sum() + 1e-6)

        if peak_value > 0.8 and central_ratio > 0.3:
            selected_event_ids.append(event_id)

            # Save heatmap as .npy
            np.save(os.path.join(OUTPUT_NPY_DIR, f"{event_id}.npy"), heatmap)

            # Generate and save overlay visualization
            diff_seq = batch['diff'].squeeze().cpu().numpy()  # [T, H, W]
            save_overlay(diff_seq, heatmap, event_id, output_dir=OUTPUT_IMG_DIR)

    # Save the list of selected event IDs
    os.makedirs(os.path.dirname(SELECTED_LIST_PATH), exist_ok=True)
    with open(SELECTED_LIST_PATH, "w") as f:
        for eid in selected_event_ids:
            f.write(eid + "\n")

    print(f"\nSelected {len(selected_event_ids)} high-quality pseudo-labeled events.")
    print(f"Heatmaps saved to: {OUTPUT_NPY_DIR}")
    print(f"Visualizations saved to: {OUTPUT_IMG_DIR}")
    print(f"Selected event list saved to: {SELECTED_LIST_PATH}")

if __name__ == '__main__':
    predict_heatmaps()
