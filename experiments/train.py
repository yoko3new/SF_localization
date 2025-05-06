import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.solar_flare_dataset import SolarFlareDataset, get_dataloader
from models.unet import UNet
from utils.losses import heatmap_loss
from utils.train_utils import train_one_epoch, validate, save_model, predict_heatmaps

# ---------------- Configuration ---------------- #
DATA_ROOT = "/data/kyang30/flare_localization"
DIFF_ROOT = f"{DATA_ROOT}/diff_images"
HEATMAP_ROOT = f"{DATA_ROOT}/hek_heatmap"
PSEUDO_ROOT = f"{DATA_ROOT}/pseudo_heatmap"
CHECKPOINT_DIR = "checkpoints"
SELECTED_PSEUDO_TXT = "splits/pseudo_selected.txt"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- Main Training Script ---------------- #
def main():
    """
    Entry point for training or inference.

    Supports three modes:
    - "supervised": Train the model using labeled heatmaps.
    - "pseudo": Use a trained model to generate pseudo-labels for unlabeled data.
    - "joint": Fine-tune the model using both labeled and pseudo-labeled data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["supervised", "pseudo", "joint"], help="Training stage to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=40, out_channels=1).to(device)

    if args.stage == "supervised":
        # Stage 1: Supervised pretraining using ground-truth labels
        train_file = "splits/train_labeled.txt"
        val_file = "splits/val_labeled.txt"

        train_loader = get_dataloader(train_file, DIFF_ROOT, HEATMAP_ROOT, batch_size=8)
        val_loader = get_dataloader(val_file, DIFF_ROOT, HEATMAP_ROOT, batch_size=8)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(1, 21):
            train_one_epoch(model, train_loader, optimizer, device)
            validate(model, val_loader, device)
            save_model(model, os.path.join(CHECKPOINT_DIR, "supervised_best.pth"))

    elif args.stage == "pseudo":
        # Stage 2: Generate pseudo-labels for unlabeled data
        unlabeled_file = "splits/pseudo_unlabeled.txt"
        loader = get_dataloader(unlabeled_file, DIFF_ROOT, heatmap_root=None, batch_size=1, shuffle=False)

        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "supervised_best.pth"), map_location=device))
        model.eval()

        predict_heatmaps(model, loader, device, output_dir=PSEUDO_ROOT)

    elif args.stage == "joint":
        # Stage 3: Semi-supervised fine-tuning using labeled and pseudo-labeled data
        joint_file = "splits/train_labeled.txt"
        pseudo_file = SELECTED_PSEUDO_TXT
        val_file = "splits/val_labeled.txt"

        train_loader = get_dataloader(
            joint_file, DIFF_ROOT, HEATMAP_ROOT, batch_size=8,
            pseudo_root=PSEUDO_ROOT, pseudo_list=pseudo_file
        )
        val_loader = get_dataloader(val_file, DIFF_ROOT, HEATMAP_ROOT, batch_size=8)

        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "supervised_best.pth"), map_location=device))
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(1, 21):
            train_one_epoch(model, train_loader, optimizer, heatmap_loss, device, epoch, pseudo_weight=0.3)
            validate(model, val_loader, heatmap_loss, device)
            save_model(model, os.path.join(CHECKPOINT_DIR, "joint_best.pth"))

if __name__ == '__main__':
    main()
