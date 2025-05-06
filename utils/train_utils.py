import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_one_epoch(model, dataloader, optimizer, device, use_pseudo=False, pseudo_weight=0.3):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        diff = batch['diff'].to(device)              # [B, T, H, W]
        event_ids = batch['event_id']
        heatmaps = batch['heatmap']

        optimizer.zero_grad()
        output = model(diff).squeeze(1)              # [B, H, W]

        if use_pseudo:
            loss = 0.0
            valid_count = 0
            for i in range(len(event_ids)):
                if heatmaps[i] is None:
                    continue
                h = torch.from_numpy(heatmaps[i]).float().to(device)
                weight = pseudo_weight if 'pseudo' in event_ids[i] else 1.0
                loss += weight * F.mse_loss(output[i], h)
                valid_count += 1
            if valid_count > 0:
                loss /= valid_count
            else:
                continue  # Skip this batch
        else:
            heatmap_tensor = torch.from_numpy(np.stack(heatmaps)).float().to(device)  # [B, H, W]
            loss = F.mse_loss(output, heatmap_tensor)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            diff = batch['diff'].to(device)
            heatmap_tensor = torch.from_numpy(np.stack(batch['heatmap'])).float().to(device)

            output = model(diff).squeeze(1)
            loss = F.mse_loss(output, heatmap_tensor)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_overlay(diff_seq, heatmap, event_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_img = diff_seq[10]  # 中间帧（假设 T=21）
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(base_img, cmap='gray', alpha=1.0)
    ax.imshow(heatmap, cmap='hot', alpha=0.5)
    ax.set_title(event_id)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{event_id}.png"))
    plt.close()


def predict_heatmaps(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating pseudo heatmaps"):
            event_id = batch['event_id'][0]
            diff = batch['diff'].to(device)
            pred = model(diff).squeeze().cpu().numpy()
            np.save(os.path.join(output_dir, f"{event_id}.npy"), pred)
