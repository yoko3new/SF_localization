import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class SolarFlareDataset(Dataset):
    def __init__(self, event_list_file, diff_root, heatmap_root=None, transform=None, pseudo_root=None, pseudo_list=None):
        with open(event_list_file, 'r') as f:
            base_ids = [line.strip() for line in f if line.strip()]

        # If a pseudo-label list is provided, load only events in the intersection
        if pseudo_list and pseudo_root:
            with open(pseudo_list, 'r') as pf:
                pseudo_ids = [line.strip() for line in pf if line.strip()]
            self.event_ids = sorted(set(base_ids) & set(pseudo_ids))
        else:
            self.event_ids = base_ids

        self.diff_root = diff_root
        self.heatmap_root = heatmap_root
        self.pseudo_root = pseudo_root
        self.transform = transform

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        diff_path = os.path.join(self.diff_root, event_id, "diff.npy")
        diff_seq = np.load(diff_path).astype(np.float32)

        heatmap = None
        # Load ground-truth heatmap if available, otherwise load pseudo-label if provided
        if self.heatmap_root and os.path.exists(os.path.join(self.heatmap_root, event_id, "heatmap.npy")):
            heatmap = np.load(os.path.join(self.heatmap_root, event_id, "heatmap.npy")).astype(np.float32)
        elif self.pseudo_root and os.path.exists(os.path.join(self.pseudo_root, f"{event_id}.npy")):
            heatmap = np.load(os.path.join(self.pseudo_root, f"{event_id}.npy")).astype(np.float32)

        if self.transform:
            diff_seq, heatmap = self.transform(diff_seq, heatmap)

        return {
            'event_id': event_id,
            'diff': diff_seq,
            'heatmap': heatmap  # None if unlabeled
        }

from torch.utils.data import DataLoader

def custom_collate(batch):
    batch_dict = {}
    batch_dict['event_id'] = [d['event_id'] for d in batch]
    batch_dict['diff'] = torch.tensor(np.stack([d['diff'] for d in batch]), dtype=torch.float32)
    
    # Heatmaps may be None (for unlabeled samples)
    heatmaps = [d['heatmap'] for d in batch]
    if any(h is not None for h in heatmaps):
        batch_dict['heatmap'] = heatmaps
    else:
        batch_dict['heatmap'] = [None] * len(batch)

    return batch_dict

def get_dataloader(event_list_file, diff_root, heatmap_root=None, batch_size=8, shuffle=True, pseudo_root=None, pseudo_list=None):
    dataset = SolarFlareDataset(
        event_list_file=event_list_file,
        diff_root=diff_root,
        heatmap_root=heatmap_root,
        pseudo_root=pseudo_root,
        pseudo_list=pseudo_list
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
