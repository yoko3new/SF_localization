from torch.utils.data import DataLoader
from datasets.solar_flare_dataset import SolarFlareDataset

def get_dataloader(event_list_file, diff_root, heatmap_root=None, batch_size=8, shuffle=True, pseudo_root=None, pseudo_list=None):
    dataset = SolarFlareDataset(
        event_list_file,
        diff_root=diff_root,
        heatmap_root=heatmap_root,
        pseudo_root=pseudo_root,
        pseudo_list=pseudo_list
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)