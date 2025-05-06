import torch.nn.functional as F

def heatmap_loss(pred, target):
    return F.mse_loss(pred, target)