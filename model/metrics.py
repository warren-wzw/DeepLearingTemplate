import torch
from torchmetrics import AverageMeter

"""StereoMatching Common evaluation methods"""
class EPEMetric(AverageMeter):
    def __init__(self):
        super().__init__()

    def update(self, pred, target, mask):
        error = torch.abs(target - pred) * mask
        error = torch.flatten(error, 1).sum(-1)
        count = torch.flatten(mask, 1).sum(-1)
        epe = error / count
        super().update(epe[count > 0])

class RateMetric(AverageMeter):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def update(self, pred, target, mask):
        error = torch.abs(target - pred)
        error = (error < self.threshold) & mask
        error = torch.flatten(error, 1).float().sum(-1)
        count = torch.flatten(mask, 1).sum(-1)
        rate = error / count * 100
        super().update(rate[count > 0])