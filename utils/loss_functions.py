import sys

import torch
import torch.nn as nn

from config import CFG

sys.path.insert(0, "..")


# ====================================================
# Criterion - ['LabelSmoothing', 'FocalLoss' 'FocalCosineLoss', 'SymmetricCrossEntropyLoss',
#              'BiTemperedLoss', 'TaylorCrossEntropyLoss']
# ====================================================


def get_criterion():
    if CFG.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif CFG.criterion == "LabelSmoothing":
        criterion = LabelSmoothingLoss()
    return criterion


# ====================================================
# Label Smoothing
# ====================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=CFG.target_size, smoothing=CFG.smooth_alpha, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
