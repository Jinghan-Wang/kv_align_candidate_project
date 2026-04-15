from __future__ import annotations

import torch


def dice_score(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (prob > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (den + eps)).mean()
