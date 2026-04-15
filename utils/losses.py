from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.image_ops import soft_centroid_x


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def bce_dice_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_with_logits(logits, target)
    return bce + dice


def bce_dice_prob(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(prob, target)
    dice = soft_dice_loss(prob, target)
    return bce + dice


def rowwidth_l1(prob_or_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    p = prob_or_mask.sum(dim=3).squeeze(1)
    t = target_mask.sum(dim=3).squeeze(1)
    return F.l1_loss(p, t)


def axis_smooth_loss(prob: torch.Tensor) -> torch.Tensor:
    xc = soft_centroid_x(prob)
    if xc.shape[1] < 3:
        return torch.tensor(0.0, device=prob.device, dtype=prob.dtype)
    second = xc[:, 2:] - 2 * xc[:, 1:-1] + xc[:, :-2]
    return second.abs().mean()


def listwise_ce(scores: torch.Tensor, pos_index: int) -> torch.Tensor:
    target = torch.full((scores.shape[0],), pos_index, dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


def ranking_margin_loss(pos_score: torch.Tensor, neg_scores: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    # pos_score: [B], neg_scores: [B,N]
    loss = F.relu(margin - pos_score.unsqueeze(1) + neg_scores)
    return loss.mean()
