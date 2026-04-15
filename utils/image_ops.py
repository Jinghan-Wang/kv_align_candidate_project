from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def shift_mask(mask: torch.Tensor, dx: int) -> torch.Tensor:
    # mask: [B,1,H,W]
    if dx == 0:
        return mask.clone()
    b, c, h, w = mask.shape
    out = torch.zeros_like(mask)
    if dx > 0:
        out[..., :, dx:] = mask[..., :, : w - dx]
    else:
        dx_abs = -dx
        out[..., :, : w - dx_abs] = mask[..., :, dx_abs:]
    return out


@torch.no_grad()
def shift_mask_list(mask: torch.Tensor, shifts: List[int]) -> List[torch.Tensor]:
    return [shift_mask(mask, dx) for dx in shifts]


def sigmoid_to_binary(x: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (x > thr).float()


def soft_band(mask: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    pad = kernel // 2
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)


def gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    radius = max(int(truncate * sigma + 0.5), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    return kernel


def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k1 = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    ky = k1.view(1, 1, -1, 1)
    kx = k1.view(1, 1, 1, -1)
    c = x.shape[1]
    ky = ky.repeat(c, 1, 1, 1)
    kx = kx.repeat(c, 1, 1, 1)
    pad_y = ky.shape[2] // 2
    pad_x = kx.shape[3] // 2
    x = F.conv2d(F.pad(x, (0, 0, pad_y, pad_y), mode='reflect'), ky, groups=c)
    x = F.conv2d(F.pad(x, (pad_x, pad_x, 0, 0), mode='reflect'), kx, groups=c)
    return x


def bbox_from_mask(mask: torch.Tensor, thr: float = 0.5) -> Tuple[int, int, int, int]:
    # mask: [1,H,W] or [H,W]
    if mask.ndim == 3:
        mask2 = mask[0]
    else:
        mask2 = mask
    idx = (mask2 > thr).nonzero(as_tuple=False)
    h, w = mask2.shape[-2], mask2.shape[-1]
    if idx.numel() == 0:
        return 0, h - 1, 0, w - 1
    y0 = int(idx[:, 0].min().item())
    y1 = int(idx[:, 0].max().item())
    x0 = int(idx[:, 1].min().item())
    x1 = int(idx[:, 1].max().item())
    return y0, y1, x0, x1


def expand_bbox(y0: int, y1: int, x0: int, x1: int, h: int, w: int, my: int, mx: int) -> Tuple[int, int, int, int]:
    y0 = max(0, y0 - my)
    y1 = min(h - 1, y1 + my)
    x0 = max(0, x0 - mx)
    x1 = min(w - 1, x1 + mx)
    return y0, y1, x0, x1


def crop_and_resize(tensor: torch.Tensor, bbox: Tuple[int, int, int, int], out_hw: Tuple[int, int], mode: str = 'bilinear') -> torch.Tensor:
    # tensor: [B,C,H,W]
    y0, y1, x0, x1 = bbox
    crop = tensor[..., y0:y1 + 1, x0:x1 + 1]
    if crop.shape[-2] < 2 or crop.shape[-1] < 2:
        crop = F.pad(crop, (0, max(0, 2 - crop.shape[-1]), 0, max(0, 2 - crop.shape[-2])))
    return F.interpolate(crop, size=out_hw, mode=mode, align_corners=False if mode in ('bilinear', 'bicubic') else None)


def build_scorer_input(
    kv: torch.Tensor,
    cand: torch.Tensor,
    scorer_hw: Tuple[int, int],
    margin_y: int,
    margin_x: int,
    band_kernel: int,
) -> torch.Tensor:
    # kv, cand: [B,1,H,W]
    outs = []
    b = kv.shape[0]
    band = soft_band(cand, kernel=band_kernel)
    for i in range(b):
        _, h, w = cand[i].shape
        y0, y1, x0, x1 = bbox_from_mask(band[i])
        y0, y1, x0, x1 = expand_bbox(y0, y1, x0, x1, h, w, margin_y, margin_x)
        kv_i = crop_and_resize(kv[i:i+1], (y0, y1, x0, x1), scorer_hw, mode='bilinear')
        cand_i = crop_and_resize(cand[i:i+1], (y0, y1, x0, x1), scorer_hw, mode='nearest')
        band_i = crop_and_resize(band[i:i+1], (y0, y1, x0, x1), scorer_hw, mode='nearest')
        outs.append(torch.cat([kv_i, cand_i, band_i], dim=1))
    return torch.cat(outs, dim=0)


def canonicalize_by_mask(
    image_or_mask: torch.Tensor,
    mask_for_bbox: torch.Tensor,
    out_hw: Tuple[int, int],
    margin_y: int,
    margin_x: int,
    mode: str = 'bilinear',
    thr: float = 0.2,
) -> torch.Tensor:
    # image_or_mask, mask_for_bbox: [B,1,H,W]
    outs = []
    for i in range(image_or_mask.shape[0]):
        _, h, w = image_or_mask[i].shape
        y0, y1, x0, x1 = bbox_from_mask(mask_for_bbox[i].detach(), thr=thr)
        y0, y1, x0, x1 = expand_bbox(y0, y1, x0, x1, h, w, margin_y, margin_x)
        outs.append(crop_and_resize(image_or_mask[i:i+1], (y0, y1, x0, x1), out_hw, mode=mode))
    return torch.cat(outs, dim=0)


def row_width_profile(mask_can: torch.Tensor) -> torch.Tensor:
    # [B,1,H,W] -> [B,H]
    return mask_can.sum(dim=3).squeeze(1)


def soft_centroid_x(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # [B,1,H,W] -> [B,H]
    b, _, h, w = mask.shape
    xs = torch.linspace(0, w - 1, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
    num = (mask * xs).sum(dim=3)
    den = mask.sum(dim=3).clamp_min(eps)
    return (num / den).squeeze(1)


def entropy_from_probs(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: [B,K]
    return -(q * q.clamp_min(eps).log()).sum(dim=1)
