from __future__ import annotations

import random
from typing import Dict, Optional

import torch

from utils.image_ops import gaussian_blur2d, soft_band


def _rand_range(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def make_synthetic_kv(
    drr: torch.Tensor,
    spine: torch.Tensor,
    bone: Optional[torch.Tensor],
    cfg_syn: Dict,
) -> torch.Tensor:
    # drr/spine/bone: [B,1,H,W], values expected in [0,1]
    device = drr.device
    dtype = drr.dtype

    a_drr = _rand_range(cfg_syn['drr_alpha_min'], cfg_syn['drr_alpha_max'])
    a_bone = _rand_range(cfg_syn['bone_alpha_min'], cfg_syn['bone_alpha_max']) if bone is not None else 0.0
    sigma = _rand_range(cfg_syn['blur_sigma_min'], cfg_syn['blur_sigma_max'])
    contrast = _rand_range(cfg_syn['contrast_min'], cfg_syn['contrast_max'])
    gamma = _rand_range(cfg_syn['gamma_min'], cfg_syn['gamma_max'])
    noise_std = _rand_range(cfg_syn['noise_std_min'], cfg_syn['noise_std_max'])
    shading_scale = _rand_range(cfg_syn['shading_scale_min'], cfg_syn['shading_scale_max'])

    x = a_drr * drr
    if bone is not None:
        x = x + a_bone * bone
    else:
        x = x + 0.25 * soft_band(spine, kernel=35)

    x = gaussian_blur2d(x, sigma=sigma)

    noise = torch.randn_like(x) * noise_std
    shading = torch.randn_like(x)
    shading = gaussian_blur2d(shading, sigma=max(sigma * 5.0, 5.0))
    shading = shading / shading.abs().max().clamp_min(1e-6)

    x = x * contrast + shading_scale * shading + noise
    x = x.clamp(0.0, 1.0)
    x = x.pow(gamma)

    # weak darkening around spine channel to mimic KV ambiguity rather than perfect visibility
    dark = gaussian_blur2d(soft_band(spine, kernel=41), sigma=max(sigma * 1.5, 1.5))
    x = (x * (1.0 - 0.15 * dark)).clamp(0.0, 1.0)
    return x.to(device=device, dtype=dtype)
