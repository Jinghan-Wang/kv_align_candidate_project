from __future__ import annotations

import argparse
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.student_net import StudentNet
from utils.config import ensure_dir, get_device, load_config
from utils.io_utils import IMG_EXTS, basename_no_ext, load_array, percentile_normalize, save_array_like_reference, to_tensor


def collect_inputs(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(path, f'*{ext}')))
        return sorted(files)
    return [path]


def save_overlay(img: np.ndarray, prob: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.imshow(prob, cmap='jet', alpha=0.35)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    ensure_dir(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = StudentNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    for path in collect_inputs(args.input):
        key = basename_no_ext(path)
        arr = load_array(path)
        norm = percentile_normalize(arr, cfg['normalization']['kv_percentile_low'], cfg['normalization']['kv_percentile_high'])
        x = to_tensor(norm).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            prob = out['prob'][0, 0].cpu().numpy().astype(np.float32)
            mask = (prob > float(cfg['image']['pred_bin_thresh'])).astype(np.float32)
            conf = float(out['conf'][0, 0].cpu().item())

        save_array_like_reference(os.path.join(args.output_dir, f'{key}_prob.npy'), prob, None)
        save_array_like_reference(os.path.join(args.output_dir, f'{key}_mask.npy'), mask, None)
        save_array_like_reference(os.path.join(args.output_dir, f'{key}_prob.nii.gz'), prob, path if path.lower().endswith('.nii') or path.lower().endswith('.nii.gz') else None)
        save_array_like_reference(os.path.join(args.output_dir, f'{key}_mask.nii.gz'), mask, path if path.lower().endswith('.nii') or path.lower().endswith('.nii.gz') else None)
        save_overlay(norm, prob, os.path.join(args.output_dir, f'{key}_overlay.png'))
        with open(os.path.join(args.output_dir, f'{key}_conf.txt'), 'w', encoding='utf-8') as f:
            f.write(f'pred_conf={conf:.6f}\n')
        print(f'[Infer] {key}: pred_conf={conf:.4f}')


if __name__ == '__main__':
    main()
