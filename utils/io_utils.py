from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

IMG_EXTS = ['.nii.gz', '.nii', '.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff']


def split_stem_and_ext(filename: str) -> Tuple[str, str]:
    lower = filename.lower()
    for ext in sorted(IMG_EXTS, key=len, reverse=True):
        if lower.endswith(ext):
            return filename[:-len(ext)], ext
    base, ext = os.path.splitext(filename)
    return base, ext


def basename_no_ext(path: str) -> str:
    return split_stem_and_ext(os.path.basename(path))[0]


def load_array(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith('.nii') or lower.endswith('.nii.gz'):
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[arr.shape[0] // 2]
        return arr.astype(np.float32)
    if lower.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32)
    img = Image.open(path).convert('F')
    arr = np.array(img, dtype=np.float32)
    return arr


def save_array_like_reference(path: str, arr: np.ndarray, ref_path: str | None = None) -> None:
    lower = path.lower()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = arr.astype(np.float32)
    if lower.endswith('.nii') or lower.endswith('.nii.gz'):
        img = sitk.GetImageFromArray(arr[None] if arr.ndim == 2 else arr)
        if ref_path and os.path.exists(ref_path) and (ref_path.lower().endswith('.nii') or ref_path.lower().endswith('.nii.gz')):
            ref = sitk.ReadImage(ref_path)
            img.CopyInformation(ref)
        sitk.WriteImage(img, path)
        return
    if lower.endswith('.npy'):
        np.save(path, arr)
        return
    vis = arr.copy()
    vis = vis - vis.min()
    denom = max(float(vis.max()), 1e-6)
    vis = (vis / denom * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(vis).save(path)


def percentile_normalize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def binarize_mask(arr: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (arr > thr).astype(np.float32)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D array, got shape {arr.shape}')
    return torch.from_numpy(arr[None]).float()
