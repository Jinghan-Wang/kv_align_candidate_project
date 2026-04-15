from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from utils.io_utils import IMG_EXTS, basename_no_ext, binarize_mask, load_array, percentile_normalize, to_tensor
from utils.image_ops import canonicalize_by_mask


@dataclass
class CasePaths:
    key: str
    kv: Optional[str]
    drr: Optional[str]
    spine: Optional[str]
    bone: Optional[str]


def _index_dir(dir_path: str) -> Dict[str, str]:
    out = {}
    if not os.path.isdir(dir_path):
        return out
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(dir_path, f'*{ext}')))
    for p in sorted(files):
        out[basename_no_ext(p)] = p
    return out


def collect_cases(data_root: str, split: str) -> List[CasePaths]:
    base = os.path.join(data_root, split)
    kv_map = _index_dir(os.path.join(base, 'kv'))
    drr_map = _index_dir(os.path.join(base, 'drr'))
    spine_map = _index_dir(os.path.join(base, 'spine'))
    bone_map = _index_dir(os.path.join(base, 'bone'))

    keys = sorted(set(kv_map.keys()) | set(drr_map.keys()) | set(spine_map.keys()) | set(bone_map.keys()))
    cases: List[CasePaths] = []
    for k in keys:
        cases.append(CasePaths(
            key=k,
            kv=kv_map.get(k),
            drr=drr_map.get(k),
            spine=spine_map.get(k),
            bone=bone_map.get(k),
        ))
    return cases


class PairedSpineDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, require_kv: bool = True, require_drr: bool = True, require_spine: bool = True):
        self.cfg = cfg
        self.split = split
        self.data_root = cfg['paths']['data_root']
        self.norm_cfg = cfg['normalization']
        self.cases = collect_cases(self.data_root, split)
        filtered = []
        for c in self.cases:
            if require_kv and c.kv is None:
                continue
            if require_drr and c.drr is None:
                continue
            if require_spine and c.spine is None:
                continue
            filtered.append(c)
        self.cases = filtered
        if len(self.cases) == 0:
            raise RuntimeError(f'No cases found for split={split} under {self.data_root}')

    def __len__(self) -> int:
        return len(self.cases)

    def _load_kv(self, path: str) -> torch.Tensor:
        arr = load_array(path)
        arr = percentile_normalize(arr, self.norm_cfg['kv_percentile_low'], self.norm_cfg['kv_percentile_high'])
        return to_tensor(arr)

    def _load_drr(self, path: str) -> torch.Tensor:
        arr = load_array(path)
        arr = percentile_normalize(arr, self.norm_cfg['drr_percentile_low'], self.norm_cfg['drr_percentile_high'])
        return to_tensor(arr)

    def _load_mask(self, path: str) -> torch.Tensor:
        arr = load_array(path)
        arr = binarize_mask(arr, thr=0.5)
        return to_tensor(arr)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | None]:
        c = self.cases[index]
        out: Dict[str, torch.Tensor | str | None] = {'key': c.key, 'kv_path': c.kv, 'drr_path': c.drr, 'spine_path': c.spine, 'bone_path': c.bone}
        if c.kv is not None:
            out['kv'] = self._load_kv(c.kv)
        if c.drr is not None:
            out['drr'] = self._load_drr(c.drr)
        if c.spine is not None:
            out['spine'] = self._load_mask(c.spine)
        if c.bone is not None:
            # bone may be image-like or mask-like; normalize to [0,1]
            arr = load_array(c.bone)
            arr = percentile_normalize(arr, self.norm_cfg['drr_percentile_low'], self.norm_cfg['drr_percentile_high'])
            out['bone'] = to_tensor(arr)
        else:
            ref = out.get('drr', out.get('kv'))
            if isinstance(ref, torch.Tensor):
                out['bone'] = torch.zeros_like(ref)
            else:
                raise RuntimeError('Bone missing and no reference tensor available to create placeholder.')
        return out


class TeacherCanonicalDataset(PairedSpineDataset):
    def __init__(self, cfg: Dict, split: str):
        super().__init__(cfg, split, require_kv=False, require_drr=True, require_spine=True)
        self.img_cfg = cfg['image']

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        base = super().__getitem__(index)
        drr = base['drr']
        spine = base['spine']
        assert isinstance(drr, torch.Tensor) and isinstance(spine, torch.Tensor)
        drr_can = canonicalize_by_mask(
            drr.unsqueeze(0),
            spine.unsqueeze(0),
            out_hw=(self.img_cfg['canonical_h'], self.img_cfg['canonical_w']),
            margin_y=self.img_cfg['canonical_margin_y'],
            margin_x=self.img_cfg['canonical_margin_x'],
            mode='bilinear',
        ).squeeze(0)
        spine_can = canonicalize_by_mask(
            spine.unsqueeze(0),
            spine.unsqueeze(0),
            out_hw=(self.img_cfg['canonical_h'], self.img_cfg['canonical_w']),
            margin_y=self.img_cfg['canonical_margin_y'],
            margin_x=self.img_cfg['canonical_margin_x'],
            mode='nearest',
        ).squeeze(0)
        return {
            'key': base['key'],
            'drr_can': drr_can,
            'spine_can': spine_can,
        }
