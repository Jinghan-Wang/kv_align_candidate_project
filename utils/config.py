from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: Dict[str, Any]) -> torch.device:
    name = str(cfg.get('device', 'cuda'))
    if name == 'cuda' and not torch.cuda.is_available():
        name = 'cpu'
    return torch.device(name)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
