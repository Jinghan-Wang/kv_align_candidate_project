from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.paired_spine_dataset import TeacherCanonicalDataset
from models.teacher_shape_net import TeacherShapeNet
from utils.config import ensure_dir, get_device, load_config, set_seed
from utils.losses import bce_dice_with_logits, rowwidth_l1
from utils.metrics import dice_score


@torch.no_grad()
def evaluate(model: TeacherShapeNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        x = batch['drr_can'].to(device)
        y = batch['spine_can'].to(device)
        out = model(x)
        loss = bce_dice_with_logits(out['logits'], y) + 0.5 * rowwidth_l1(out['prob'], y)
        total_loss += float(loss.item())
        total_dice += float(dice_score(out['prob'], y).item())
        n += 1
    return {'loss': total_loss / max(n, 1), 'dice': total_dice / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['seed']))
    device = get_device(cfg)

    train_ds = TeacherCanonicalDataset(cfg, split='train')
    val_ds = TeacherCanonicalDataset(cfg, split='val')
    train_loader = DataLoader(train_ds, batch_size=int(cfg['teacher']['batch_size']), shuffle=True, num_workers=int(cfg['num_workers']))
    val_loader = DataLoader(val_ds, batch_size=int(cfg['teacher']['batch_size']), shuffle=False, num_workers=int(cfg['num_workers']))

    model = TeacherShapeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['teacher']['lr']), weight_decay=float(cfg['teacher']['weight_decay']))

    out_dir = os.path.join(cfg['paths']['output_root'], 'teacher')
    ensure_dir(out_dir)
    best_dice = -1.0

    for epoch in range(1, int(cfg['teacher']['epochs']) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Teacher {epoch}')
        for batch in pbar:
            x = batch['drr_can'].to(device)
            y = batch['spine_can'].to(device)
            out = model(x)
            loss = bce_dice_with_logits(out['logits'], y) + 0.5 * rowwidth_l1(out['prob'], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        val_stats = evaluate(model, val_loader, device)
        print(f'[Teacher][Epoch {epoch}] val_loss={val_stats["loss"]:.4f} val_dice={val_stats["dice"]:.4f}')

        ckpt = {
            'model': model.state_dict(),
            'cfg': cfg,
            'epoch': epoch,
            'val_dice': val_stats['dice'],
        }
        torch.save(ckpt, os.path.join(out_dir, 'last.pt'))
        if val_stats['dice'] > best_dice:
            best_dice = val_stats['dice']
            torch.save(ckpt, os.path.join(out_dir, 'best.pt'))
            print(f'[Teacher] saved new best with dice={best_dice:.4f}')


if __name__ == '__main__':
    main()
