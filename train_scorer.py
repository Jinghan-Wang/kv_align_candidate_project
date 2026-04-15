from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.paired_spine_dataset import PairedSpineDataset
from models.align_scorenet import AlignScoreNet
from utils.config import ensure_dir, get_device, load_config, set_seed
from utils.image_ops import build_scorer_input, shift_mask_list
from utils.losses import listwise_ce, ranking_margin_loss
from utils.synthetic_kv import make_synthetic_kv


def score_candidates(model: AlignScoreNet, kv: torch.Tensor, spine: torch.Tensor, cfg: dict) -> torch.Tensor:
    shifts = list(cfg['candidates']['shifts'])
    cands = shift_mask_list(spine, shifts)
    scores = []
    for cand in cands:
        inp = build_scorer_input(
            kv,
            cand,
            scorer_hw=(cfg['image']['scorer_h'], cfg['image']['scorer_w']),
            margin_y=int(cfg['image']['scorer_margin_y']),
            margin_x=int(cfg['image']['scorer_margin_x']),
            band_kernel=int(cfg['image']['band_kernel']),
        )
        s = model(inp)
        scores.append(s)
    return torch.stack(scores, dim=1)  # [B,K]


@torch.no_grad()
def evaluate(model: AlignScoreNet, loader: DataLoader, device: torch.device, cfg: dict) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    pos_idx = list(cfg['candidates']['shifts']).index(0)
    for batch in loader:
        drr = batch['drr'].to(device)
        spine = batch['spine'].to(device)
        bone = batch['bone'].to(device)
        bone_in = bone if bone.abs().max().item() > 0 else None
        kv_syn = make_synthetic_kv(drr, spine, bone_in, cfg['scorer']['synthetic'])
        scores = score_candidates(model, kv_syn, spine, cfg)
        ce = listwise_ce(scores, pos_idx)
        pos = scores[:, pos_idx]
        neg = torch.cat([scores[:, :pos_idx], scores[:, pos_idx + 1:]], dim=1)
        rank = ranking_margin_loss(pos, neg, margin=float(cfg['scorer']['rank_margin']))
        loss = float(cfg['scorer']['list_ce_weight']) * ce + float(cfg['scorer']['rank_weight']) * rank
        total_loss += float(loss.item())
        pred = scores.argmax(dim=1)
        total_acc += float((pred == pos_idx).float().mean().item())
        n += 1
    return {'loss': total_loss / max(n, 1), 'acc': total_acc / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['seed']))
    device = get_device(cfg)

    train_ds = PairedSpineDataset(cfg, split='train', require_kv=False, require_drr=True, require_spine=True)
    val_ds = PairedSpineDataset(cfg, split='val', require_kv=False, require_drr=True, require_spine=True)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['scorer']['batch_size']), shuffle=True, num_workers=int(cfg['num_workers']))
    val_loader = DataLoader(val_ds, batch_size=int(cfg['scorer']['batch_size']), shuffle=False, num_workers=int(cfg['num_workers']))

    model = AlignScoreNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['scorer']['lr']), weight_decay=float(cfg['scorer']['weight_decay']))

    out_dir = os.path.join(cfg['paths']['output_root'], 'scorer')
    ensure_dir(out_dir)
    best_acc = -1.0
    pos_idx = list(cfg['candidates']['shifts']).index(0)

    for epoch in range(1, int(cfg['scorer']['epochs']) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Scorer {epoch}')
        for batch in pbar:
            drr = batch['drr'].to(device)
            spine = batch['spine'].to(device)
            bone = batch['bone'].to(device)
            bone_in = bone if bone.abs().max().item() > 0 else None
            kv_syn = make_synthetic_kv(drr, spine, bone_in, cfg['scorer']['synthetic'])
            scores = score_candidates(model, kv_syn, spine, cfg)
            ce = listwise_ce(scores, pos_idx)
            pos = scores[:, pos_idx]
            neg = torch.cat([scores[:, :pos_idx], scores[:, pos_idx + 1:]], dim=1)
            rank = ranking_margin_loss(pos, neg, margin=float(cfg['scorer']['rank_margin']))
            loss = float(cfg['scorer']['list_ce_weight']) * ce + float(cfg['scorer']['rank_weight']) * rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc = (scores.argmax(dim=1) == pos_idx).float().mean().item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc:.3f}')

        val_stats = evaluate(model, val_loader, device, cfg)
        print(f'[Scorer][Epoch {epoch}] val_loss={val_stats["loss"]:.4f} val_acc={val_stats["acc"]:.4f}')
        ckpt = {'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch, 'val_acc': val_stats['acc']}
        torch.save(ckpt, os.path.join(out_dir, 'last.pt'))
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            torch.save(ckpt, os.path.join(out_dir, 'best.pt'))
            print(f'[Scorer] saved new best with acc={best_acc:.4f}')


if __name__ == '__main__':
    main()
