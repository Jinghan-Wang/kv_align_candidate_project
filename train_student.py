from __future__ import annotations

import argparse
import math
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.paired_spine_dataset import PairedSpineDataset
from models.align_scorenet import AlignScoreNet
from models.student_net import StudentNet
from models.teacher_shape_net import TeacherShapeNet
from utils.config import ensure_dir, get_device, load_config, set_seed
from utils.image_ops import build_scorer_input, canonicalize_by_mask, entropy_from_probs, shift_mask_list
from utils.losses import axis_smooth_loss, bce_dice_prob, rowwidth_l1
from utils.metrics import dice_score


def load_frozen_teacher(cfg: dict, device: torch.device) -> TeacherShapeNet:
    ckpt = torch.load(cfg['paths']['teacher_ckpt'], map_location=device)
    model = TeacherShapeNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_frozen_scorer(cfg: dict, device: torch.device) -> AlignScoreNet:
    ckpt = torch.load(cfg['paths']['scorer_ckpt'], map_location=device)
    model = AlignScoreNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def compute_candidate_targets(
    scorer: AlignScoreNet,
    kv: torch.Tensor,
    spine_gt: torch.Tensor,
    cfg: dict,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    shifts = list(cfg['candidates']['shifts'])
    cand_list = shift_mask_list(spine_gt, shifts)
    scores = []
    for cand in cand_list:
        inp = build_scorer_input(
            kv,
            cand,
            scorer_hw=(cfg['image']['scorer_h'], cfg['image']['scorer_w']),
            margin_y=int(cfg['image']['scorer_margin_y']),
            margin_x=int(cfg['image']['scorer_margin_x']),
            band_kernel=int(cfg['image']['band_kernel']),
        )
        scores.append(scorer(inp))
    scores = torch.stack(scores, dim=1)  # [B,K]

    prior = torch.tensor([-(abs(dx) * float(cfg['candidates']['shift_prior_lambda'])) for dx in shifts], device=kv.device, dtype=kv.dtype)
    scores = scores + prior.unsqueeze(0)
    q = torch.softmax(scores / float(cfg['candidates']['tau']), dim=1)

    g_soft = torch.zeros_like(spine_gt)
    for i, cand in enumerate(cand_list):
        g_soft = g_soft + q[:, i:i+1, None, None] * cand

    entropy = entropy_from_probs(q)
    w_pos = 1.0 - entropy / math.log(len(shifts))
    return cand_list, scores, q, g_soft, w_pos


@torch.no_grad()
def teacher_targets(teacher: TeacherShapeNet, drr: torch.Tensor, spine_gt: torch.Tensor, cfg: dict) -> torch.Tensor:
    drr_can = canonicalize_by_mask(
        drr,
        spine_gt,
        out_hw=(cfg['image']['canonical_h'], cfg['image']['canonical_w']),
        margin_y=int(cfg['image']['canonical_margin_y']),
        margin_x=int(cfg['image']['canonical_margin_x']),
        mode='bilinear',
    )
    out = teacher(drr_can)
    return out['prob'].detach()


def evaluate(
    student: StudentNet,
    teacher: TeacherShapeNet,
    scorer: AlignScoreNet,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict:
    student.eval()
    total_loss = 0.0
    total_fixed_dice = 0.0
    total_bestcand_dice = 0.0
    total_conf = 0.0
    n = 0
    shifts = list(cfg['candidates']['shifts'])
    for batch in loader:
        kv = batch['kv'].to(device)
        drr = batch['drr'].to(device)
        spine_gt = batch['spine'].to(device)

        cand_list, scores, q, g_soft, w_pos = compute_candidate_targets(scorer, kv, spine_gt, cfg)
        out = student(kv)
        pred = out['prob']
        pred_conf = out['conf']

        l_pos = bce_dice_prob(pred, g_soft)
        l_pos = (w_pos.mean() * l_pos)
        teach_can = teacher_targets(teacher, drr, spine_gt, cfg)
        stud_can = canonicalize_by_mask(
            pred,
            pred,
            out_hw=(cfg['image']['canonical_h'], cfg['image']['canonical_w']),
            margin_y=int(cfg['image']['canonical_margin_y']),
            margin_x=int(cfg['image']['canonical_margin_x']),
            mode='bilinear',
            thr=float(cfg['image']['pred_bin_thresh']),
        )
        l_shape = bce_dice_prob(stud_can, teach_can) * float(cfg['student']['teacher_mask_weight'])
        l_shape = l_shape + float(cfg['student']['teacher_rowwidth_weight']) * rowwidth_l1(stud_can, teach_can)
        l_axis = axis_smooth_loss(pred)
        l_conf = F.binary_cross_entropy(pred_conf, w_pos.unsqueeze(1))
        loss = float(cfg['student']['lambda_pos']) * l_pos + float(cfg['student']['lambda_shape']) * l_shape + float(cfg['student']['lambda_axis']) * l_axis + float(cfg['student']['lambda_conf']) * l_conf

        fixed_dice = float(dice_score(pred, spine_gt, thr=float(cfg['image']['pred_bin_thresh'])).item())
        bestcand = 0.0
        for cand in cand_list:
            bestcand = max(bestcand, float(dice_score(pred, cand, thr=float(cfg['image']['pred_bin_thresh'])).item()))
        total_loss += float(loss.item())
        total_fixed_dice += fixed_dice
        total_bestcand_dice += bestcand
        total_conf += float(pred_conf.mean().item())
        n += 1
    return {
        'loss': total_loss / max(n, 1),
        'fixed_dice': total_fixed_dice / max(n, 1),
        'bestcand_dice': total_bestcand_dice / max(n, 1),
        'conf': total_conf / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['seed']))
    device = get_device(cfg)

    train_ds = PairedSpineDataset(cfg, split='train', require_kv=True, require_drr=True, require_spine=True)
    val_ds = PairedSpineDataset(cfg, split='val', require_kv=True, require_drr=True, require_spine=True)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['student']['batch_size']), shuffle=True, num_workers=int(cfg['num_workers']))
    val_loader = DataLoader(val_ds, batch_size=int(cfg['student']['batch_size']), shuffle=False, num_workers=int(cfg['num_workers']))

    teacher = load_frozen_teacher(cfg, device)
    scorer = load_frozen_scorer(cfg, device)
    student = StudentNet().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg['student']['lr']), weight_decay=float(cfg['student']['weight_decay']))

    out_dir = os.path.join(cfg['paths']['output_root'], 'student')
    ensure_dir(out_dir)
    best_metric = -1.0

    for epoch in range(1, int(cfg['student']['epochs']) + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f'Student {epoch}')
        for batch in pbar:
            kv = batch['kv'].to(device)
            drr = batch['drr'].to(device)
            spine_gt = batch['spine'].to(device)

            with torch.no_grad():
                cand_list, scores, q, g_soft, w_pos = compute_candidate_targets(scorer, kv, spine_gt, cfg)
                teach_can = teacher_targets(teacher, drr, spine_gt, cfg)

            out = student(kv)
            pred = out['prob']
            pred_conf = out['conf']

            l_pos = bce_dice_prob(pred, g_soft)
            l_pos = (w_pos.mean() * l_pos)

            stud_can = canonicalize_by_mask(
                pred,
                pred,
                out_hw=(cfg['image']['canonical_h'], cfg['image']['canonical_w']),
                margin_y=int(cfg['image']['canonical_margin_y']),
                margin_x=int(cfg['image']['canonical_margin_x']),
                mode='bilinear',
                thr=float(cfg['image']['pred_bin_thresh']),
            )
            l_shape = bce_dice_prob(stud_can, teach_can) * float(cfg['student']['teacher_mask_weight'])
            l_shape = l_shape + float(cfg['student']['teacher_rowwidth_weight']) * rowwidth_l1(stud_can, teach_can)
            l_axis = axis_smooth_loss(pred)
            l_conf = F.binary_cross_entropy(pred_conf, w_pos.unsqueeze(1))

            loss = float(cfg['student']['lambda_pos']) * l_pos + float(cfg['student']['lambda_shape']) * l_shape + float(cfg['student']['lambda_axis']) * l_axis + float(cfg['student']['lambda_conf']) * l_conf

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                wpos=f'{w_pos.mean().item():.3f}',
                conf=f'{pred_conf.mean().item():.3f}',
            )

        val_stats = evaluate(student, teacher, scorer, val_loader, device, cfg)
        print(
            f'[Student][Epoch {epoch}] '
            f'val_loss={val_stats["loss"]:.4f} '
            f'fixed_dice={val_stats["fixed_dice"]:.4f} '
            f'bestcand_dice={val_stats["bestcand_dice"]:.4f} '
            f'conf={val_stats["conf"]:.4f}'
        )
        metric = val_stats['bestcand_dice']
        ckpt = {'model': student.state_dict(), 'cfg': cfg, 'epoch': epoch, 'metric': metric, 'val_stats': val_stats}
        torch.save(ckpt, os.path.join(out_dir, 'last.pt'))
        if metric > best_metric:
            best_metric = metric
            torch.save(ckpt, os.path.join(out_dir, 'best.pt'))
            print(f'[Student] saved new best with bestcand_dice={best_metric:.4f}')


if __name__ == '__main__':
    main()
