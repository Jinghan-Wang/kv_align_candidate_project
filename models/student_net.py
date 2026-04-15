from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.blocks import ConvNormAct, DownBlock, UpBlock


class StudentNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_ch, base_ch, 3, 1),
            ConvNormAct(base_ch, base_ch, 3, 1),
        )
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            ConvNormAct(base_ch * 8, base_ch * 8, 3, 1),
            ConvNormAct(base_ch * 8, base_ch * 8, 3, 1),
        )

        self.up2 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up0 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.mask_head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_ch * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)

        u2 = self.up2(b, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)
        logits = self.mask_head(u0)
        conf_logit = self.conf_head(b)
        return {
            'logits': logits,
            'prob': torch.sigmoid(logits),
            'conf_logit': conf_logit,
            'conf': torch.sigmoid(conf_logit),
        }
