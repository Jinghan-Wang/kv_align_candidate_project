from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import ConvNormAct, DownBlock, ResBlock


class AlignScoreNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_ch, base_ch, 3, 1),
            ConvNormAct(base_ch, base_ch, 3, 1),
        )
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)
        self.res = nn.Sequential(
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_ch * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.res(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x).squeeze(1)
        return x
