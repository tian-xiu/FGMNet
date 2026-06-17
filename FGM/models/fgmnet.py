"""
FGMNet: Frequency-Guided Gated Mamba Network
Simple working implementation (lightweight) for development and testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .srm import SRM
from .fam import FAM
from .fai import FAI
from .cmfusion import CMFusion


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class FGMNet(nn.Module):
    """
    Minimal encoder-decoder segmentation network as a placeholder for the full FGMNet.
    Integrates SRM/FAM/FAI/CMFusion modules as lightweight components.
    Outputs a single-channel logit map.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super(FGMNet, self).__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c, c*2)
        self.enc3 = ConvBlock(c*2, c*4)

        # simple feature modules
        self.srm = SRM(c*4, c*4)
        self.fam = FAM(c*4, c*2)
        self.fai = FAI(c*2, c)
        self.cmf = CMFusion(c, c)

        # decoder
        self.up2 = UpConv(c*4, c*2)
        self.dec2 = ConvBlock(c*4, c*2)
        self.up1 = UpConv(c*2, c)
        self.dec1 = ConvBlock(c*2, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)        # [B, c, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 2c, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 4c, H/4, W/4]

        f = self.srm(e3)
        f = self.fam(f)
        f = self.fai(f)
        f = self.cmf(f)

        u2 = self.up2(f)  # [B, 2c, H/2, W/2]
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)  # [B, c, H, W]
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.head(d1)  # logits
        return out
