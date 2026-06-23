"""
Cross-Modal Fusion Module (CMFusion)
Fuses multi-level or multi-modal features using a combination of
spatial and channel attention guided by frequency-domain cues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class CMFusion(nn.Module):
    """
    Cross-Modal Fusion Module.
    Fuses features by combining frequency-domain and spatial-domain information
    with channel-wise and spatial attention mechanisms.
    """
    def __init__(self, channels, reduction=4):
        super(CMFusion, self).__init__()
        self.channels = channels

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # Frequency gating
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Final gate
        self.final_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def _frequency_features(self, x):
        """Extract frequency-domain features."""
        x_fft = fft.fft2(x, norm='forward')
        x_fft_shifted = fft.fftshift(x_fft, dim=(-2, -1))

        amplitude = torch.abs(x_fft_shifted)
        return amplitude

    def forward(self, x):
        identity = x

        # Channel attention path
        ca = self.channel_attn(x)
        ca_out = x * ca

        # Spatial attention path
        sa = self.spatial_attn(x)
        sa_out = x * sa

        # Frequency gating path
        freq_amp = self._frequency_features(x)
        fg = self.freq_gate(freq_amp)
        freq_out = x * fg

        # Combine all paths
        combined = ca_out + sa_out + freq_out

        # Adaptive fusion with identity
        gate_input = torch.cat([combined, identity], dim=1)
        g = self.final_gate(gate_input)
        out = g * combined + (1 - g) * identity

        # Refinement
        out = self.refine(out)
        out = F.relu(out + identity)

        return out
