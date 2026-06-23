"""
Frequency Attention Module (FAM)
Applies channel-wise attention in the frequency domain, enabling the network to
selectively emphasize or suppress different frequency bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class FAM(nn.Module):
    """
    Frequency Attention Module.
    Decomposes features into different frequency components via FFT,
    learns attention weights per frequency band, and reweights them.
    Also projects channels from in_ch to out_ch.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Channel projection (1x1 conv)
        self.channel_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Frequency attention weights — learnable per frequency band
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial refinement after frequency attention
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Post-processing in frequency domain
        self.freq_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _frequency_attention(self, x):
        """Apply attention in the frequency domain."""
        # FFT
        x_fft = fft.fft2(x, norm='forward')
        x_fft_shifted = fft.fftshift(x_fft, dim=(-2, -1))

        # Amplitude for attention computation
        amplitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)

        # Generate frequency attention from amplitude
        # Use global context to predict frequency importance
        B, C, H, W = amplitude.shape
        attn = self.freq_attn(amplitude)  # [B, C, 1, 1]

        # Apply attention to amplitude
        attended_amplitude = amplitude * attn

        # Reconstruct
        recon = attended_amplitude * torch.cos(phase) + 1j * attended_amplitude * torch.sin(phase)

        # Inverse FFT
        recon_shifted = fft.ifftshift(recon, dim=(-2, -1))
        recon_spatial = fft.ifft2(recon_shifted, norm='forward')
        recon_spatial = torch.real(recon_spatial)

        return recon_spatial

    def forward(self, x):
        # Project channels
        x_proj = self.channel_proj(x)

        # Frequency attention path
        freq_out = self._frequency_attention(x_proj)
        freq_out = self.freq_refine(freq_out)

        # Spatial refinement path
        spatial_out = self.refine(x_proj)

        # Combine
        out = freq_out + spatial_out
        return out
