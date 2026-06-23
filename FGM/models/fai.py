"""
Frequency-Aware Interaction Module (FAI)
Bidirectional interaction between frequency-domain and spatial-domain features,
enabling the model to leverage complementary information from both domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class FAI(nn.Module):
    """
    Frequency-Aware Interaction Module.
    Performs bidirectional feature interaction:
      - Spatial → Frequency: spatial features guide frequency reweighting
      - Frequency → Spatial: frequency features enhance spatial details
    Then projects channels from in_ch to out_ch.
    """
    def __init__(self, in_channels, out_channels):
        super(FAI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Channel projection
        self.channel_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Spatial-to-frequency guidance (1x1 conv, no BN — operates on pooled 1x1 features)
        self.spatial_to_freq = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Frequency-to-spatial enhancement (1x1 conv, no BN)
        self.freq_to_spatial = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _frequency_branch(self, x):
        """Compute frequency-domain features."""
        x_fft = fft.fft2(x, norm='forward')
        x_fft_shifted = fft.fftshift(x_fft, dim=(-2, -1))

        amplitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)

        # Learnable frequency features
        log_amplitude = torch.log(amplitude + 1e-8)
        freq_feat = torch.cat([log_amplitude, phase], dim=1)

        return freq_feat, amplitude, phase, x_fft_shifted

    def _inverse_fft(self, amplitude, phase):
        """Reconstruct spatial features from amplitude and phase."""
        recon = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
        recon_shifted = fft.ifftshift(recon, dim=(-2, -1))
        recon_spatial = fft.ifft2(recon_shifted, norm='forward')
        return torch.real(recon_spatial)

    def forward(self, x):
        # Project channels
        x_proj = self.channel_proj(x)

        # Frequency branch
        freq_feat, amplitude, phase, x_fft = self._frequency_branch(x_proj)

        # Spatial-to-frequency guidance: use spatial context to modulate frequency
        spatial_context = F.adaptive_avg_pool2d(x_proj, 1)
        freq_gate = self.spatial_to_freq(spatial_context)
        modulated_amplitude = amplitude * freq_gate

        # Frequency-to-spatial enhancement: use frequency info to enhance spatial
        freq_context = F.adaptive_avg_pool2d(freq_feat, 1)
        freq_context_2d = freq_context.view(freq_context.size(0), -1, 1, 1)
        # Take first out_channels channels as spatial gate
        spatial_gate = self.freq_to_spatial(freq_context_2d[:, :self.out_channels, :, :])

        # Reconstruct frequency-enhanced features
        freq_enhanced = self._inverse_fft(modulated_amplitude, phase)
        spatial_enhanced = x_proj * spatial_gate

        # Fuse and refine
        out = self.refine(freq_enhanced + spatial_enhanced)
        return out
