"""
Spectral Residual Module (SRM)
Enhances feature representations by computing spectral residual in the frequency domain.
The spectral residual highlights frequency anomalies, which are useful for detecting
camouflaged objects that blend with the background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class SRM(nn.Module):
    """
    Spectral Residual Module.
    Uses FFT to extract spectral residual — the difference between log amplitude
    and its averaged counterpart — to emphasize novel frequency patterns.
    """
    def __init__(self, channels, kernel_size=3):
        super(SRM, self).__init__()
        self.channels = channels

        # Post-processing convolution after spectral residual
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Gating for adaptive fusion
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def _spectral_residual(self, x):
        """Compute spectral residual in frequency domain."""
        # FFT: shift to center low frequencies
        x_fft = fft.fft2(x, norm='forward')
        x_fft_shifted = fft.fftshift(x_fft, dim=(-2, -1))

        # Amplitude and phase
        amplitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)

        # Log amplitude
        log_amplitude = torch.log(amplitude + 1e-8)

        # Local average of log amplitude (spectral residual)
        avg_log_amplitude = F.avg_pool2d(log_amplitude, kernel_size=3, stride=1, padding=1)
        spectral_residual = log_amplitude - avg_log_amplitude

        # Reconstruct using original phase + residual amplitude
        residual_amplitude = torch.exp(spectral_residual)
        recon = residual_amplitude * torch.cos(phase) + 1j * residual_amplitude * torch.sin(phase)

        # Inverse FFT
        recon_shifted = fft.ifftshift(recon, dim=(-2, -1))
        recon_spatial = fft.ifft2(recon_shifted, norm='forward')
        recon_spatial = torch.real(recon_spatial)

        return recon_spatial

    def forward(self, x):
        identity = x
        # Spectral residual branch
        freq_out = self._spectral_residual(x)
        freq_out = self.conv(freq_out)

        # Adaptive gated fusion with original features
        gate_input = torch.cat([identity, freq_out], dim=1)
        g = self.gate(gate_input)
        out = g * freq_out + (1 - g) * identity

        return out
