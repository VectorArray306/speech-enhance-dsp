"""1D U-Net: multi-channel EMG (time) -> log-mel (time)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EMGToMelCNN(nn.Module):
    """Encoder-decoder 1D U-Net. Skips are concatenated at each decoder stage."""

    def __init__(
        self,
        n_emg_channels: int = 8,
        n_mels: int = 80,
        base_channels: int = 128,
    ):
        super().__init__()
        self.n_emg_channels = n_emg_channels
        self.n_mels = n_mels
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.enc1 = ConvBlock(n_emg_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose1d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.head = nn.Conv1d(c1, n_mels, kernel_size=1)

    @staticmethod
    def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
        diff = length - x.size(-1)
        if diff > 0:
            return F.pad(x, (0, diff))
        if diff < 0:
            return x[..., :length]
        return x

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emg: ``(batch, channels, time)`` aligned to mel frames.
        Returns:
            log-mel ``(batch, n_mels, time)``.
        """
        length = emg.size(-1)
        # Three stride-2 pools need a multiple of 8 (and at least 8 frames).
        padded = max(8, ((length + 7) // 8) * 8)
        if padded != length:
            emg = F.pad(emg, (0, padded - length))

        e1 = self.enc1(emg)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self._match_length(d3, e3.size(-1))
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_length(d2, e2.size(-1))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_length(d1, e1.size(-1))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)[..., :length]
