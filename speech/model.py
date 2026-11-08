"""CNN-BiLSTM: dysarthric log-mel to clear log-mel, same length."""

from __future__ import annotations

import torch
import torch.nn as nn


class CnnBiLstm(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        conv_channels: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        residual: bool = False,
        gated: bool = False,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.residual = bool(residual)
        self.gated = bool(gated) and self.residual
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Conv1d(lstm_hidden * 2, n_mels, kernel_size=1)
        if self.residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        if self.gated:
            self.gate = nn.Conv1d(lstm_hidden * 2, 1, kernel_size=1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: ``(batch, n_mels, time)``
        Returns:
            log-mel ``(batch, n_mels, time)``
        """
        hidden = self.conv(mel)
        hidden = hidden.transpose(1, 2)
        hidden, _ = self.lstm(hidden)
        hidden = hidden.transpose(1, 2)
        delta = self.head(hidden)
        if not self.residual:
            return delta
        if not self.gated:
            return mel + delta
        gate = torch.sigmoid(self.gate(hidden))
        self.last_gate = gate
        return mel + gate * delta
