"""Composite losses for log-mel prediction (L1, SSIM, multi-scale)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def log_mel_l1(pred: Tensor, target: Tensor) -> Tensor:
    """L1 on HiFi-GAN log-mel (dynamic-range-compressed log magnitudes)."""
    return F.l1_loss(pred, target)


def _gaussian_kernel_2d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def _ssim_map(
    pred: Tensor,
    target: Tensor,
    window: Tensor,
    c1: float,
    c2: float,
) -> Tensor:
    """SSIM map for ``(B, 1, H, W)`` tensors in ``[0, 1]``."""
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(pred, window, padding=pad)
    mu_y = F.conv2d(target, window, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return num / (den + 1e-8)


def mel_ssim_loss(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """
    ``1 - SSIM`` on mel as a 2D map ``(n_mels, time)``.

    Inputs are log-mel; each item is min–max normalized to ``[0, 1]`` for SSIM.
    """
    b = pred.shape[0]
    pred_n = []
    target_n = []
    for i in range(b):
        pair = torch.stack([pred[i], target[i]])
        lo = pair.min()
        hi = pair.max()
        span = (hi - lo).clamp_min(1e-5)
        pred_n.append((pred[i] - lo) / span)
        target_n.append((target[i] - lo) / span)
    pred_n = torch.stack(pred_n).unsqueeze(1)
    target_n = torch.stack(target_n).unsqueeze(1)

    window = _gaussian_kernel_2d(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(1, 1, window_size, window_size)
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    ssim = _ssim_map(pred_n, target_n, window, c1, c2).mean()
    return 1.0 - ssim


def multi_scale_log_mel_loss(
    pred: Tensor,
    target: Tensor,
    scales: tuple[int, ...] = (1, 2, 4, 8),
) -> Tensor:
    """Log-mel L1 at several temporal resolutions (avg-pool along time)."""
    time_len = pred.shape[-1]
    terms: list[Tensor] = []
    for s in scales:
        if s > time_len:
            continue
        if s == 1:
            p, t = pred, target
        else:
            p = F.avg_pool1d(pred, kernel_size=s, stride=s)
            t = F.avg_pool1d(target, kernel_size=s, stride=s)
        terms.append(F.l1_loss(p, t))
    if not terms:
        return log_mel_l1(pred, target)
    return torch.stack(terms).mean()


@dataclass
class MelLossWeights:
    log_l1: float = 1.0
    ssim: float = 0.2
    multi_scale: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {
            "log_l1": self.log_l1,
            "ssim": self.ssim,
            "multi_scale": self.multi_scale,
        }


def mel_training_loss(
    pred: Tensor,
    target: Tensor,
    weights: MelLossWeights,
) -> tuple[Tensor, dict[str, float]]:
    l_log = log_mel_l1(pred, target)
    l_ssim = mel_ssim_loss(pred, target)
    l_ms = multi_scale_log_mel_loss(pred, target)
    total = (
        weights.log_l1 * l_log
        + weights.ssim * l_ssim
        + weights.multi_scale * l_ms
    )
    parts = {
        "log_l1": float(l_log.detach()),
        "ssim": float(l_ssim.detach()),
        "multi_scale": float(l_ms.detach()),
        "total": float(total.detach()),
    }
    return total, parts
