"""Inverse-variance depth fusion used by observation variant D."""

from __future__ import annotations

import torch


def fuse_depths(
    depths: torch.Tensor,
    variances: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse stacked depth maps with per-pixel variances.

    ``depths`` and ``variances`` have shape ``(S, ...)`` for S sensors.
    Returns ``(fused_depth, fused_variance)``. Invalid or non-finite inputs
    receive infinite variance so they do not pull the estimate.
    """
    if depths.shape != variances.shape:
        raise ValueError("depths and variances must have the same shape.")
    if depths.shape[0] < 2:
        raise ValueError("fuse_depths requires at least two stacked measurements.")

    variance = variances.clone()
    bad = ~torch.isfinite(depths) | ~torch.isfinite(variances) | (variances <= 0)
    if valid is not None:
        if valid.shape != depths.shape:
            raise ValueError("valid must match depths.")
        bad = bad | ~valid.to(dtype=torch.bool)
    variance = torch.where(bad, torch.full_like(variance, float("inf")), variance)
    precision = torch.where(torch.isfinite(variance), 1.0 / variance, torch.zeros_like(variance))
    precision_sum = precision.sum(dim=0)
    fused = torch.where(
        precision_sum > 0,
        (precision * depths).sum(dim=0) / precision_sum,
        torch.full_like(precision_sum, float("nan")),
    )
    fused_variance = torch.where(precision_sum > 0, 1.0 / precision_sum, torch.full_like(precision_sum, float("inf")))
    return fused, fused_variance
