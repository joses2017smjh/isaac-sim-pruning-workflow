"""Batched VL53L8CX-inspired noise for ray-cast ranges."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class ToFStatus(IntEnum):
    VALID = 0
    OUT_OF_RANGE = 1
    RANDOM_DROPOUT = 2
    THIN_TARGET_DROPOUT = 3


@dataclass(frozen=True)
class ToFNoiseConfig:
    """Initial, deliberately explicit ToF domain-randomization parameters."""

    min_range_m: float = 0.03
    max_range_m: float = 3.4
    min_sigma_m: float = 0.003
    range_sigma_fraction: float = 0.03
    dropout_probability: float = 0.05
    thin_radius_threshold_m: float = 0.008
    thin_dropout_probability: float = 0.30
    invalid_value: float = float("nan")

    def __post_init__(self) -> None:
        if not 0 < self.min_range_m < self.max_range_m:
            raise ValueError("Expected 0 < min_range_m < max_range_m.")
        if self.min_sigma_m < 0 or self.range_sigma_fraction < 0:
            raise ValueError("Noise scales must be non-negative.")
        for name in ("dropout_probability", "thin_dropout_probability"):
            probability = getattr(self, name)
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.thin_radius_threshold_m <= 0:
            raise ValueError("thin_radius_threshold_m must be positive.")


@dataclass(frozen=True)
class ToFObservation:
    range_m: torch.Tensor
    variance_m2: torch.Tensor
    valid: torch.Tensor
    status: torch.Tensor


def _random_like(values: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )


def apply_tof_noise(
    ranges_m: torch.Tensor,
    *,
    hit_radii_m: torch.Tensor | None = None,
    config: ToFNoiseConfig | None = None,
    generator: torch.Generator | None = None,
) -> ToFObservation:
    """Apply range-dependent noise and status-aware dropout to arbitrary batches.

    ``hit_radii_m`` should come from the ray hit's cylinder metadata. If it is
    unavailable, only range noise and random zone dropout are applied.
    """
    cfg = config or ToFNoiseConfig()
    if not torch.is_floating_point(ranges_m):
        raise TypeError("ranges_m must be a floating-point tensor.")

    finite_in_range = torch.isfinite(ranges_m) & (ranges_m >= cfg.min_range_m) & (ranges_m <= cfg.max_range_m)
    sigma = torch.maximum(
        torch.full_like(ranges_m, cfg.min_sigma_m),
        ranges_m.abs() * cfg.range_sigma_fraction,
    )
    gaussian = torch.randn(
        ranges_m.shape,
        dtype=ranges_m.dtype,
        device=ranges_m.device,
        generator=generator,
    )
    measured = ranges_m + gaussian * sigma

    random_dropout = finite_in_range & (_random_like(ranges_m, generator) < cfg.dropout_probability)
    thin_dropout = torch.zeros_like(finite_in_range)
    if hit_radii_m is not None:
        radii = torch.broadcast_to(hit_radii_m.to(device=ranges_m.device), ranges_m.shape)
        thin_target = torch.isfinite(radii) & (radii < cfg.thin_radius_threshold_m)
        thin_dropout = (
            finite_in_range
            & ~random_dropout
            & thin_target
            & (_random_like(ranges_m, generator) < cfg.thin_dropout_probability)
        )

    measured_in_range = torch.isfinite(measured) & (measured >= cfg.min_range_m) & (measured <= cfg.max_range_m)
    valid = finite_in_range & measured_in_range & ~random_dropout & ~thin_dropout

    status = torch.full(
        ranges_m.shape,
        int(ToFStatus.OUT_OF_RANGE),
        dtype=torch.int8,
        device=ranges_m.device,
    )
    status[valid] = int(ToFStatus.VALID)
    status[random_dropout] = int(ToFStatus.RANDOM_DROPOUT)
    status[thin_dropout] = int(ToFStatus.THIN_TARGET_DROPOUT)

    invalid_fill = torch.full_like(measured, cfg.invalid_value)
    output_range = torch.where(valid, measured, invalid_fill)
    variance = torch.where(valid, sigma.square(), torch.full_like(sigma, float("inf")))
    return ToFObservation(
        range_m=output_range,
        variance_m2=variance,
        valid=valid,
        status=status,
    )
