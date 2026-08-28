"""Continuous robustness ladder ``d ∈ [0, 1]`` for the pruning domain."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LadderSample:
    tree_xy_m: torch.Tensor
    tree_yaw_rad: torch.Tensor
    tof_sigma_fraction: torch.Tensor
    tof_dropout: torch.Tensor
    thin_dropout: torch.Tensor
    camera_translation_m: torch.Tensor
    camera_rotation_rad: torch.Tensor
    joint_damping_scale: torch.Tensor
    cut_point_error_m: torch.Tensor
    bark_index: torch.Tensor
    lighting_intensity_scale: torch.Tensor


def lerp(progress: torch.Tensor, start: float, end: float) -> torch.Tensor:
    if torch.any((progress < 0) | (progress > 1)):
        raise ValueError("Ladder progress d must be in [0, 1].")
    return start + (end - start) * progress


def sample_ladder(
    progress: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> LadderSample:
    """Draw one sample per env at difficulty ``progress``.

    Axes at d=0 are nominal; at d=1 they match the plan's upper randomization.
    """
    d = progress.to(dtype=dtype)
    if device is not None:
        d = d.to(device=device)
    batch = d.shape[0]
    device = d.device

    def _uniform(low: torch.Tensor, high: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
        return low + (high - low) * torch.rand(size, device=device, dtype=dtype, generator=generator)

    xy_limit = lerp(d, 0.0, 0.15).unsqueeze(-1)
    yaw_limit = lerp(d, 0.0, float(torch.deg2rad(torch.tensor(10.0))))
    trans_limit = lerp(d, 0.0, 0.005).unsqueeze(-1)
    rot_limit = lerp(d, 0.0, float(torch.deg2rad(torch.tensor(2.0)))).unsqueeze(-1)
    damp_limit = lerp(d, 0.0, 0.30)
    return LadderSample(
        tree_xy_m=_uniform(-xy_limit, xy_limit, (batch, 2)),
        tree_yaw_rad=_uniform(-yaw_limit, yaw_limit, (batch,)),
        tof_sigma_fraction=lerp(d, 0.0, 0.05),
        tof_dropout=lerp(d, 0.0, 0.15),
        thin_dropout=lerp(d, 0.0, 0.50),
        camera_translation_m=_uniform(-trans_limit, trans_limit, (batch, 3)),
        camera_rotation_rad=_uniform(-rot_limit, rot_limit, (batch, 3)),
        joint_damping_scale=1.0 + _uniform(-damp_limit, damp_limit, (batch,)),
        cut_point_error_m=lerp(d, 0.0, 0.10),
        bark_index=(lerp(d, 0.0, 3.999).floor().clamp(max=3)).to(dtype=torch.long),
        lighting_intensity_scale=_uniform(lerp(d, 1.0, 0.3), lerp(d, 1.0, 3.0), (batch,)),
    )


def inject_cut_point_error(
    cut_position_w: torch.Tensor,
    error_m: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Offset the episode-start cut by a random direction of length ``error_m``."""
    noise = torch.randn(
        cut_position_w.shape,
        device=cut_position_w.device,
        dtype=cut_position_w.dtype,
        generator=generator,
    )
    direction = noise / torch.clamp(torch.linalg.vector_norm(noise, dim=-1, keepdim=True), min=1e-8)
    scale = error_m.unsqueeze(-1) if error_m.ndim == 1 else error_m
    return cut_position_w + direction * scale
