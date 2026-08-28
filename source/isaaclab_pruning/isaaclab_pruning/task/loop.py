"""Episode-start perception vs high-rate control loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab_pruning.geometry.cut_point import CutPoint
from isaaclab_pruning.ladder.axes import inject_cut_point_error


@dataclass
class EpisodeTarget:
    position_w: torch.Tensor
    axis_w: torch.Tensor
    radius_m: torch.Tensor
    length_m: torch.Tensor
    confidence: torch.Tensor
    source: str


def episode_start_target(
    cut: CutPoint,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    injected_error_m: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    source: str = "oracle",
) -> EpisodeTarget:
    """Run once per episode. DA2-ft or the cylinder oracle belongs here, not in PPO."""
    position = torch.as_tensor(cut.position_w, device=device, dtype=dtype).expand(batch, 3).clone()
    if injected_error_m is not None:
        position = inject_cut_point_error(position, injected_error_m, generator)
    return EpisodeTarget(
        position_w=position,
        axis_w=torch.as_tensor(cut.axis_w, device=device, dtype=dtype).expand(batch, 3).clone(),
        radius_m=torch.full((batch,), float(cut.radius_m), device=device, dtype=dtype),
        length_m=torch.full((batch,), float(cut.length_m), device=device, dtype=dtype),
        confidence=torch.full((batch,), float(cut.confidence), device=device, dtype=dtype),
        source=source,
    )
