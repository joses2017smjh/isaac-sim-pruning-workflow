"""Dense pruning reward and weight ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab_pruning.task.success import CutSuccess


@dataclass(frozen=True)
class RewardWeights:
    distance: float = 1.0
    align: float = 0.5
    success: float = 10.0
    collision: float = 5.0
    action: float = 0.01
    time: float = 0.01


def dense_pruning_reward(
    eef_position_w: torch.Tensor,
    cut_position_w: torch.Tensor,
    cut_success: CutSuccess,
    actions: torch.Tensor,
    *,
    weights: RewardWeights | None = None,
) -> torch.Tensor:
    """Approach, perpendicularity, terminal success, collision, smoothness."""
    cfg = weights or RewardWeights()
    distance = torch.linalg.vector_norm(eef_position_w - cut_position_w, dim=-1)
    align = torch.cos(torch.deg2rad(cut_success.perpendicularity_error_deg)).abs()
    action_penalty = torch.sum(actions.square(), dim=-1)
    collision = (~cut_success.collision_free).to(dtype=eef_position_w.dtype)
    success = cut_success.success.to(dtype=eef_position_w.dtype)
    return (
        cfg.distance * (-distance)
        + cfg.align * align
        + cfg.success * success
        - cfg.collision * collision
        - cfg.action * action_penalty
        - cfg.time
    )
