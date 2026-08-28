"""Episode evaluation metrics for pruning variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab_pruning.task.success import CutSuccess


@dataclass(frozen=True)
class EpisodeMetrics:
    success: torch.Tensor
    cut_point_error_m: torch.Tensor
    perpendicularity_error_deg: torch.Tensor
    collisions: torch.Tensor
    steps_to_success: torch.Tensor


def episode_metrics(
    cut_success: CutSuccess,
    eef_position_w: torch.Tensor,
    cut_position_w: torch.Tensor,
    *,
    step_index: torch.Tensor,
    previous_success: torch.Tensor | None = None,
) -> EpisodeMetrics:
    success = cut_success.success
    error = torch.linalg.vector_norm(eef_position_w - cut_position_w, dim=-1)
    first_success = success if previous_success is None else success & ~previous_success
    steps = torch.where(first_success, step_index.to(dtype=error.dtype), torch.full_like(error, float("inf")))
    return EpisodeMetrics(
        success=success,
        cut_point_error_m=error,
        perpendicularity_error_deg=cut_success.perpendicularity_error_deg,
        collisions=(~cut_success.collision_free).to(dtype=error.dtype),
        steps_to_success=steps,
    )


def success_vs_cut_error(
    successes: torch.Tensor,
    injected_error_m: torch.Tensor,
    bin_edges_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-bin success rate for the injected cut-point-error sweep."""
    if successes.shape != injected_error_m.shape:
        raise ValueError("successes and injected_error_m must match.")
    rates = []
    centers = []
    for left, right in zip(bin_edges_m[:-1], bin_edges_m[1:], strict=True):
        mask = (injected_error_m >= left) & (injected_error_m < right)
        nan = torch.tensor(float("nan"), device=successes.device)
        rates.append(torch.where(mask.any(), successes[mask].float().mean(), nan))
        centers.append(0.5 * (left + right))
    return torch.stack(centers), torch.stack(rates)
