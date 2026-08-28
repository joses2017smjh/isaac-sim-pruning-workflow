"""Broad-phase tests of nearby wood against the cutter failure zone."""

from __future__ import annotations

import torch

from isaaclab_pruning.task.success import OrientedBox, segment_intersects_obb


def nearby_wood_in_failure_zone(
    centroids_w: torch.Tensor,
    axes_w: torch.Tensor,
    lengths_m: torch.Tensor,
    failure_box: OrientedBox,
    *,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return whether any non-target cylinder segment hits the failure box.

    Shapes: centroids/axes ``(N, M, 3)``, lengths ``(N, M)``, result ``(N,)``.
    """
    if centroids_w.shape != axes_w.shape or centroids_w.shape[-1] != 3:
        raise ValueError("centroids_w and axes_w must share shape (..., 3).")
    if lengths_m.shape != centroids_w.shape[:-1]:
        raise ValueError("lengths_m must match the cylinder batch without the last dim.")

    half = 0.5 * lengths_m.unsqueeze(-1) * axes_w
    starts = centroids_w - half
    ends = centroids_w + half
    batch, count, _ = centroids_w.shape
    hits = segment_intersects_obb(
        starts.reshape(batch * count, 3),
        ends.reshape(batch * count, 3),
        _broadcast_box(failure_box, count),
    )
    hits = hits.reshape(batch, count)
    if exclude_mask is not None:
        hits = hits & ~exclude_mask.to(dtype=torch.bool)
    return hits.any(dim=-1)


def _broadcast_box(box: OrientedBox, count: int) -> OrientedBox:
    batch = box.center_w.shape[0]
    return OrientedBox(
        center_w=box.center_w.repeat_interleave(count, dim=0),
        rotation_bw=box.rotation_bw.repeat_interleave(count, dim=0),
        half_extents=(
            box.half_extents.repeat_interleave(count, dim=0)
            if box.half_extents.ndim == 2
            else box.half_extents.expand(batch, 3).repeat_interleave(count, dim=0)
        ),
    )
