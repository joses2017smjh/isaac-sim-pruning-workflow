"""Ground-truth cut-point selection from cylinder metadata."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from isaaclab_pruning.geometry.cylinders import Cylinder


@dataclass(frozen=True)
class CutPoint:
    """A metric cut target in world coordinates."""

    record_id: str
    part_name: str
    position_w: np.ndarray
    axis_w: np.ndarray
    radius_m: float
    length_m: float
    neighbor_count: int
    confidence: float = 1.0


def neighbor_counts(
    cylinders: Iterable[Cylinder],
    *,
    neighborhood_radius_m: float = 0.10,
) -> np.ndarray:
    """Count other cylinder centroids within ``neighborhood_radius_m``."""
    if neighborhood_radius_m <= 0:
        raise ValueError("neighborhood_radius_m must be positive.")
    records = list(cylinders)
    if not records:
        return np.zeros(0, dtype=np.int64)
    centroids = np.stack([cylinder.centroid for cylinder in records])
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    return (distances <= neighborhood_radius_m).sum(axis=1).astype(np.int64) - 1


def oracle_cut_candidates(
    cylinders: Iterable[Cylinder],
    *,
    organ_classes: Iterable[str] = ("spur",),
    neighborhood_radius_m: float = 0.10,
) -> list[CutPoint]:
    """Return GT cuts, thickest then least crowded first.

    Phase 3 curriculum starts on thick, exposed wood and progresses to occluded
    thin spurs. This sort order is that curriculum, not a learned policy.
    """
    records = list(cylinders)
    allowed = {name.lower() for name in organ_classes}
    counts = neighbor_counts(records, neighborhood_radius_m=neighborhood_radius_m)
    candidates = [
        CutPoint(
            record_id=cylinder.record_id,
            part_name=cylinder.part_name,
            position_w=np.asarray(cylinder.centroid, dtype=np.float64),
            axis_w=np.asarray(cylinder.orientation, dtype=np.float64),
            radius_m=cylinder.radius,
            length_m=cylinder.length,
            neighbor_count=int(count),
            confidence=1.0,
        )
        for cylinder, count in zip(records, counts, strict=True)
        if cylinder.organ_class in allowed
    ]
    candidates.sort(key=lambda item: (-item.radius_m, item.neighbor_count, item.part_name))
    return candidates


def oracle_cut_point(
    cylinders: Iterable[Cylinder],
    *,
    organ_classes: Iterable[str] = ("spur",),
    neighborhood_radius_m: float = 0.10,
) -> CutPoint:
    """Return the first curriculum cut, or raise if none match."""
    candidates = oracle_cut_candidates(
        cylinders,
        organ_classes=organ_classes,
        neighborhood_radius_m=neighborhood_radius_m,
    )
    if not candidates:
        raise ValueError("No cylinders matched the requested organ classes.")
    return candidates[0]
