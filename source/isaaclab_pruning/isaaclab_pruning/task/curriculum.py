"""Cut-point curriculum from cylinder radius and neighbourhood clutter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from isaaclab_pruning.geometry.cut_point import CutPoint, oracle_cut_candidates
from isaaclab_pruning.geometry.cylinders import Cylinder

# Thick exposed branches first, occluded thin spurs last.
STAGES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("branch",), 0.015),
    (("branch", "spur"), 0.008),
    (("spur",), 0.004),
    (("spur",), 0.0),
)


def curriculum_stage(progress: float) -> tuple[tuple[str, ...], float]:
    """Map ``progress`` in ``[0, 1]`` onto an organ-class / min-radius stage."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1].")
    index = min(int(progress * len(STAGES)), len(STAGES) - 1)
    return STAGES[index]


def select_curriculum_cut(
    cylinders: Sequence[Cylinder],
    progress: float,
) -> CutPoint:
    organ_classes, min_radius = curriculum_stage(progress)
    candidates = oracle_cut_candidates(cylinders, organ_classes=organ_classes)
    filtered = [candidate for candidate in candidates if candidate.radius_m >= min_radius]
    pool: Iterable[CutPoint] = filtered or candidates
    try:
        return next(iter(pool))
    except StopIteration as error:
        raise ValueError("No curriculum cut is available for this tree.") from error
