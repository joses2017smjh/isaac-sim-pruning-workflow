"""Isaac Lab → PyBullet sim2sim ranking comparison."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingResult:
    isaac_order: tuple[str, ...]
    pybullet_order: tuple[str, ...]
    inverted: bool
    inverted_pairs: tuple[tuple[str, str], ...]


def ranking_inversion(isaac_scores: dict[str, float], pybullet_scores: dict[str, float]) -> RankingResult:
    """True when the highest-Isaac policy is not the highest-PyBullet policy.

    This is the pruning-domain analogue of the humanoid ladder finding that
    the highest training reward fell 23% of the time under sim2sim.
    """
    if set(isaac_scores) != set(pybullet_scores):
        missing = set(isaac_scores) ^ set(pybullet_scores)
        raise ValueError(f"Variant keys do not match: {sorted(missing)}.")
    isaac_order = tuple(sorted(isaac_scores, key=isaac_scores.__getitem__, reverse=True))
    pybullet_order = tuple(sorted(pybullet_scores, key=pybullet_scores.__getitem__, reverse=True))
    inverted_pairs = []
    names = list(isaac_scores)
    for left in names:
        for right in names:
            if left >= right:
                continue
            isaac_left_better = isaac_scores[left] > isaac_scores[right]
            pybullet_left_better = pybullet_scores[left] > pybullet_scores[right]
            if isaac_left_better != pybullet_left_better:
                inverted_pairs.append((left, right))
    return RankingResult(
        isaac_order=isaac_order,
        pybullet_order=pybullet_order,
        inverted=isaac_order != pybullet_order,
        inverted_pairs=tuple(inverted_pairs),
    )
