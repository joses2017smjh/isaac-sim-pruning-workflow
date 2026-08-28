"""Pruning task predicates independent of the simulator environment."""

from .curriculum import STAGES, curriculum_stage, select_curriculum_cut
from .loop import EpisodeTarget, episode_start_target
from .reward import RewardWeights, dense_pruning_reward
from .success import CutSuccess, OrientedBox, evaluate_cut_success, segment_intersects_obb

__all__ = [
    "CutSuccess",
    "EpisodeTarget",
    "OrientedBox",
    "RewardWeights",
    "STAGES",
    "curriculum_stage",
    "dense_pruning_reward",
    "episode_start_target",
    "evaluate_cut_success",
    "segment_intersects_obb",
    "select_curriculum_cut",
]
