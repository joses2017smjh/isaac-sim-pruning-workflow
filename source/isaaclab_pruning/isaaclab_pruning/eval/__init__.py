"""Evaluation metrics, 30 cm box check, and sim2sim ranking."""

from .box_check import CAMERA_RECT_DEPTH_M, assert_box_agrees, camera_rect_extent
from .blender_trunk import TRUNK_MEDIAN_LIMIT_M, load_blender_pose_centroids, score_blender_trunk
from .metrics import EpisodeMetrics, episode_metrics, success_vs_cut_error
from .sim2sim import RankingResult, ranking_inversion

__all__ = [
    "CAMERA_RECT_DEPTH_M",
    "TRUNK_MEDIAN_LIMIT_M",
    "EpisodeMetrics",
    "RankingResult",
    "assert_box_agrees",
    "camera_rect_extent",
    "episode_metrics",
    "load_blender_pose_centroids",
    "ranking_inversion",
    "score_blender_trunk",
    "success_vs_cut_error",
]
