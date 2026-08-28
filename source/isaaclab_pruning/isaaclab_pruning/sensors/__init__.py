"""Sensor models and fixed hardware rig definitions."""

from .fusion import fuse_depths
from .tof_noise import ToFNoiseConfig, ToFObservation, apply_tof_noise
from .wrist_camera import (
    CANDIDATES,
    RANGE_BAND_M,
    WristCameraCandidate,
    ray_finite_cylinder_t,
    score_candidate,
    score_candidate_on_tree,
    select_wrist_camera,
)

__all__ = [
    "CANDIDATES",
    "RANGE_BAND_M",
    "ToFNoiseConfig",
    "ToFObservation",
    "WristCameraCandidate",
    "apply_tof_noise",
    "fuse_depths",
    "ray_finite_cylinder_t",
    "score_candidate",
    "score_candidate_on_tree",
    "select_wrist_camera",
]
