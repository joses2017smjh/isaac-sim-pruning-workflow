"""Sensor models and fixed hardware rig definitions."""

from .fusion import fuse_depths
from .tof_noise import ToFNoiseConfig, ToFObservation, apply_tof_noise
from .wrist_camera import CANDIDATES, WristCameraCandidate, score_candidate

__all__ = [
    "CANDIDATES",
    "ToFNoiseConfig",
    "ToFObservation",
    "WristCameraCandidate",
    "apply_tof_noise",
    "fuse_depths",
    "score_candidate",
]
