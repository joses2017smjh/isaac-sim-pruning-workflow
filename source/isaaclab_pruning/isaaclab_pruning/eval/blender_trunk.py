"""Match one Blender orchard pose against L-Py cylinders (trunk median, metres).

Per-frame ``cylinders_world`` annotations are centroids only and are never
loaded as cylinder sidecars. This module uses that list as a pose check:
apply the orchard tilt, recover the rigid translation from the first
centroid, and measure trunk centroid error. Gate: median < 2 mm.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab_pruning.geometry.cylinders import Cylinder, transform_cylinders
from isaaclab_pruning.usd.cylinders import DEFAULT_TREE_TILT_X_DEG, _local_transform

TRUNK_MEDIAN_LIMIT_M = 0.002


def load_blender_pose_centroids(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the centroid list from one Blender shot JSON. Not a cylinder sidecar."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    world = payload.get("cylinders_world")
    if not isinstance(world, list) or not world:
        raise ValueError(f"{source} has no cylinders_world list.")
    centroids = np.asarray(world, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 3 or not np.all(np.isfinite(centroids)):
        raise ValueError(f"{source} cylinders_world must be finite (N, 3) centroids.")
    return centroids, payload


def score_blender_trunk(
    cylinders: Iterable[Cylinder],
    blender_centroids: np.ndarray,
    *,
    tilt_x_deg: float = DEFAULT_TREE_TILT_X_DEG,
    limit_m: float = TRUNK_MEDIAN_LIMIT_M,
) -> dict[str, Any]:
    """Rigidly align tilted L-Py cylinders to one Blender pose and score trunks."""
    records = list(cylinders)
    world = np.asarray(blender_centroids, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("blender_centroids must have shape (N, 3).")
    if len(records) != world.shape[0]:
        raise ValueError(
            f"Cylinder count {len(records)} does not match Blender centroids {world.shape[0]}."
        )
    tilted = transform_cylinders(records, _local_transform(tilt_x_deg, (0.0, 0.0, 0.0)))
    translation = world[0] - tilted[0].centroid
    errors = np.array(
        [float(np.linalg.norm(cylinder.centroid + translation - point)) for cylinder, point in zip(tilted, world)],
        dtype=np.float64,
    )
    trunk_mask = np.array([cylinder.organ_class == "trunk" for cylinder in tilted], dtype=bool)
    if not np.any(trunk_mask):
        raise ValueError("No trunk cylinders to score.")
    trunk_errors = errors[trunk_mask]
    median_m = float(np.median(trunk_errors))
    return {
        "n_cylinders": len(tilted),
        "n_trunk": int(trunk_mask.sum()),
        "tilt_x_deg": float(tilt_x_deg),
        "translation_m": translation.tolist(),
        "median_trunk_error_m": median_m,
        "median_trunk_error_mm": median_m * 1000.0,
        "max_trunk_error_m": float(np.max(trunk_errors)),
        "p95_trunk_error_m": float(np.percentile(trunk_errors, 95)),
        "median_all_error_m": float(np.median(errors)),
        "limit_m": float(limit_m),
        "pass": median_m < float(limit_m),
    }
