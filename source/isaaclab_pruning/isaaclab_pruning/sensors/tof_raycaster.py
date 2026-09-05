"""Isaac Lab 3 ray-caster configuration for the two VL53L8CX sites.

The geometry and camera-model calculations in this module are Isaac-free.  The
factory imports Isaac Lab lazily so CPU-only code can inspect and test the
hardware contract without starting Kit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isaaclab_pruning.sim.prim_paths import ROBOT_PRIM_EXPR, TREE_PRIM_EXPR

VL53L8CX_IMAGE_HEIGHT = 8
VL53L8CX_IMAGE_WIDTH = 8
VL53L8CX_DIAGONAL_FOV_DEG = 65.0
VL53L8CX_MIN_RANGE_M = 0.03
VL53L8CX_MAX_RANGE_M = 3.4
VL53L8CX_UPDATE_RATE_HZ = 15.0
VL53L8CX_UPDATE_PERIOD_S = 1.0 / VL53L8CX_UPDATE_RATE_HZ
VL53L8CX_DATA_TYPE = "distance_to_camera"
VL53L8CX_DEPTH_CLIPPING_BEHAVIOR = "none"
VL53L8CX_OFFSET_CONVENTION = "ros"

# The content-addressed v1 URDF is imported with merge_fixed_joints=False.  The
# two site Xforms remain on the stage and the importer verifies their poses.
#
# Isaac Lab 3.0.0b2's PhysX ray-camera backend resolves a non-rigid site to its
# rigid ancestor, but BaseRayCasterCamera then overwrites that resolved
# site-to-body transform with ``cfg.offset``.  Binding directly to a site would
# therefore cast from ``mock_pruner__base`` despite the correct-looking prim
# path.  Track the rigid base explicitly and supply the already provenance-
# checked site offset until the beta backend preserves site transforms.
MOCK_PRUNER_BASE_PRIM_EXPR = (
    f"{ROBOT_PRIM_EXPR}/Geometry/world/ur5e__base_link/"
    "ur5e__base_link_inertia/ur5e__shoulder_link/ur5e__upper_arm_link/"
    "ur5e__forearm_link/ur5e__wrist_1_link/ur5e__wrist_2_link/"
    "ur5e__wrist_3_link/ur5e__flange/ur5e__tool0/"
    "rotation_correction_plate__base/rotation_correction_plate__body/"
    "rotation_correction_plate__tool0/dovetail_male_mount__base/"
    "dovetail_male_mount__body/dovetail_male_mount__tool0/"
    "dovetail_female_mount__base/dovetail_female_mount__body/"
    "dovetail_female_mount__tool0/mock_pruner__base"
)
TOF_SITE_PRIM_EXPRS = {
    "tof0": f"{MOCK_PRUNER_BASE_PRIM_EXPR}/mock_pruner__tof0",
    "tof1": f"{MOCK_PRUNER_BASE_PRIM_EXPR}/mock_pruner__tof1",
}
TOF_SITE_OFFSETS_M = {
    "tof0": (0.04685226669, 0.0, 0.14444246761),
    "tof1": (-0.04685226669, 0.0, 0.14444246761),
}


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Square-pixel pinhole model derived from a diagonal field of view."""

    width: int
    height: int
    diagonal_fov_deg: float
    horizontal_fov_deg: float
    vertical_fov_deg: float
    focal_length_px: float
    principal_point_px: tuple[float, float]

    @property
    def matrix_row_major(self) -> tuple[float, ...]:
        """Return ``[fx, 0, cx, 0, fy, cy, 0, 0, 1]``."""
        cx, cy = self.principal_point_px
        return (
            self.focal_length_px,
            0.0,
            cx,
            0.0,
            self.focal_length_px,
            cy,
            0.0,
            0.0,
            1.0,
        )


def pinhole_intrinsics_from_diagonal_fov(
    *,
    width: int,
    height: int,
    diagonal_fov_deg: float,
) -> PinholeIntrinsics:
    """Convert diagonal FOV into a centered, square-pixel pinhole model."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer.")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer.")
    if not math.isfinite(diagonal_fov_deg) or not 0.0 < diagonal_fov_deg < 180.0:
        raise ValueError("diagonal_fov_deg must be finite and between 0 and 180 degrees.")

    diagonal_fov_rad = math.radians(diagonal_fov_deg)
    focal_length_px = math.hypot(width, height) / (2.0 * math.tan(diagonal_fov_rad / 2.0))
    horizontal_fov_deg = math.degrees(2.0 * math.atan(width / (2.0 * focal_length_px)))
    vertical_fov_deg = math.degrees(2.0 * math.atan(height / (2.0 * focal_length_px)))
    return PinholeIntrinsics(
        width=width,
        height=height,
        diagonal_fov_deg=float(diagonal_fov_deg),
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        focal_length_px=focal_length_px,
        principal_point_px=(width / 2.0, height / 2.0),
    )


VL53L8CX_INTRINSICS = pinhole_intrinsics_from_diagonal_fov(
    width=VL53L8CX_IMAGE_WIDTH,
    height=VL53L8CX_IMAGE_HEIGHT,
    diagonal_fov_deg=VL53L8CX_DIAGONAL_FOV_DEG,
)


def make_vl53l8cx_raycaster_cfg(
    sensor_name: str,
    *,
    mesh_prim_paths: Sequence[Any] = (TREE_PRIM_EXPR,),
):
    """Build the pinned-v60 ``MultiMeshRayCasterCameraCfg`` for one ToF site.

    The generated URDF owns the fixed site pose and the importer verifies it.
    Because of the pinned-v60 site-offset issue described above, ``spawn=None``
    tracks the rigid mock-pruner base and ``offset.pos`` mirrors the verified
    site translation. Isaac Lab's camera offset quaternion is ``xyzw`` (unlike
    the rig YAML's ``wxyz``), hence identity is ``(0, 0, 0, 1)``.
    ``convention='ros'`` makes the optical axis +Z in the site/base frame.

    ``depth_clipping_behavior='none'`` preserves misses as infinity.  The
    existing :func:`isaaclab_pruning.sensors.tof_noise.apply_tof_noise` stage
    must enforce both the 0.03 m minimum and 3.4 m maximum validity limits.
    """
    if sensor_name not in TOF_SITE_PRIM_EXPRS:
        choices = ", ".join(sorted(TOF_SITE_PRIM_EXPRS))
        raise ValueError(f"Unknown ToF sensor {sensor_name!r}; expected one of: {choices}.")
    if not mesh_prim_paths:
        raise ValueError("mesh_prim_paths must contain at least one ray-cast target.")

    try:
        from isaaclab.sensors import MultiMeshRayCasterCameraCfg
        from isaaclab.sensors.ray_caster import patterns
    except ImportError as error:  # pragma: no cover - exercised only outside the v60 runtime
        raise RuntimeError("VL53L8CX ray-caster configs require the pinned Isaac Lab 3 v60 runtime.") from error

    pattern_cfg = patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
        intrinsic_matrix=list(VL53L8CX_INTRINSICS.matrix_row_major),
        width=VL53L8CX_IMAGE_WIDTH,
        height=VL53L8CX_IMAGE_HEIGHT,
    )
    return MultiMeshRayCasterCameraCfg(
        prim_path=MOCK_PRUNER_BASE_PRIM_EXPR,
        mesh_prim_paths=list(mesh_prim_paths),
        spawn=None,
        update_period=VL53L8CX_UPDATE_PERIOD_S,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=TOF_SITE_OFFSETS_M[sensor_name],
            rot=(0.0, 0.0, 0.0, 1.0),
            convention=VL53L8CX_OFFSET_CONVENTION,
        ),
        debug_vis=False,
        pattern_cfg=pattern_cfg,
        max_distance=VL53L8CX_MAX_RANGE_M,
        data_types=[VL53L8CX_DATA_TYPE],
        depth_clipping_behavior=VL53L8CX_DEPTH_CLIPPING_BEHAVIOR,
    )


__all__ = [
    "MOCK_PRUNER_BASE_PRIM_EXPR",
    "PinholeIntrinsics",
    "TOF_SITE_OFFSETS_M",
    "TOF_SITE_PRIM_EXPRS",
    "VL53L8CX_DATA_TYPE",
    "VL53L8CX_DEPTH_CLIPPING_BEHAVIOR",
    "VL53L8CX_DIAGONAL_FOV_DEG",
    "VL53L8CX_IMAGE_HEIGHT",
    "VL53L8CX_IMAGE_WIDTH",
    "VL53L8CX_INTRINSICS",
    "VL53L8CX_MAX_RANGE_M",
    "VL53L8CX_MIN_RANGE_M",
    "VL53L8CX_OFFSET_CONVENTION",
    "VL53L8CX_UPDATE_PERIOD_S",
    "VL53L8CX_UPDATE_RATE_HZ",
    "make_vl53l8cx_raycaster_cfg",
    "pinhole_intrinsics_from_diagonal_fov",
]
