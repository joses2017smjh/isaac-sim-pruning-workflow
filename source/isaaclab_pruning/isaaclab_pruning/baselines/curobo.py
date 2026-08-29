"""CuRobo UR5e+pruner oracle interface.

Link spheres are fitted from licensed pybullet-tree-sim collision STLs.
``configured`` is true only when the imported USD and the sphere JSON both exist.
That is not a runtime success rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab_pruning.geometry.cutter import load_binary_stl
from isaaclab_pruning.robot.ur5e_pruner import imported_usd_path, repository_root

UR5E_JOINT_NAMES: tuple[str, ...] = (
    "ur5e__shoulder_pan_joint",
    "ur5e__shoulder_lift_joint",
    "ur5e__elbow_joint",
    "ur5e__wrist_1_joint",
    "ur5e__wrist_2_joint",
    "ur5e__wrist_3_joint",
)

# Conservative fallback if STLs are not fetched. Do not report as an oracle.
PLACEHOLDER_SPHERES: dict[str, list[tuple[tuple[float, float, float], float]]] = {
    "ur5e__base_link": [((0.0, 0.0, 0.06), 0.08)],
    "ur5e__shoulder_link": [((0.0, 0.0, 0.08), 0.07)],
    "ur5e__upper_arm_link": [((0.0, 0.0, 0.21), 0.07)],
    "ur5e__forearm_link": [((0.0, 0.0, 0.19), 0.06)],
    "ur5e__wrist_1_link": [((0.0, 0.0, 0.05), 0.05)],
    "ur5e__wrist_2_link": [((0.0, 0.0, 0.05), 0.05)],
    "ur5e__wrist_3_link": [((0.0, 0.0, 0.04), 0.045)],
    "mock_pruner__tool0": [((0.0, 0.0, 0.08), 0.06), ((0.0, 0.0, 0.14), 0.04)],
}

_LINK_MESHES: tuple[tuple[str, str], ...] = (
    ("ur5e__base_link", "base.stl"),
    ("ur5e__shoulder_link", "shoulder.stl"),
    ("ur5e__upper_arm_link", "upperarm.stl"),
    ("ur5e__forearm_link", "forearm.stl"),
    ("ur5e__wrist_1_link", "wrist1.stl"),
    ("ur5e__wrist_2_link", "wrist2.stl"),
    ("ur5e__wrist_3_link", "wrist3.stl"),
    ("mock_pruner__tool0", "new_cutter_decimated.stl"),
)


@dataclass(frozen=True)
class CuroboOracleStatus:
    configured: bool
    reason: str


def collision_mesh_dir() -> Path:
    return (
        repository_root()
        / "third_party"
        / "src"
        / "pybullet_tree_sim"
        / "pybullet_tree_sim"
        / "urdf"
        / "ur5e"
        / "collision"
    )


def bounding_sphere(vertices: np.ndarray) -> tuple[tuple[float, float, float], float]:
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    return (float(center[0]), float(center[1]), float(center[2])), radius


def fit_spheres_from_stls(
    mesh_dir: str | Path | None = None,
) -> dict[str, list[tuple[tuple[float, float, float], float]]]:
    root = Path(mesh_dir) if mesh_dir is not None else collision_mesh_dir()
    spheres: dict[str, list[tuple[tuple[float, float, float], float]]] = {}
    for link, filename in _LINK_MESHES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Collision mesh missing for {link}: {path}")
        center, radius = bounding_sphere(load_binary_stl(path))
        spheres[link] = [(center, radius)]
    return spheres


def default_spheres_path() -> Path:
    return repository_root() / "docs" / "evidence" / "curobo_spheres.json"


def ur5e_pruner_oracle_status(
    *,
    urdf_usd_path: str | None,
    spheres_path: str | Path | None = None,
) -> CuroboOracleStatus:
    if not urdf_usd_path:
        return CuroboOracleStatus(
            configured=False,
            reason="UR5e+pruner USD is not imported yet; placeholder spheres must not be reported as an oracle.",
        )
    usd = Path(urdf_usd_path)
    if not usd.is_file():
        return CuroboOracleStatus(configured=False, reason=f"Imported USD missing: {usd}")
    path = Path(spheres_path) if spheres_path is not None else default_spheres_path()
    if not path.is_file():
        return CuroboOracleStatus(
            configured=False,
            reason="Collision spheres are not fitted yet; run tools/fit_curobo_spheres.py.",
        )
    return CuroboOracleStatus(
        configured=True,
        reason=(
            "USD present and link spheres fitted from pybullet-tree-sim collision STLs "
            "(BSD-3-Clause 4d9f838). CuRobo runtime still needs an Isaac job log."
        ),
    )


def spheres_payload(
    spheres: dict[str, list[tuple[tuple[float, float, float], float]]],
) -> dict[str, Any]:
    usd = imported_usd_path()
    return {
        "source": "pybullet_tree_sim/urdf/ur5e/collision",
        "license": "BSD-3-Clause",
        "revision": "4d9f8384da9ddd3329175cc8ce1f2c7df9720387",
        "usd": str(usd) if usd.is_file() else None,
        "spheres": {
            link: [{"center_m": list(center), "radius_m": radius} for center, radius in values]
            for link, values in spheres.items()
        },
    }
