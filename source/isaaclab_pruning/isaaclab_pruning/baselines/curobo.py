"""CuRobo UR5e+pruner oracle interface. Sphere geometry is a placeholder."""

from __future__ import annotations

from dataclasses import dataclass

UR5E_JOINT_NAMES: tuple[str, ...] = (
    "ur5e__shoulder_pan_joint",
    "ur5e__shoulder_lift_joint",
    "ur5e__elbow_joint",
    "ur5e__wrist_1_joint",
    "ur5e__wrist_2_joint",
    "ur5e__wrist_3_joint",
)

# Conservative link spheres until the imported URDF is available for a real fit.
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


@dataclass(frozen=True)
class CuroboOracleStatus:
    configured: bool
    reason: str


def ur5e_pruner_oracle_status(*, urdf_usd_path: str | None) -> CuroboOracleStatus:
    if not urdf_usd_path:
        return CuroboOracleStatus(
            configured=False,
            reason="UR5e+pruner USD is not imported yet; placeholder spheres must not be reported as an oracle.",
        )
    return CuroboOracleStatus(configured=True, reason="USD present; generate collision spheres before claiming ~60%.")
