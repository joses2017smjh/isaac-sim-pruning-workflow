"""UR5e + mock-pruner articulation constants. Isaac Lab is not required."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PACKAGE = "isaaclab_pruning.config.robot"


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    lower: float
    upper: float
    stiffness: float
    damping: float
    effort_limit: float | None = None


@dataclass(frozen=True)
class Ur5ePrunerSpec:
    eef_body: str
    slider_held_fixed: bool
    action_dim: int
    arm_joints: tuple[JointSpec, ...]
    slider_joint: JointSpec
    ik_method: str
    ik_lambda: float
    mouth_half_extents_m: tuple[float, float, float]
    failure_half_extents_m: tuple[float, float, float]
    failure_offset_m: tuple[float, float, float]
    perpendicularity_tolerance_deg: float
    joint_names_expr: tuple[str, ...]


def load_ur5e_pruner_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        source = resources.files(_CONFIG_PACKAGE).joinpath("ur5e_pruner.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(source)
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_ur5e_pruner_spec(path: str | Path | None = None) -> Ur5ePrunerSpec:
    cfg = load_ur5e_pruner_config(path)
    actuators = cfg["actuators"]
    slider = cfg["joints"]["slider"][0]
    arm = [
        JointSpec(
            name=joint["name"],
            joint_type=joint["type"],
            lower=float(joint["lower_rad"]),
            upper=float(joint["upper_rad"]),
            stiffness=float(actuators["arm"]["stiffness"]),
            damping=float(actuators["arm"]["damping"]),
            effort_limit=float(joint["effort_limit"]),
        )
        for joint in cfg["joints"]["arm"]
    ]
    slider_spec = JointSpec(
        name=slider["name"],
        joint_type=slider["type"],
        lower=float(slider["lower_m"]),
        upper=float(slider["upper_m"]),
        stiffness=float(slider["stiffness"]),
        damping=float(slider["damping"]),
    )
    cutter = cfg["cutter"]
    return Ur5ePrunerSpec(
        eef_body=str(cfg["eef_body"]),
        slider_held_fixed=bool(cfg["slider_held_fixed"]),
        action_dim=int(cfg["action_dim"]),
        arm_joints=tuple(arm),
        slider_joint=slider_spec,
        ik_method=str(cfg["ik"]["method"]),
        ik_lambda=float(cfg["ik"]["lambda_val"]),
        mouth_half_extents_m=tuple(float(v) for v in cutter["mouth_half_extents_m"]),
        failure_half_extents_m=tuple(float(v) for v in cutter["failure_half_extents_m"]),
        failure_offset_m=tuple(float(v) for v in cutter["failure_offset_m"]),
        perpendicularity_tolerance_deg=float(cutter["perpendicularity_tolerance_deg"]),
        joint_names_expr=("ur5e__.*", "linear_slider__joint1"),
    )
