"""Deterministic ray target used only by the live-ToF GPU smoke.

The production environment casts against the procedural tree.  A smoke test,
however, must distinguish sensor plumbing failures from a particular tree
having no wood in either 8x8 frustum at the robot's zero-joint pose.  This
module describes a broad, non-colliding wall that the smoke opts into
explicitly; normal environment configurations never enable or target it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from isaaclab_pruning.sim.prim_paths import TOF_SMOKE_TARGET_PRIM_EXPR


@dataclass(frozen=True)
class ToFSmokeTargetSpec:
    """Authored properties of the smoke-only cuboid."""

    prim_path: str
    prim_expr: str
    position_w_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    shape: str = "CuboidCfg"
    collision_enabled: bool = False
    rigid_body_enabled: bool = False

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe record of exactly what the smoke authors."""
        return asdict(self)


# From the reviewed content-addressed URDF at its configured all-zero arm pose,
# tof0/tof1 are near (0.771/0.864, 0.422, 0.064) m and look along world +Y.
# At the wall's front face (y=0.94 m), all 64 pixel-centre rays from both 65
# degree diagonal frusta have at least 0.14 m lateral margin.  The 0.52--0.60 m
# ranges are well inside the VL53L8CX 0.03--3.4 m contract.
TOF_SMOKE_TARGET = ToFSmokeTargetSpec(
    prim_path="/World/envs/env_0/ToFSmokeTarget",
    prim_expr=TOF_SMOKE_TARGET_PRIM_EXPR,
    position_w_m=(0.81741087, 0.95, 0.06425),
    size_m=(0.8, 0.02, 0.7),
)


def enable_tof_smoke_target(cfg: Any) -> None:
    """Opt an environment config into the deterministic smoke wall.

    This intentionally uses duck typing so importing the helper remains
    Isaac-free.  The live smoke passes a :class:`PruningEnvCfg`; CPU tests can
    verify the mutation contract with a tiny stand-in.
    """
    scene = getattr(cfg, "scene", None)
    if scene is None or int(getattr(scene, "num_envs", 0)) != 1:
        raise ValueError("The deterministic ToF smoke target is validated only for one environment.")
    for name in ("tof0_cfg", "tof1_cfg"):
        sensor_cfg = getattr(cfg, name, None)
        if sensor_cfg is None:
            raise ValueError(f"Cannot enable the ToF smoke target without cfg.{name}.")
        sensor_cfg.mesh_prim_paths = [TOF_SMOKE_TARGET.prim_expr]
    cfg.tof_smoke_target_enabled = True


__all__ = ["TOF_SMOKE_TARGET", "ToFSmokeTargetSpec", "enable_tof_smoke_target"]
