"""Isaac Lab ArticulationCfg for the imported UR5e + mock-pruner USD."""

from __future__ import annotations

from pathlib import Path

from isaaclab_pruning.sim.prim_paths import ROBOT_PRIM_EXPR
from isaaclab_pruning.sim.pruning_env import require_isaaclab


def make_ur5e_pruner_articulation_cfg(usd_path: str | Path | None = None):
    """Spawn cfg for a reviewed BDS UR5e + mock-pruner asset.

    v1 does not spawn ``linear_slider__joint1``. The real system positions the
    slider before the approach; matching that joint against this USD fails at
    env construct. Actuator gains come from ``ur5e_pruner.yaml``. Stiffness was
    missing on the importer's PhysX drives.
    """
    from isaaclab_pruning.robot import (
        assert_runtime_usd_ready,
        imported_usd_path,
        load_ur5e_pruner_config,
        repository_root,
    )

    cfg = load_ur5e_pruner_config()
    path = Path(usd_path) if usd_path is not None else imported_usd_path(cfg)
    configured_path = repository_root() / str(cfg["usd"]["relative_path"])
    explicit_asset = path.resolve() != configured_path.resolve()
    assert_runtime_usd_ready(cfg, explicit_asset=explicit_asset, usd_path=path)
    require_isaaclab()

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import ArticulationRootPropertiesCfg, UsdFileCfg

    from isaaclab_pruning.robot import load_ur5e_pruner_spec

    spec = load_ur5e_pruner_spec()
    if not path.is_file():
        raise FileNotFoundError(f"Imported UR5e USD is missing: {path}")
    arm = cfg["actuators"]["arm"]
    if any("linear_slider" in expr for expr in arm["joint_names_expr"]):
        raise ValueError("v1 ArticulationCfg must not match linear_slider__joint1.")
    return ArticulationCfg(
        # DirectRLEnv._setup_scene constructs this asset itself, so the
        # InteractiveScene placeholder is never expanded for us.  Isaac Lab 3
        # spawn functions require a globally rooted path.
        prim_path=ROBOT_PRIM_EXPR,
        spawn=UsdFileCfg(
            usd_path=str(path.resolve()),
            # Required by the v60 PhysX ContactSensor: without this the
            # spawned rigid bodies do not receive PhysxContactReportAPI.
            activate_contact_sensors=True,
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={joint.name: 0.0 for joint in spec.arm_joints},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(arm["joint_names_expr"]),
                stiffness=float(arm["stiffness"]),
                damping=float(arm["damping"]),
            ),
        },
    )
