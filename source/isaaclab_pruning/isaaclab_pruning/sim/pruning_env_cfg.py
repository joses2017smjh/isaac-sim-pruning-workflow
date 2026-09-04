"""DirectRLEnv configuration. Importing this module requires Isaac Lab."""

from __future__ import annotations

from isaaclab_pruning.sim.pruning_env import require_isaaclab

require_isaaclab()

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_pruning.policies.observations import (
    ARM_JOINT_COUNT,
    WIDTH_MATCHED_HW,
    ObservationVariant,
    observation_width,
)
from isaaclab_pruning.robot.articulation import make_ur5e_pruner_articulation_cfg
from isaaclab_pruning.robot.ur5e_pruner import load_ur5e_pruner_spec
from isaaclab_pruning.sensors.tof_raycaster import (
    VL53L8CX_MAX_RANGE_M,
    VL53L8CX_MIN_RANGE_M,
    make_vl53l8cx_raycaster_cfg,
)
from isaaclab_pruning.sim.prim_paths import ROBOT_PRIM_EXPR
from isaaclab_pruning.sim.tof_smoke_geometry import TOF_SMOKE_TARGET


@configclass
class PruningEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 60.0
    action_space = 7
    # Overwritten in __post_init__ from observation_width(variant). Leaving a
    # shared 128 is the BHL "cameras mounted, never read" trap.
    observation_space = 0
    state_space = 0
    num_envs = 1
    seed = 0
    observation_variant: str = "B_tof"
    n_joints: int = ARM_JOINT_COUNT
    flow_hw: tuple[int, int] = WIDTH_MATCHED_HW
    tof_hw: tuple[int, int] = WIDTH_MATCHED_HW
    metric_hw: tuple[int, int] = WIDTH_MATCHED_HW
    tof_min_range_m: float = VL53L8CX_MIN_RANGE_M
    tof_max_range_m: float = VL53L8CX_MAX_RANGE_M
    # Hardware-inspired noise is on for training. GPU geometry smokes turn it
    # off so their before/after range assertion is deterministic.
    tof_noise_enabled: bool = True
    # Opt-in only. The normal task always casts against the procedural tree;
    # hpc/inner/smoke_env.py enables this non-colliding deterministic wall.
    tof_smoke_target_enabled: bool = False
    tof_smoke_target_position_w_m: tuple[float, float, float] = TOF_SMOKE_TARGET.position_w_m
    tof_smoke_target_size_m: tuple[float, float, float] = TOF_SMOKE_TARGET.size_m
    robot_cfg = None  # ArticulationCfg from make_ur5e_pruner_articulation_cfg() in __post_init__
    # Warp multi-mesh ray-casters bound to the reviewed fixed ToF site frames.
    # Runtime construction and scene registration live in PruningEnv._setup_scene.
    tof0_cfg = make_vl53l8cx_raycaster_cfg("tof0")
    tof1_cfg = make_vl53l8cx_raycaster_cfg("tof1")
    contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{ROBOT_PRIM_EXPR}/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        render=RenderCfg(antialiasing_mode="Off"),
    )
    scene = InteractiveSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=False,
        replicate_physics=True,
        env_spacing=5.0,
    )

    def __post_init__(self):
        spec = load_ur5e_pruner_spec()
        self.n_joints = len(spec.arm_joints)
        self.action_space = spec.action_dim
        self.observation_space = observation_width(
            ObservationVariant(self.observation_variant),
            n_joints=self.n_joints,
            flow_hw=self.flow_hw,
            tof_hw=self.tof_hw,
            metric_hw=self.metric_hw,
        )
        if self.observation_space == 128:
            raise ValueError("observation_space 128 is the BHL shared-bag trap; wire observation_width.")
        if tuple(self.tof_hw) != (8, 8):
            raise ValueError(f"VL53L8CX ray-caster output is fixed at 8x8; got tof_hw={self.tof_hw}.")
        if not 0.0 < self.tof_min_range_m < self.tof_max_range_m:
            raise ValueError("Expected 0 < tof_min_range_m < tof_max_range_m.")
        if self.robot_cfg is None:
            self.robot_cfg = make_ur5e_pruner_articulation_cfg()
