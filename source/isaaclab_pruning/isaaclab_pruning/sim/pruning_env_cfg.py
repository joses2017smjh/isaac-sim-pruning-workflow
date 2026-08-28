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
    robot_cfg = None  # ArticulationCfg from make_ur5e_pruner_articulation_cfg() in __post_init__
    contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
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
        if self.robot_cfg is None:
            self.robot_cfg = make_ur5e_pruner_articulation_cfg()
