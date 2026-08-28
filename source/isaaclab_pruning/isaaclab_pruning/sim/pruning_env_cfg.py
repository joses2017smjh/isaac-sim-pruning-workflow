"""DirectRLEnv configuration. Importing this module requires Isaac Lab."""

from __future__ import annotations

from isaaclab_pruning.sim.pruning_env import require_isaaclab

require_isaaclab()

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class PruningEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 60.0
    action_space = 7
    observation_space = 128
    state_space = 0
    num_envs = 1
    observation_variant: str = "B_tof"

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
    robot_cfg = None
