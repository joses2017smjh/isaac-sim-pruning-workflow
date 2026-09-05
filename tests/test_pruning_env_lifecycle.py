from __future__ import annotations

import inspect

from isaaclab_pruning.robot.articulation import make_ur5e_pruner_articulation_cfg
from isaaclab_pruning.sim.pruning_env import make_pruning_env_cls


def test_scene_entity_names_resolve_after_direct_env_initializes_physics_views() -> None:
    """Guard the v60 lifecycle ordering diagnosed by smoke job 21153411."""
    source = inspect.getsource(make_pruning_env_cls)

    base_init = source.index("super().__init__(cfg, render_mode, **kwargs)")
    entity_resolve = source.index("self.robot_entity_cfg.resolve(self.scene)")
    setup_start = source.index("def _setup_scene(self):")
    pre_physics_start = source.index("def _pre_physics_step", setup_start)

    assert base_init < entity_resolve < setup_start
    assert "self.robot_entity_cfg.resolve(self.scene)" not in source[setup_start:pre_physics_start]


def test_robot_spawner_activates_contact_reporting_for_the_contact_sensor() -> None:
    source = inspect.getsource(make_ur5e_pruner_articulation_cfg)

    assert "activate_contact_sensors=True" in source


def test_ik_uses_backend_neutral_link_jacobian_torch_view() -> None:
    """Guard the raw-Warp-array failure diagnosed by smoke job 21153625."""
    source = inspect.getsource(make_pruning_env_cls)

    assert "self.robot.data.body_link_jacobian_w" in source
    assert "root_physx_view.get_jacobians" not in source
