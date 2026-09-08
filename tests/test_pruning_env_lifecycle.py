from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


@pytest.mark.parametrize("joint_ids, expected", [(slice(None), 6), ([0, 2, 4], 3)])
def test_smoke_joint_count_supports_resolved_slice(joint_ids, expected) -> None:
    """Execute the actual report assignment without starting Isaac (job 21208115.2)."""
    path = Path(__file__).parents[1] / "hpc/inner/smoke_env.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "report['n_arm_joints']" for target in node.targets)
    )
    env = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 6))),
        robot_entity_cfg=SimpleNamespace(joint_ids=joint_ids),
    )
    namespace = {"env": env, "report": {}, "as_torch": lambda value: value}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["report"]["n_arm_joints"] == expected
