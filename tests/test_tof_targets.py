"""The ToF sensors' mesh list: unchanged by default, with the ground relief only when asked for."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from isaaclab_pruning.sim.tof_targets import TOF_GROUND_ENV, tof_ground_requested, tof_ray_targets

ROOT = Path(__file__).resolve().parents[1]
COLLISION = ["/World/envs/env_0/Tree/branch", "/World/envs/env_0/Tree/spur/Selected", "/World/Orchard/post3"]
SELECTED = "/World/envs/env_0/Tree/spur/Selected"
BODY = object()


def test_the_default_list_is_the_one_every_earlier_recording_used():
    # Before the switch existed the recorder built this list inline.
    earlier = [path for path in COLLISION if path != SELECTED]
    earlier.append(BODY)
    assert tof_ray_targets(COLLISION, SELECTED, BODY) == earlier
    assert not tof_ground_requested({}) and not tof_ground_requested({TOF_GROUND_ENV: "0"})


def test_the_ground_is_added_only_when_asked_and_before_the_tracked_body():
    ground = ["/World/Orchard/ground/ground_mesh"]
    assert tof_ground_requested({TOF_GROUND_ENV: "1"})
    assert tof_ray_targets(COLLISION, SELECTED, BODY, ground) == [COLLISION[0], COLLISION[2], ground[0], BODY]


def test_bad_requests_are_refused():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        tof_ground_requested({TOF_GROUND_ENV: "yes"})
    with pytest.raises(ValueError, match="distinct"):
        tof_ray_targets(COLLISION, SELECTED, BODY, ["/g", "/g"])
    with pytest.raises(ValueError, match="already among"):
        tof_ray_targets(COLLISION, SELECTED, BODY, [COLLISION[0]])


def test_the_recorder_builds_its_tof_targets_through_this_function():
    source = (ROOT / "hpc/inner/render_pruning_workflow.py").read_text()
    tree = ast.parse(source)
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.count("tof_ray_targets") == 1 and calls.count("tof_ground_requested") == 1
    # Both sensors still share one list, and the ground paths are recorded only when the switch is on.
    assert (
        "self.cfg.tof0_cfg.mesh_prim_paths = targets" in source
        and "self.cfg.tof1_cfg.mesh_prim_paths = targets" in source
    )
    assert 'self.blender_scene.evidence["tof_ground_mesh_paths"] = ground_paths' in source
