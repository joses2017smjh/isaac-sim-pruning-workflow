"""The controls are fixed definitions: poses, cameras, lighting factors and pairings cannot drift."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def controls(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("generalization_controls", TOOLS / "generalization_controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AXES = dict(forward=[0.0, 1.0, 0.0], right=[1.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])


def test_every_condition_has_a_registered_baseline_and_a_unique_id(controls):
    ids = [c["id"] for c in controls.CONDITIONS]
    assert len(set(ids)) == len(ids) == 10
    for cond in controls.CONDITIONS:
        assert cond["camera_model"] in controls.CAMERA_MODELS
        assert cond["pose_set"] in controls.POSE_SETS
        assert cond["lighting"] in ("source", "evening", "overcast") or cond["lighting"] in controls.LIGHTING_CONTROLS
        if cond["id"] != "source":
            assert cond["id"] in controls.BASELINE_OF
    for variant in controls.PHOTOMETRIC_VARIANTS:
        assert controls.BASELINE_OF[f"evening/{variant}"] == "evening"
    assert controls.BASELINE_OF["source/close/isaac_wrist"] == "source/close/training_rig"
    assert controls.frames_per_tree() == 62


def test_camera_models_reproduce_the_training_and_isaac_intrinsics(controls):
    training = controls.intrinsics("training_rig")
    assert training[0][0] == pytest.approx(398.22, abs=0.01) and training[0][2] == 255.5
    isaac = controls.intrinsics("isaac_wrist")
    assert isaac[0][0] == 320 and isaac[1][1] == 320 and isaac[0][2] == 239.5 and isaac[1][2] == 159.5
    assert controls.horizontal_fov_deg("training_rig") == pytest.approx(65.47, abs=0.01)
    assert controls.horizontal_fov_deg("isaac_wrist") == pytest.approx(73.74, abs=0.01)


def test_far_rig_reproduces_the_matrix_poses(controls):
    poses = controls.poses(controls.POSE_SETS["far"], reference_xy=(-9.8, -7.1), rotation_ref=[1.5, 0, 3.1], **AXES)
    assert [p["id"] for p in poses] == ["rig0_l", "rig0_r", "rig1_l", "rig1_r", "rig2_l", "rig2_r"]
    assert poses[0]["location"] == pytest.approx([-9.92, -7.1, 0.85])
    assert poses[1]["location"] == pytest.approx([-9.68, -7.1, 0.85])
    assert poses[5]["location"][2] == 1.15 and poses[5]["rotation_euler"] == [1.5, 0, 3.1]


def test_close_rigs_sit_at_isaac_distances_in_front_of_the_spur(controls):
    centroid = [2.0, 5.0, 0.85]
    poses = controls.poses(
        controls.POSE_SETS["close"], reference_xy=(0, 0), rotation_ref=[1.5, 0, 3.1], target_centroid=centroid, **AXES
    )
    assert [p["id"] for p in poses] == ["d0_l", "d0_r", "d1_l", "d1_r", "d2_l", "d2_r"]
    # 0.16 m back along the viewing direction, 0.02 m to each side.
    assert poses[0]["location"] == pytest.approx([1.98, 4.84, 0.85])
    assert poses[1]["location"] == pytest.approx([2.02, 4.84, 0.85])
    assert poses[4]["location"] == pytest.approx([1.98, 4.61, 0.85])
    with pytest.raises(ValueError):
        controls.poses(controls.POSE_SETS["close"], reference_xy=(0, 0), rotation_ref=[0, 0, 0], **AXES)


def test_pitched_rig_looks_up_at_the_spur_from_below(controls):
    centroid = [0.0, 0.0, 1.0]
    view = controls.pitched_forward(AXES["forward"], AXES["up"], controls.ISAAC_PITCH_DEG)
    assert view[2] == pytest.approx(math.sin(math.radians(39.7)))
    poses = controls.poses(
        controls.POSE_SETS["close_up40"],
        reference_xy=(0, 0),
        rotation_ref=[1.5, 0, 3.1],
        target_centroid=centroid,
        pitched_rotation=[0.9, 0, 3.1],
        **AXES,
    )
    centre = [(a + b) / 2 for a, b in zip(poses[0]["location"], poses[1]["location"])]
    assert centre[2] < centroid[2]  # camera below the spur
    assert math.dist(centre, centroid) == pytest.approx(0.16)
    assert poses[0]["rotation_euler"] == [0.9, 0, 3.1]
    with pytest.raises(ValueError):
        controls.poses(
            controls.POSE_SETS["close_up40"],
            reference_xy=(0, 0),
            rotation_ref=[0, 0, 0],
            target_centroid=centroid,
            **AXES,
        )


def test_sweep_spans_isaac_range_to_matrix_range(controls):
    spec = controls.POSE_SETS["sweep"]
    poses = controls.poses(spec, reference_xy=(0, 0), rotation_ref=[0, 0, 0], target_centroid=[0, 0, 0], **AXES)
    assert [p["id"] for p in poses] == [f"sweep{i}" for i in range(8)]
    assert [-p["location"][1] for p in poses] == spec["distances_m"]
    assert spec["distances_m"][0] == 0.16 and spec["distances_m"][-1] == 2.3


def test_repo_local_lighting_scales_a_companion_preset_without_writing_it(controls, monkeypatch):
    calls = []
    fake = types.ModuleType("daylight_presets")
    fake.PRESETS = {
        "evening": {"world_color": [0.1, 0.1, 0.2], "world_strength": 0.144, "sun": {"energy": 2.903}},
        "overcast": {"world_color": [0.7, 0.75, 0.82], "world_strength": 0.5, "sun": None},
    }

    def preset(name):
        import copy

        return copy.deepcopy(fake.PRESETS[name])

    def apply_preset(scene, name, seed):
        calls.append((name, preset(name), seed))
        return {"name": name, "seed": seed}

    fake.preset, fake.apply_preset = preset, apply_preset
    monkeypatch.setitem(sys.modules, "daylight_presets", fake)

    config, origin = controls.lighting_config("evening_x2.6")
    assert config["sun"]["energy"] == pytest.approx(2.903 * 2.6) and config["world_strength"] == pytest.approx(0.3744)
    assert origin["factor"] == 2.6 and origin["base_preset"] == "evening"
    config, _ = controls.lighting_config("overcast_div2.6")
    assert config["sun"] is None and config["world_strength"] == pytest.approx(0.5 / 2.6)
    with pytest.raises(ValueError):
        controls.lighting_config("dusk")

    applied = controls.apply_lighting(object(), "evening_x2.6", 7)
    assert applied["control"]["origin"].startswith("repo-local")
    assert calls[-1][1]["sun"]["energy"] == pytest.approx(2.903 * 2.6)
    assert "evening_x2.6" not in fake.PRESETS  # injected for one call only
    applied = controls.apply_lighting(object(), "evening", 7)
    assert applied["control"]["origin"].startswith("companion") and calls[-1][0] == "evening"


def test_dino_scope_and_geometry_sharing_are_explicit(controls):
    assert controls.dino_rig_compatible("training_rig", "far")
    assert not controls.dino_rig_compatible("isaac_wrist", "far")
    assert not controls.dino_rig_compatible("training_rig", "close")
    assert controls.geometry_key("training_rig", "far") == "far__training_rig"
    lighting = [c for c in controls.CONDITIONS if c["group"] == "lighting"]
    assert {controls.geometry_key(c["camera_model"], c["pose_set"]) for c in lighting} == {"far__training_rig"}
