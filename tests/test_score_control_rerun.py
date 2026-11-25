"""tools/score_control_rerun.py on four synthetic batches, with the offline geometry stubbed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
MESHES = ["/World/envs/env_0/Tree/a", "/World/Orchard/post3"]
GROUND = ["/World/Orchard/ground/ground_mesh"]


@pytest.fixture(scope="module")
def scorer():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("score_control_rerun", TOOLS / "score_control_rerun.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(start=71, held=(73, 74, 75, 76, 77), detach=77, tof_stop=None):
    frames = []
    for index in range(200):
        cut = {"phase": "closing" if start is not None and start <= index < start + 6 else "approach"}
        if index == detach:
            cut["detach_event"] = True
        frame = {"index": index, "live_vision": {"cut": cut, "closure_hold": {"held": index in held}}}
        if tof_stop is not None and index >= tof_stop:
            frame["sensor_stop_reason"] = "tof_minimum_clearance"
        frames.append(frame)
    return frames


def _batch(root, scorer, name, *, frames=None, ok=True, ground_paths=None, meshes=MESHES, matches=True):
    strategy = scorer.STRATEGIES[name]
    tree, vertex = (0, 21386) if name in scorer.REPEAT_BATCHES else (1, 14944)
    row = {
        "index": 0,
        "daylight": "source",
        "photometric_normalization": "raw",
        "target_tree_index": tree,
        "component_first_vertex": vertex,
        "strategy": {"name": strategy},
    }
    batch = root / name
    run = batch / scorer.run_label(0, row)
    run.mkdir(parents=True)
    (batch / "plan.json").write_text(
        json.dumps({"protocol": scorer.PROTOCOL, "strategy": {"name": strategy}, "runs": [row]})
    )
    scene = {"target": {"id": scorer.TARGETS[name]}, "collision_mesh_paths": meshes}
    if ground_paths:
        scene["tof_ground_mesh_paths"] = ground_paths
    (run / "report.json").write_text(json.dumps({"blender_scene": scene, "node": "cn-gpu5", "job_id": "1"}))
    (run / "frames.json").write_text(json.dumps({"frames": frames if frames is not None else _frames()}))
    checks = {f"check_{k}": True for k in range(17)}
    (run / "sequence_grade.json").write_text(json.dumps({"checks": checks, "failed_checks": []}))
    (run / "experiment_result.json").write_text(json.dumps({"ok": ok, "configuration_matches": matches}))
    return batch


def _geometry(ground_class="above", trees=(1.0, 1.0), full=(1.0, 1.0)):
    def check(validity, within):
        return {
            "zones": 1000,
            "validity_same": round(1000 * validity),
            "both_valid": 400,
            "within_1mm_as_ray_length": round(400 * within),
        }

    def geometry(run_dir, report, frames, last):
        on = bool((report.get("blender_scene") or {}).get("tof_ground_mesh_paths"))
        # With the switch on the recorded grids hold ground returns: the trees-only cast misses them.
        return {
            "ground": {
                "class": ground_class,
                "anything_beneath_frames": 0,
                "kinematics_ok": True,
                "max_tool_point_error_mm": 0.01,
            },
            "tof_trees_only": check(0.82, 1.0) if on else check(*trees),
            "tof_with_ground": check(*full),
        }

    return geometry


def _all(root, scorer, **overrides):
    batches = []
    for name in (*scorer.REPEAT_BATCHES, scorer.TOF_BATCH):
        kwargs = dict(overrides.get(name, {}))
        if name == scorer.TOF_BATCH:
            kwargs.setdefault("ground_paths", GROUND)
            kwargs.setdefault("frames", _frames(held=()))
        batches.append(_batch(root, scorer, name, **kwargs))
    return batches


REFERENCES = {"tree0_SPUR_component_21386": MESHES, "tree1_SPUR_component_14944": MESHES}


def test_the_hold_pattern_and_the_tof_criterion(scorer):
    assert scorer.hold_structure(_frames())["as_registered"]
    assert not scorer.hold_structure(_frames(held=(73, 74, 75, 76)))["as_registered"]  # stops one frame early
    assert not scorer.hold_structure(_frames(detach=78))["as_registered"]
    assert scorer.hold_structure(_frames(start=None, held=(), detach=None))["closure_start"] is None
    good = {"zones": 1000, "validity_same": 990, "both_valid": 400, "within_1mm_as_ray_length": 396}
    assert scorer.tof_agreement(good) == {
        "zones": 1000,
        "validity_agreement": 0.99,
        "within_1mm": 0.99,
        "reproduced": True,
    }
    assert not scorer.tof_agreement({**good, "validity_same": 820})["reproduced"]


def test_four_clean_runs_support_every_prediction(scorer, tmp_path):
    result = scorer.score(_all(tmp_path, scorer), _geometry(), REFERENCES)
    assert {name: item["verdict"] for name, item in result["verdicts"].items()} == {
        "R0_physical_scenes": "supported",
        "R1_three_repeats_pass": "supported",
        "R2_hold_engages_as_before": "supported",
        "R3_switch_off_changes_nothing": "supported",
        "R4_switch_on_adds_the_ground": "supported",
        "R5_switch_on_changes_no_outcome": "supported",
    }
    assert [run["batch"] for run in result["runs"]] == sorted((*scorer.REPEAT_BATCHES, scorer.TOF_BATCH))


def test_a_failed_repeat_a_changed_hold_and_a_leaked_switch_refute_their_predictions(scorer, tmp_path):
    a, b, c = scorer.REPEAT_BATCHES
    overrides = {a: {"ok": False}, b: {"frames": _frames(held=(73, 74))}, c: {"ground_paths": GROUND}}
    verdicts = scorer.score(_all(tmp_path, scorer, **overrides), _geometry(), REFERENCES)["verdicts"]
    assert verdicts["R1_three_repeats_pass"]["verdict"] == "refuted"
    assert verdicts["R2_hold_engages_as_before"]["departures"] == [
        f"{b}/run_00_source_tree0_v21386_planned_pose_jaw_hold_ctl"
    ]
    assert verdicts["R3_switch_off_changes_nothing"]["verdict"] == "refuted"
    assert verdicts["R4_switch_on_adds_the_ground"]["verdict"] == "supported"


def test_a_run_beneath_the_ground_is_not_scored_and_supports_nothing(scorer, tmp_path):
    verdicts = scorer.score(_all(tmp_path, scorer), _geometry(ground_class="tool"), REFERENCES)["verdicts"]
    assert verdicts["R0_physical_scenes"]["verdict"] == "refuted" and verdicts["R0_physical_scenes"]["scored"] == 0
    assert verdicts["R1_three_repeats_pass"]["verdict"] == "not supported: unscored runs"


def test_the_switch_run_must_show_the_ground_and_change_no_outcome(scorer, tmp_path):
    name = scorer.TOF_BATCH
    silent = scorer.score(_all(tmp_path / "silent", scorer, **{name: {"ground_paths": None}}), _geometry(), REFERENCES)
    assert silent["verdicts"]["R4_switch_on_adds_the_ground"]["verdict"] == "refuted"
    stopped = {name: {"frames": _frames(held=(), tof_stop=40)}}
    verdicts = scorer.score(_all(tmp_path / "stopped", scorer, **stopped), _geometry(), REFERENCES)["verdicts"]
    assert verdicts["R5_switch_on_changes_no_outcome"]["verdict"] == "refuted"
    assert verdicts["R4_switch_on_adds_the_ground"]["verdict"] == "supported"


def test_only_the_four_registered_batches_are_scored_together(scorer, tmp_path):
    batches = _all(tmp_path, scorer)
    with pytest.raises(ValueError, match="scored together"):
        scorer.score(batches[:3], _geometry(), REFERENCES)
    stray = tmp_path / "some-other-batch"
    stray.mkdir()
    (stray / "plan.json").write_text(json.dumps({"protocol": scorer.PROTOCOL, "runs": []}))
    with pytest.raises(ValueError, match="not a batch registered"):
        scorer.score([*batches, stray], _geometry(), REFERENCES)
