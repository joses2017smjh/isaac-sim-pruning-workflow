"""Protect experiment isolation, failure accounting and scheduler scope."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scripts(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    modules = []
    for name in ("queue_vision_robustness", "run_vision_experiment"):
        spec = importlib.util.spec_from_file_location(name, tools / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def test_submission_strips_inherited_allocation_and_experiment_overrides(scripts):
    queue, _ = scripts
    env = queue.submission_environment(
        {
            "SLURM_JOB_ID": "old",
            "SLURMD_NODENAME": "old-node",
            "SBATCH_ARRAY_INX": "0-90",
            "SBATCH_DEPENDENCY": "afterok:old",
            "PRUNING_RENDER_FRAMES": "30",
            "PATH": "/bin",
            "USER": "tester",
        }
    )
    assert env == {"PATH": "/bin", "USER": "tester"}


def _frozen_fixture(tmp_path, queue):
    batch = tmp_path / "batch"
    (batch / "code").mkdir(parents=True)
    (batch / "code/runner.py").write_text("original")
    plan = queue.experiment_plan()
    plan.update(
        code_revision="test-revision",
        source_sha256={"runner.py": hashlib.sha256(b"original").hexdigest()},
        external_assets_sha256={str(batch / "code/runner.py"): hashlib.sha256(b"original").hexdigest()},
    )
    (batch / "plan.json").write_text(json.dumps(plan))
    return batch, plan


@pytest.mark.parametrize("capture_code,grade_ok", [(0, True), (0, False), (1, True)])
def test_task_grade_is_independent_and_failures_are_preserved(tmp_path, monkeypatch, scripts, capture_code, grade_ok):
    queue, runner = scripts
    batch, plan = _frozen_fixture(tmp_path, queue)

    def capture(command, *, env, check):
        assert command == ["bash", str(batch / "code/hpc/inner/render_pruning_workflow.sh")]
        assert env["PRUNING_RENDER_FRAMES"] == "200"
        assert env["PRUNING_DAYLIGHT"] == "morning"
        assert env["PRUNING_PHOTOMETRIC_NORMALIZATION"] == "clahe"
        assert "PRUNING_SMOKE_ONLY" not in env
        output = Path(env["PRUNING_RENDER_DIR"])
        (output / "report.json").write_text(
            json.dumps(
                {
                    "frame_count": 200,
                    "photometric_normalization": "clahe",
                    "blender_scene": {"daylight": {"preset": "morning"}},
                }
            )
        )
        (output / "frames.json").write_text('{"frames": []}')
        return SimpleNamespace(returncode=capture_code)

    monkeypatch.setenv("PRUNING_SMOKE_ONLY", "1")
    monkeypatch.setattr(runner.subprocess, "run", capture)
    monkeypatch.setattr(runner, "grade_sequence", lambda *_: {"ok": grade_ok})
    assert runner.execute(batch, 3) == (0 if capture_code == 0 and grade_ok else 1)
    output = batch / "run_03_morning_clahe"
    result = json.loads((output / "experiment_result.json").read_text())
    assert result["sequence_ok"] == grade_ok
    assert result["capture_exit_code"] == capture_code
    assert json.loads((output / "sequence_grade.json").read_text())["ok"] == grade_ok
    with pytest.raises(FileExistsError):
        runner.execute(batch, 3)


def test_changed_frozen_source_fails_before_launch(tmp_path, monkeypatch, scripts):
    queue, runner = scripts
    batch, _ = _frozen_fixture(tmp_path, queue)
    (batch / "code/runner.py").write_text("changed")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kw: pytest.fail("Must not launch changed code"))
    assert runner.execute(batch, 0) == 1
    result = json.loads((batch / "run_00_source_raw/experiment_result.json").read_text())
    assert "Frozen source changed" in result["traceback"]


def test_freeze_excludes_uncommitted_work_and_survives_worktree_edits(tmp_path, scripts):
    queue, _ = scripts
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / "source.py").write_text("committed")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    (root / "source.py").write_text("uncommitted edit")
    (root / "private.txt").write_text("not in the snapshot")
    snapshot = tmp_path / "snapshot"
    hashes = queue.freeze_source(root, snapshot, "HEAD")
    assert (snapshot / "source.py").read_text() == "committed"
    assert not (snapshot / "private.txt").exists()
    assert hashes == {"source.py": hashlib.sha256(b"committed").hexdigest()}


def test_queue_refuses_duplicate_batch_before_calling_sbatch(tmp_path, monkeypatch, scripts):
    queue, _ = scripts
    (tmp_path / "artifacts/vision_robustness/reused").mkdir(parents=True)
    monkeypatch.setattr(queue, "_git", lambda _root, *args: b"" if args[0] == "diff" else b"abc123")
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *_args, **_kw: b"old-job|name|PENDING|Priority\n")
    monkeypatch.setattr(queue.subprocess, "run", lambda *_args, **_kw: pytest.fail("Must not submit duplicates"))
    with pytest.raises(FileExistsError):
        queue.queue_batch(tmp_path, "reused")


def test_plan_without_targets_is_unchanged_six_run_lighting_pilot(scripts):
    queue, _ = scripts
    plan = queue.experiment_plan()
    assert len(plan["runs"]) == 6
    assert plan["maximum_gpu_minutes"] == 150
    assert "target_tree_index" not in plan["runs"][0]
    assert sorted({run["photometric_normalization"] for run in plan["runs"]}) == ["clahe", "raw"]


def test_target_sweep_plan_registers_every_target_and_scales_the_budget(scripts):
    queue, _ = scripts
    targets = [
        {"target_tree_index": 0, "component_first_vertex": 530},
        {"target_tree_index": 1, "component_first_vertex": 1012},
    ]
    plan = queue.experiment_plan(targets, daylight="source", photometric_normalization="raw")
    assert len(plan["runs"]) == len(targets)
    assert plan["maximum_gpu_minutes"] == queue.MINUTES_PER_TRIAL * len(targets)
    assert plan["max_concurrent_gpus"] == 1
    assert [run["component_first_vertex"] for run in plan["runs"]] == [530, 1012]
    assert {run["daylight"] for run in plan["runs"]} == {"source"}
    # A dropped target would silently shrink the denominator.
    assert [run["index"] for run in plan["runs"]] == list(range(len(targets)))


def test_repeated_target_is_refused_before_any_submission(scripts):
    queue, _ = scripts
    repeated = [
        {"target_tree_index": 0, "component_first_vertex": 530},
        {"target_tree_index": 0, "component_first_vertex": 530},
    ]
    with pytest.raises(ValueError, match="unique"):
        queue.experiment_plan(repeated)


def test_target_register_must_agree_with_its_own_count(tmp_path, scripts):
    queue, _ = scripts
    register = tmp_path / "targets.json"
    register.write_text(
        json.dumps({"target_count": 3, "targets": [{"target_tree_index": 0, "component_first_vertex": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagrees with its own target_count"):
        queue.load_targets(register)


def test_target_sweep_forwards_the_spur_to_the_renderer(tmp_path, scripts):
    _, run = scripts
    plan = {"render_samples": 64, "overview_width": 1280, "frames": 200}
    row = {
        "index": 0,
        "daylight": "source",
        "photometric_normalization": "raw",
        "target_tree_index": 1,
        "component_first_vertex": 37796,
    }
    env = run.run_environment(plan, row, tmp_path, tmp_path / "out", {})
    assert env["PRUNING_TARGET_TREE"] == "1"
    assert env["PRUNING_COMPONENT_VERTEX"] == "37796"
    # A lighting row must not inject a target and change the renderer's default.
    lighting = run.run_environment(
        plan, {"daylight": "source", "photometric_normalization": "raw"}, tmp_path, tmp_path, {}
    )
    assert "PRUNING_COMPONENT_VERTEX" not in lighting
    assert "PRUNING_TARGET_TREE" not in lighting


def test_run_label_identifies_the_target_it_recorded(scripts):
    _, run = scripts
    sweep = {
        "daylight": "morning",
        "photometric_normalization": "raw",
        "target_tree_index": 1,
        "component_first_vertex": 42,
    }
    assert run.run_label(7, sweep) == "run_07_morning_tree1_v42"
    lighting = {"daylight": "evening", "photometric_normalization": "clahe"}
    assert run.run_label(5, lighting) == "run_05_evening_clahe"


def test_capture_of_the_wrong_spur_fails_configuration_match(scripts):
    _, run = scripts
    plan = {"frames": 200}
    row = {
        "daylight": "source",
        "photometric_normalization": "raw",
        "target_tree_index": 0,
        "component_first_vertex": 530,
    }

    def report(target_id):
        return {
            "photometric_normalization": "raw",
            "frame_count": 200,
            "blender_scene": {"daylight": {"preset": "source"}, "target": {"id": target_id}},
        }

    assert run.configuration_matches(report("tree0_SPUR_component_530"), row, plan) is True
    assert run.configuration_matches(report("tree0_SPUR_component_8235"), row, plan) is False
    assert run.configuration_matches(report("tree1_SPUR_component_530"), row, plan) is False


def test_dependency_is_optional_and_only_constrains_the_new_job(scripts):
    queue, _ = scripts
    assert queue.dependency_option(None) == []
    assert queue.dependency_option("21360571") == ["--dependency", "afterany:21360571"]
    # afterany, not afterok: a failed earlier batch must still release this one.
    assert "afterok" not in " ".join(queue.dependency_option(123))


def test_dependency_refuses_anything_that_is_not_a_job_id(scripts):
    queue, _ = scripts
    for bad in ("afterok:1", "1;scancel 2", "", "abc", "-1"):
        with pytest.raises(ValueError, match="numeric Slurm job id"):
            queue.dependency_option(bad)


def _manifest(listed_tree1):
    return {
        "branch_geometry_candidates": {
            "tree1_SPUR": {"candidates": [{"component_first_vertex": v} for v in listed_tree1]}
        }
    }


def test_tree1_components_resolve_and_only_constructive_refusals_remain(scripts):
    queue, _ = scripts
    targets = [
        {"target_tree_index": 0, "component_first_vertex": 530, "max_radius_m": 0.005},
        {"target_tree_index": 1, "component_first_vertex": 15004, "max_radius_m": 0.0067},
        # An unlisted tree1 component: refused before September 26, resolved at run time since.
        {"target_tree_index": 1, "component_first_vertex": 1012, "max_radius_m": 0.006},
        {"target_tree_index": 2, "component_first_vertex": 7, "max_radius_m": 0.006},
        {"target_tree_index": 0, "component_first_vertex": 9, "max_radius_m": 0.02},
    ]
    refused = queue.unpresentable_targets(targets, _manifest([15004, 14944]))
    assert [item["component_first_vertex"] for item in refused] == [7, 9]


def test_register_with_unpresentable_targets_is_refused(tmp_path, scripts):
    queue, _ = scripts
    register = tmp_path / "targets.json"
    register.write_text(
        json.dumps(
            {
                "target_count": 1,
                "targets": [{"target_tree_index": 0, "component_first_vertex": 12, "max_radius_m": 0.03}],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest([15004])), encoding="utf-8")
    with pytest.raises(ValueError, match="would be refused by the renderer before recording"):
        queue.load_targets(register, manifest)
    # The Sept 23 register's ten unlisted tree1 targets now load.
    targets, _ = queue.load_targets(Path(__file__).resolve().parents[1] / "docs/evidence/eval_targets_2026-09-23.json")
    assert sum(1 for item in targets if item["target_tree_index"] == 1) == 10


def test_strategy_rows_forward_the_approach_and_are_checked_against_the_capture(tmp_path, scripts):
    launcher, run = scripts
    targets = [
        {"target_tree_index": 0, "component_first_vertex": 530},
        {"target_tree_index": 0, "component_first_vertex": 7524},
    ]
    plan = launcher.experiment_plan(targets, "source", "raw", "tool_axis_standoff")
    assert plan["frames"] == 200 and plan["strategy"]["standoff_m"] == 0.06
    assert plan["maximum_gpu_minutes"] == launcher.MINUTES_PER_TRIAL * 2
    assert plan["protocol"].endswith("STRATEGIES_2026-09-23.md")
    row = plan["runs"][0]
    assert row["strategy"] == {
        "name": "tool_axis_standoff",
        "mode": "tool_axis_standoff",
        "standoff_m": 0.06,
        "max_step_m": 0.004,
    }
    env = run.run_environment(plan, row, tmp_path, tmp_path / "out", {})
    assert env["PRUNING_APPROACH_MODE"] == "tool_axis_standoff"
    assert env["PRUNING_STANDOFF_M"] == "0.06" and env["PRUNING_MAX_STEP_M"] == "0.004"
    assert run.run_label(3, row) == "run_03_source_tree0_v530_tool_axis_standoff"
    # The fine step doubles the episode and the reserved minutes with it.
    fine = launcher.experiment_plan(targets, "source", "raw", "fine_step")
    assert fine["frames"] == 400 and fine["maximum_gpu_minutes"] == launcher.MINUTES_PER_TRIAL * 4
    # A plan without a strategy row forwards nothing new.
    baseline = launcher.experiment_plan(targets, "source", "raw")
    assert "strategy" not in baseline["runs"][0]
    assert "PRUNING_APPROACH_MODE" not in run.run_environment(baseline, baseline["runs"][0], tmp_path, tmp_path, {})

    def report(strategy):
        return {
            "photometric_normalization": "raw",
            "frame_count": 200,
            "approach_strategy": strategy,
            "blender_scene": {"daylight": {"preset": "source"}, "target": {"id": "tree0_SPUR_component_530"}},
        }

    assert run.configuration_matches(
        report({"mode": "tool_axis_standoff", "standoff_m": 0.06, "max_step_m": 0.004}), row, plan
    )
    assert not run.configuration_matches(
        report({"mode": "straight", "standoff_m": 0.0, "max_step_m": 0.004}), row, plan
    )
    assert not run.configuration_matches(report(None), row, plan)
    with pytest.raises(ValueError):
        launcher.experiment_plan(None, "source", "raw", "fine_step")
    with pytest.raises(ValueError):
        launcher.experiment_plan(targets, "source", "raw", "teleport")


def test_trial_time_limit_scales_with_the_episode(scripts):
    launcher, _ = scripts
    assert launcher.trial_time_limit({"frames": 200}) == "00:25:00"
    assert launcher.trial_time_limit({"frames": 400}) == "00:50:00"
    assert launcher.trial_time_limit({"frames": 140}) == "00:25:00"


def test_perception_variants_forward_tracker_and_mount_and_are_checked_against_the_capture(tmp_path, scripts):
    launcher, run = scripts
    targets = [{"target_tree_index": 0, "component_first_vertex": 590}]
    plan = launcher.experiment_plan(targets, "source", "raw", "similarity_mount")
    row = plan["runs"][0]
    assert row["strategy"]["motion_model"] == "similarity" and row["strategy"]["mount_side_rule"] == "mirror_if_end_on"
    assert plan["frames"] == 200 and plan["maximum_gpu_minutes"] == launcher.MINUTES_PER_TRIAL
    env = run.run_environment(plan, row, tmp_path, tmp_path / "out", {})
    assert env["PRUNING_MOTION_MODEL"] == "similarity" and env["PRUNING_MOUNT_SIDE_RULE"] == "mirror_if_end_on"
    assert env["PRUNING_APPROACH_MODE"] == "straight"
    tracker_only = launcher.experiment_plan(targets, "source", "raw", "similarity_tracker")["runs"][0]
    assert "PRUNING_MOUNT_SIDE_RULE" not in run.run_environment(plan, tracker_only, tmp_path, tmp_path, {})

    def report(motion_model, rule):
        return {
            "photometric_normalization": "raw",
            "frame_count": 200,
            "approach_strategy": {"mode": "straight", "standoff_m": 0.0, "max_step_m": 0.004},
            "tracker_config": {"motion_model": motion_model},
            "camera_mount_selection": {"rule": rule},
            "blender_scene": {"daylight": {"preset": "source"}, "target": {"id": "tree0_SPUR_component_590"}},
        }

    assert run.configuration_matches(report("similarity", "mirror_if_end_on"), row, plan)
    assert not run.configuration_matches(report("translation", "mirror_if_end_on"), row, plan)
    assert not run.configuration_matches(report("similarity", "fixed"), row, plan)
    assert run.run_label(0, row) == "run_00_source_tree0_v590_similarity_mount"


def test_vision_storage_preflight_scales_with_runs_and_frames_and_refuses_over_the_line(tmp_path, scripts):
    queue, _ = scripts
    targets = [{"target_tree_index": 0, "component_first_vertex": v} for v in (530, 590)]
    plan = queue.experiment_plan(targets, "source", "raw", "fine_step")
    assert queue.estimated_output_bytes(plan) == 2 * queue.BYTES_PER_200_FRAME_RUN * 2
    report = queue.vision_storage_preflight(tmp_path, plan, share_used_bytes=1_000_000_000_000)
    assert report["ok"] and json.loads((tmp_path / "storage_preflight.json").read_text())["ok"]
    with pytest.raises(RuntimeError):
        queue.vision_storage_preflight(tmp_path, plan, share_used_bytes=2_198_000_000_000)
    with pytest.raises(ValueError):
        queue.vision_storage_preflight(tmp_path, plan)
    empty = tmp_path / "du.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        queue.vision_storage_preflight(tmp_path, plan, share_du_file=empty)


def test_planned_rows_carry_each_targets_own_orientation_and_are_checked(tmp_path, scripts):
    launcher, run = scripts
    quat = [0.6104, -0.3641, -0.5968, -0.3724]
    targets = [
        {"target_tree_index": 1, "component_first_vertex": 19444, "planned_final_tool_quat_wxyz": quat},
        {"target_tree_index": 1, "component_first_vertex": 14944, "planned_final_tool_quat_wxyz": None},
    ]
    plan = launcher.experiment_plan(targets, "source", "raw", "planned_pose")
    first, second = plan["runs"]
    assert first["strategy"]["planned_tool_quat_wxyz"] == quat and second["strategy"]["planned_tool_quat_wxyz"] is None
    env = run.run_environment(plan, first, tmp_path, tmp_path / "out", {})
    assert env["PRUNING_APPROACH_MODE"] == "planned_pose_standoff" and env["PRUNING_MAX_ROTATION_DEG"] == "1.5"
    assert [float(v) for v in env["PRUNING_PLANNED_TOOL_QUAT"].split(",")] == quat
    assert "PRUNING_PLANNED_TOOL_QUAT" not in run.run_environment(plan, second, tmp_path, tmp_path, {})

    def report(recorded_quat):
        return {
            "photometric_normalization": "raw",
            "frame_count": 200,
            "approach_strategy": {
                "mode": "planned_pose_standoff",
                "standoff_m": 0.06,
                "max_step_m": 0.004,
                "planned_tool_quat_wxyz": recorded_quat,
            },
            "blender_scene": {"daylight": {"preset": "source"}, "target": {"id": "tree1_SPUR_component_19444"}},
        }

    assert run.configuration_matches(report(quat), first, plan)
    assert not run.configuration_matches(report(None), first, plan)
    assert not run.configuration_matches(report([1.0, 0.0, 0.0, 0.0]), first, plan)


def test_placement_overrides_are_validated_and_ordered_after_the_defaults(scripts):
    queue, _ = scripts
    assert queue.placement_options(None) == []
    options = queue.placement_options({"partition": "gpu,ampere", "constraint": "a40|rtx8000", "minutes_per_task": 45})
    assert options == ["--partition", "gpu,ampere", "--constraint", "a40|rtx8000", "--time", "00:45:00"]
    assert queue.placement_options({"minutes_per_task": 90}) == ["--time", "01:30:00"]
    for bad in (
        {"partition": "gpu; rm -rf"},
        {"constraint": "a40 || x"},
        {"minutes_per_task": 0},
        {"minutes_per_task": 4.5},
    ):
        with pytest.raises(ValueError):
            queue.placement_options(bad)


def test_jaw_shadow_counterfactual_is_forwarded_and_the_readback_decides_the_arm(tmp_path, scripts):
    launcher, run = scripts
    targets = [{"target_tree_index": 1, "component_first_vertex": 14944}]
    plan = launcher.experiment_plan(targets, "evening", "raw", "jaw_no_shadow")
    row = plan["runs"][0]
    assert row["strategy"]["jaw_casts_shadow"] is False and row["strategy"]["mode"] == "straight"
    assert run.run_environment(plan, row, tmp_path, tmp_path / "out", {})["PRUNING_JAW_CASTS_SHADOW"] == "0"
    baseline = launcher.experiment_plan(targets, "evening", "raw", "baseline")
    assert "PRUNING_JAW_CASTS_SHADOW" not in run.run_environment(baseline, baseline["runs"][0], tmp_path, tmp_path, {})

    def report(shadow):
        scene = {"daylight": {"preset": "evening"}, "target": {"id": "tree1_SPUR_component_14944"}}
        if shadow is not None:
            scene["jaw_proxy_shadow"] = shadow
        return {
            "photometric_normalization": "raw",
            "frame_count": 200,
            "approach_strategy": {"mode": "straight", "standoff_m": 0.0, "max_step_m": 0.004},
            "blender_scene": scene,
        }

    off = {"casts_shadow_requested": False, "do_not_cast_shadows_readback": [True, True]}
    assert run.configuration_matches(report(off), row, plan)
    # The primvar must be on both cubes; a request without the readback is not the arm.
    assert not run.configuration_matches(report({**off, "do_not_cast_shadows_readback": [True, None]}), row, plan)
    assert not run.configuration_matches(report(None), row, plan)
    assert run.run_label(0, row) == "run_00_evening_tree1_v14944_jaw_no_shadow"


def _jaw_hold_report(target_id, quat, jaw_block, hold_block, tracker_minimum=140):
    report = {
        "photometric_normalization": "raw",
        "frame_count": 200,
        "approach_strategy": {
            "mode": "planned_pose_standoff",
            "standoff_m": 0.06,
            "max_step_m": 0.004,
            "planned_tool_quat_wxyz": quat,
        },
        "blender_scene": {"daylight": {"preset": "source"}, "target": {"id": target_id}},
        "tracker_config": {"min_unmasked_patch_elements": tracker_minimum},
    }
    if jaw_block is not None:
        report["jaw_self_mask"] = jaw_block
    if hold_block is not None:
        report["closure_hold"] = hold_block
    return json.loads(json.dumps(report))


def test_jaw_hold_strategy_forwards_both_flags_and_the_capture_must_show_them(tmp_path, scripts):
    from isaaclab_pruning.perception.jaw_self_mask import registered_closure_hold, registered_jaw_self_mask

    launcher, run = scripts
    quat = [-0.0037, -0.3524, 0.3494, 0.8682]
    targets = [
        {"target_tree_index": 0, "component_first_vertex": 530, "planned_final_tool_quat_wxyz": quat},
        {"target_tree_index": 1, "component_first_vertex": 14944, "planned_final_tool_quat_wxyz": None},
    ]
    plan = launcher.experiment_plan(targets, "source", "raw", "planned_pose_jaw_hold")
    assert plan["frames"] == 200 and plan["maximum_gpu_minutes"] == launcher.MINUTES_PER_TRIAL * 2
    row = plan["runs"][0]
    assert row["strategy"]["jaw_self_mask"] is True and row["strategy"]["closure_hold"] is True
    assert row["strategy"]["mode"] == "planned_pose_standoff" and row["strategy"]["planned_tool_quat_wxyz"] == quat
    env = run.run_environment(plan, row, tmp_path, tmp_path / "out", {"PRUNING_JAW_SELF_MASK": "0"})
    assert env["PRUNING_JAW_SELF_MASK"] == "1" and env["PRUNING_CLOSURE_HOLD"] == "1"
    assert env["PRUNING_APPROACH_MODE"] == "planned_pose_standoff"
    assert run.run_label(0, row) == "run_00_source_tree0_v530_planned_pose_jaw_hold"
    # Every earlier strategy forwards neither flag, so the renderer keeps its default (off).
    for name in launcher.STRATEGIES:
        if name == "planned_pose_jaw_hold":
            continue
        other = launcher.experiment_plan(targets, "source", "raw", name)
        other_env = run.run_environment(other, other["runs"][0], tmp_path, tmp_path, {})
        assert "PRUNING_JAW_SELF_MASK" not in other_env and "PRUNING_CLOSURE_HOLD" not in other_env

    jaw = {"enabled": True, "constants": registered_jaw_self_mask(), "model": "two-box visual surrogate"}
    hold = {"enabled": True, "thresholds": registered_closure_hold()}
    target_id = "tree0_SPUR_component_530"
    assert run.configuration_matches(_jaw_hold_report(target_id, quat, jaw, hold), row, plan)
    refused = [
        (None, None),
        ({**jaw, "enabled": False}, hold),
        (jaw, {**hold, "enabled": False}),
        (jaw, None),
        (None, hold),
        ({**jaw, "constants": {**registered_jaw_self_mask(), "mask_margin_px": 1.0}}, hold),
        ({**jaw, "constants": {**registered_jaw_self_mask(), "min_unmasked_patch_elements": 60}}, hold),
        (jaw, {**hold, "thresholds": {"max_translation_m": 0.001, "max_rotation_deg": 0.25}}),
        ({**jaw, "enabled": "true"}, hold),
    ]
    for jaw_block, hold_block in refused:
        assert not run.configuration_matches(_jaw_hold_report(target_id, quat, jaw_block, hold_block), row, plan)
    # The tracker that ran must have used the registered minimum too.
    assert not run.configuration_matches(_jaw_hold_report(target_id, quat, jaw, hold, tracker_minimum=60), row, plan)
    # A planned-pose capture without the arms still matches its own (flagless) strategy row.
    planned = launcher.experiment_plan(targets, "source", "raw", "planned_pose")
    assert run.configuration_matches(_jaw_hold_report(target_id, quat, None, None), planned["runs"][0], planned)
    # ...but a flagless row refuses a capture that ran with either arm enabled.
    assert not run.configuration_matches(_jaw_hold_report(target_id, quat, jaw, None), planned["runs"][0], planned)
    assert not run.configuration_matches(_jaw_hold_report(target_id, quat, None, hold), planned["runs"][0], planned)


def test_jaw_hold_per_batch_registers_split_the_planned_pose_register(scripts):
    launcher, _ = scripts
    evidence = Path(__file__).resolve().parents[1] / "docs/evidence"
    source = json.loads((evidence / "eval_targets_planned_pose_2026-09-27.json").read_text())
    by_vertex = {item["component_first_vertex"]: item for item in source["targets"]}
    seen = []
    for name, partner in (("a", 19444), ("b", 14944), ("c", 15004)):
        path = evidence / f"eval_targets_jaw_hold_{name}_2026-09-30.json"
        targets, provenance = launcher.load_targets(path)
        assert [item["component_first_vertex"] for item in targets] == [530, partner]
        assert all(item == by_vertex[item["component_first_vertex"]] for item in targets)
        derived = json.loads(path.read_text())["derived_from"][0]
        expected = hashlib.sha256((evidence / "eval_targets_planned_pose_2026-09-27.json").read_bytes()).hexdigest()
        assert derived["sha256"] == expected and provenance["targets_sha256"]
        plan = launcher.experiment_plan(targets, "source", "raw", "planned_pose_jaw_hold")
        assert [run["strategy"]["planned_tool_quat_wxyz"] for run in plan["runs"]] == [
            by_vertex[530]["planned_final_tool_quat_wxyz"],
            by_vertex[partner]["planned_final_tool_quat_wxyz"],
        ]
        seen.append(partner)
    assert sorted(seen) == [14944, 15004, 19444]


DEPTH_STRATEGIES = {
    "baseline_depth_appearance": "straight",
    "planned_pose_depth_appearance": "planned_pose_standoff",
    "tool_axis_standoff_depth_appearance": "tool_axis_standoff",
}


def _depth_report(block, mode="straight", standoff=0.0):
    report = {
        "photometric_normalization": "raw",
        "frame_count": 200,
        "approach_strategy": {"mode": mode, "standoff_m": standoff, "max_step_m": 0.004},
        "blender_scene": {"daylight": {"preset": "evening"}, "target": {"id": "tree1_SPUR_component_14944"}},
    }
    if block is not None:
        report["depth_appearance"] = block
    return json.loads(json.dumps(report))


def test_depth_strategies_forward_only_their_flag_and_the_capture_must_show_the_registered_constants(tmp_path, scripts):
    from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance

    launcher, run = scripts
    targets = [{"target_tree_index": 1, "component_first_vertex": 14944, "planned_final_tool_quat_wxyz": None}]
    for name, mode in DEPTH_STRATEGIES.items():
        assert launcher.STRATEGY_PROTOCOLS[name] == "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md"
        plan = launcher.experiment_plan(targets, "evening", "raw", name)
        row = plan["runs"][0]
        assert row["strategy"]["depth_appearance"] is True and row["strategy"]["mode"] == mode
        assert "jaw_casts_shadow" not in row["strategy"]
        env = run.run_environment(plan, row, tmp_path, tmp_path / "out", {"PRUNING_DEPTH_APPEARANCE": "0"})
        assert env["PRUNING_DEPTH_APPEARANCE"] == "1"
        assert not {"PRUNING_JAW_SELF_MASK", "PRUNING_CLOSURE_HOLD", "PRUNING_JAW_CASTS_SHADOW"} & set(env)
        assert run.run_label(0, row) == f"run_00_evening_tree1_v14944_{name}"
    for name in launcher.STRATEGIES:
        if name not in DEPTH_STRATEGIES:
            other = launcher.experiment_plan(targets, "evening", "raw", name)
            assert "PRUNING_DEPTH_APPEARANCE" not in run.run_environment(
                other, other["runs"][0], tmp_path, tmp_path, {}
            )

    plan = launcher.experiment_plan(targets, "evening", "raw", "baseline_depth_appearance")
    row = plan["runs"][0]
    good = {"enabled": True, "constants": registered_depth_appearance()}
    assert run.configuration_matches(_depth_report(good), row, plan)
    for bad in (
        None,
        {**good, "enabled": False},
        {**good, "enabled": "true"},
        {**good, "constants": {**registered_depth_appearance(), "max_near_fraction": 0.1}},
    ):
        assert not run.configuration_matches(_depth_report(bad), row, plan)
    baseline = launcher.experiment_plan(targets, "evening", "raw", "baseline")
    assert run.configuration_matches(_depth_report(None), baseline["runs"][0], baseline)
    assert run.configuration_matches(_depth_report({**good, "enabled": False}), baseline["runs"][0], baseline)
    assert not run.configuration_matches(_depth_report(good), baseline["runs"][0], baseline)


def test_depth_loop_registers_copy_their_sources_unchanged(scripts):
    launcher, _ = scripts
    evidence = Path(__file__).resolve().parents[1] / "docs/evidence"
    expected = {
        "tree1": ("eval_targets_tree1_listed_2026-09-23.json", [14944, 15004], "baseline_depth_appearance"),
        "15004": ("eval_targets_tree1_listed_2026-09-23.json", [15004], "baseline_depth_appearance"),
        "19444": ("eval_targets_planned_pose_2026-09-27.json", [19444], "planned_pose_depth_appearance"),
        "12142": ("eval_targets_tree0_2026-09-23.json", [12142], "tool_axis_standoff_depth_appearance"),
    }
    for name, (source, vertices, strategy) in expected.items():
        path = evidence / f"eval_targets_depth_loop_{name}_2026-10-01.json"
        targets, _ = launcher.load_targets(path)
        by_vertex = {t["component_first_vertex"]: t for t in json.loads((evidence / source).read_text())["targets"]}
        assert [t["component_first_vertex"] for t in targets] == vertices
        assert all(t == by_vertex[t["component_first_vertex"]] for t in targets)
        derived = json.loads(path.read_text())["derived_from"][0]
        assert derived["sha256"] == hashlib.sha256((evidence / source).read_bytes()).hexdigest()
        plan = launcher.experiment_plan(targets, "source", "raw", strategy)
        assert all(r["strategy"]["depth_appearance"] is True for r in plan["runs"])
    planned = launcher.experiment_plan(
        launcher.load_targets(evidence / "eval_targets_depth_loop_19444_2026-10-01.json")[0],
        "source",
        "raw",
        "planned_pose_depth_appearance",
    )
    assert planned["runs"][0]["strategy"]["planned_tool_quat_wxyz"] is not None
