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
