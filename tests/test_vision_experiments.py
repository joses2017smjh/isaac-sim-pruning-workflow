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
