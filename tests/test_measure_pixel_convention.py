"""tools/measure_pixel_convention.py: the half-pixel bound on synthetic runs."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "measure_pixel_convention", ROOT / "tools/measure_pixel_convention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(base, name, depths, states=None, focal=320.0):
    run = base / name
    run.mkdir(parents=True)
    report = {"camera": {"wrist_intrinsics": [[focal, 0.0, 240.0], [0.0, focal, 160.0], [0.0, 0.0, 1.0]]}}
    (run / "report.json").write_text(json.dumps(report))
    states = states or ["tracking"] * len(depths)
    frames = [{"live_vision": {"measurement": {"state": s, "depth_m": d}}} for s, d in zip(states, depths)]
    (run / "frames.json").write_text(json.dumps({"frames": frames}))


def test_half_pixel_offset_scales_with_depth_over_focal_length(tool):
    per_axis, total = tool.half_pixel_offsets([0.32, 0.64], 320.0)
    assert per_axis == pytest.approx([0.5, 1.0])
    assert total == pytest.approx([0.5 * math.sqrt(2), math.sqrt(2)])


def test_only_tracking_frames_count(tool, tmp_path):
    _run(tmp_path, "a/run_00", [0.2, 0.3, 0.4], states=["tracking", "invalid_depth", "tracking"])
    _run(tmp_path, "b/run_00", [0.25])
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"runs": [{"run": "a/run_00"}, {"run": "b/run_00"}]}))
    document = tool.build(tmp_path, evidence)
    assert document["tracking_frames"] == 3 and document["runs_with_tracking_frames"] == 2
    assert document["depth_m"]["p0"] == pytest.approx(0.2) and document["depth_m"]["p100"] == pytest.approx(0.4)
    assert document["offset_per_axis_mm"]["p100"] == pytest.approx(500 * 0.4 / 320)


def test_runs_with_different_focal_lengths_are_refused(tool, tmp_path):
    _run(tmp_path, "a/run_00", [0.2])
    _run(tmp_path, "b/run_00", [0.2], focal=300.0)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"runs": [{"run": "a/run_00"}, {"run": "b/run_00"}]}))
    with pytest.raises(ValueError):
        tool.build(tmp_path, evidence)


def test_percentile_interpolates(tool):
    assert tool.percentile([0.0, 1.0, 2.0, 3.0], 50) == pytest.approx(1.5)
    assert tool.percentile([], 50) is None
