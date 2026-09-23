"""Protect lighting-pilot aggregation: incomplete runs, fresh grades and pairing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def summarize(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("summarize_lighting_pilot", tools / "summarize_lighting_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(index, state, features, confidence, released=False, reason=None):
    return {
        "index": index,
        "time_s": round(0.1 * (index + 1), 3),
        "wrist_rgb_std": 51.9,
        "detachment_requested_after_capture": released,
        "live_vision": {
            "measurement": {"state": state, "reason": reason, "feature_count": features, "confidence": confidence}
        },
    }


def test_tracking_profile_separates_loss_after_release_from_loss_before(summarize):
    after = [_frame(0, "tracking", 30, 0.9), _frame(1, "tracking", 12, 0.4, released=True)]
    after.append(_frame(2, "tracking_lost", 0, 0.0, reason="optical_flow_failed"))
    profile = summarize.tracking_profile(after)
    assert profile["release_frame_index"] == 1
    assert profile["first_nontracking_frame_index"] == 2
    assert profile["first_nontracking_is_after_release"] is True
    # Minima are taken over tracking frames only, so a lost frame cannot lower them.
    assert profile["tracked_feature_count_min"] == 12
    assert profile["tracked_confidence_min"] == pytest.approx(0.4)

    before = [_frame(0, "tracking", 30, 0.9), _frame(1, "tracking_lost", 0, 0.0, reason="too_few_roundtrip_inliers")]
    early = summarize.tracking_profile(before)
    assert early["release_frame_index"] is None
    assert early["first_nontracking_frame_index"] == 1
    assert early["first_nontracking_is_after_release"] is None
    assert early["first_nontracking_reason"] == "too_few_roundtrip_inliers"


def test_planned_run_without_capture_is_reported_incomplete(tmp_path, summarize):
    batch = tmp_path / "batch"
    batch.mkdir()
    row = {"index": 3, "daylight": "morning", "photometric_normalization": "clahe"}
    entry = summarize.summarize_run(batch, row, {}, "999")
    assert entry["status"] == "incomplete"
    assert entry["grade"] is None
    assert entry["array_task"] == "999_3"
    assert "Missing report.json" in entry["incomplete_reason"]


def test_incomplete_runs_stay_in_the_table_and_are_counted(tmp_path, monkeypatch, summarize):
    batch = tmp_path / "batch"
    batch.mkdir()
    plan = {
        "code_revision": "abc",
        "created_utc": "2026-09-19T00:00:00+00:00",
        "runs": [
            {"index": 0, "daylight": "source", "photometric_normalization": "raw"},
            {"index": 1, "daylight": "source", "photometric_normalization": "clahe"},
        ],
    }
    (batch / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(summarize, "sacct_rows", lambda _job: {})
    document = summarize.build(batch, "2026-09-23")
    assert document["planned_run_count"] == 2
    assert document["graded_run_count"] == 0
    assert document["incomplete_run_count"] == 2
    assert document["passing_run_count"] == 0
    assert [run["index"] for run in document["runs"]] == [0, 1]
    # An unpaired lighting preset must not be reported as a completed comparison.
    assert document["paired_comparison"] == []


def test_paired_comparison_reports_both_margin_directions(summarize):
    def run(mode, ok, tracking, features, confidence):
        return {
            "daylight": "evening",
            "photometric_normalization": mode,
            "grade": {"ok": ok},
            "tracking": {
                "tracking_frames": tracking,
                "tracked_feature_count_min": features,
                "tracked_confidence_min": confidence,
            },
        }

    pairs = summarize.paired_comparison([run("raw", True, 78, 13, 0.4995), run("clahe", True, 77, 15, 0.4027)])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["complete_pair"] is True
    assert pair["task_completion_delta"] == "none"
    # CLAHE raised the feature floor but lowered the confidence floor; both are recorded.
    assert pair["clahe_feature_count_min"] > pair["raw_feature_count_min"]
    assert pair["clahe_confidence_min"] < pair["raw_confidence_min"]


def test_half_a_pair_is_not_reported_as_a_comparison(summarize):
    only_raw = [
        {
            "daylight": "morning",
            "photometric_normalization": "raw",
            "grade": {"ok": True},
            "tracking": {"tracking_frames": 78, "tracked_feature_count_min": 4, "tracked_confidence_min": 0.17},
        }
    ]
    assert summarize.paired_comparison(only_raw) == [{"daylight": "morning", "complete_pair": False}]


def test_sacct_helper_ignores_malformed_rows(monkeypatch, summarize):
    monkeypatch.setattr(
        summarize.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "21360571_0|21362298|COMPLETED|0:0|00:16:21|s|e|cn-r-1|gpu=1\nbroken|row\n",
    )
    rows = summarize.sacct_rows("21360571")
    assert set(rows) == {"21360571_0"}
    assert rows["21360571_0"]["JobIDRaw"] == "21362298"
    assert rows["21360571_0"]["State"] == "COMPLETED"


def test_missing_scheduler_never_fails_the_summary(monkeypatch, summarize):
    def explode(*_args, **_kwargs):
        raise OSError("sacct is unavailable")

    monkeypatch.setattr(summarize.subprocess, "check_output", explode)
    assert summarize.sacct_rows("21360571") == {}
