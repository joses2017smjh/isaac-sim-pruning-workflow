"""Protect the success-rate aggregation: the SQL is the path, and N never shrinks."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def aggregator(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("aggregate_eval", tools / "aggregate_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index, **overrides):
    row = {
        "condition": "source",
        "run_directory": f"run_{index:02d}",
        "target_tree_index": 0,
        "component_first_vertex": 100 + index,
        "daylight": "source",
        "photometric_normalization": "raw",
        "status": "graded",
        "checks_passed": 17,
        "checks_total": 17,
        "applied_commands": 68,
        "stop_reason": None,
        "tracking_frames": 78,
        "recorded_frames": 200,
        "tracked_feature_count_min": 18,
        "tracked_confidence_min": 0.58,
        "centroid_error_m": None,
    }
    row.update(overrides)
    return row


def _wilson_reference(successes, attempts, z=1.959963984540054):
    """Wilson score interval, written out independently of the SQL under test.

    Keeping a second implementation here means the query is checked against the
    formula rather than against a literal that could be mis-transcribed.
    """
    p_hat = successes / attempts
    denominator = 1 + z * z / attempts
    center = (p_hat + z * z / (2 * attempts)) / denominator
    half = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / attempts + z * z / (4 * attempts * attempts))
    return max(0.0, center - half), min(1.0, center + half)


@pytest.mark.parametrize(("successes", "attempts"), [(0, 20), (20, 20), (10, 20), (1, 20), (3, 7), (0, 1)])
def test_wilson_interval_matches_an_independent_implementation(aggregator, successes, attempts):
    rows = [_row(index, checks_passed=17 if index < successes else 8) for index in range(attempts)]
    rate = aggregator.aggregate(rows)["success_rate"]
    lower, upper = _wilson_reference(successes, attempts)
    assert rate["successes"] == successes
    assert rate["attempts"] == attempts
    assert rate["wilson_lower"] == pytest.approx(lower, abs=1e-12)
    assert rate["wilson_upper"] == pytest.approx(upper, abs=1e-12)
    # An interval outside [0, 1] is not a proportion.
    assert 0.0 <= rate["wilson_lower"] <= rate["wilson_upper"] <= 1.0


def test_wilson_anchors_hold_at_the_published_values(aggregator):
    # Two independently checked anchors, so a shared mistake in both
    # implementations would still be caught.
    rows = [_row(index, checks_passed=8) for index in range(20)]
    rate = aggregator.aggregate(rows)["success_rate"]
    assert rate["wilson_upper"] == pytest.approx(0.1611, abs=5e-4)

    rows = [_row(index, checks_passed=17 if index < 10 else 8) for index in range(20)]
    rate = aggregator.aggregate(rows)["success_rate"]
    assert (rate["wilson_lower"], rate["wilson_upper"]) == (
        pytest.approx(0.2993, abs=5e-4),
        pytest.approx(0.7007, abs=5e-4),
    )


def test_rejections_and_incompletes_stay_in_the_denominator(aggregator):
    rows = [
        _row(0),
        _row(1, status="rejected_layout_startup_contact", checks_passed=0, checks_total=0),
        _row(2, status="incomplete", checks_passed=0, checks_total=0),
        _row(3, status="rejected_visibility", checks_passed=0, checks_total=0),
    ]
    result = aggregator.aggregate(rows)
    rate = result["success_rate"]
    # One pass out of four attempts. Dropping the three would report 1/1.
    assert (rate["successes"], rate["attempts"]) == (1, 4)
    assert rate["rate"] == pytest.approx(0.25)
    outcomes = {row["outcome"] for row in result["failure_taxonomy"]}
    assert {"rejected_layout_startup_contact", "incomplete", "rejected_visibility"} <= outcomes


def test_a_partial_grade_is_never_a_pass(aggregator):
    rows = [_row(0, checks_passed=16, checks_total=17), _row(1, checks_passed=17, checks_total=17)]
    result = aggregator.aggregate(rows)
    assert result["success_rate"]["successes"] == 1
    passed = {row["run_directory"]: row["passed"] for row in result["runs"]}
    assert passed == {"run_00": False, "run_01": True}


def test_zero_checks_cannot_pass_by_vacuous_equality(aggregator):
    # checks_passed == checks_total == 0 must not count as unanimous success.
    rows = [_row(0, status="incomplete", checks_passed=0, checks_total=0)]
    result = aggregator.aggregate(rows)
    assert result["success_rate"]["successes"] == 0
    assert result["runs"][0]["passed"] is False


def test_failure_taxonomy_comes_from_recorded_stop_reasons(aggregator):
    rows = [
        _row(0, checks_passed=8, stop_reason="hazard_contact"),
        _row(1, checks_passed=9, stop_reason="hazard_contact"),
        _row(2, checks_passed=9, stop_reason="vision_invalid"),
        _row(3),
    ]
    taxonomy = aggregator.aggregate(rows)["failure_taxonomy"]
    counts = {row["outcome"]: row["runs"] for row in taxonomy}
    assert counts["stopped_hazard_contact"] == 2
    assert counts["stopped_vision_invalid"] == 1
    # The passing run must not appear in a failure table.
    assert all(not row["outcome"].startswith("pass") for row in taxonomy)


def test_per_condition_split_cannot_be_hidden_by_the_total(aggregator):
    rows = [_row(index, condition="source", daylight="source") for index in range(2)]
    rows += [
        _row(index + 2, condition="morning", daylight="morning", checks_passed=8, stop_reason="vision_invalid")
        for index in range(2)
    ]
    result = aggregator.aggregate(rows)
    assert result["success_rate"]["rate"] == pytest.approx(0.5)
    by_condition = {row["condition"]: row for row in result["per_condition"]}
    assert by_condition["source"]["rate"] == pytest.approx(1.0)
    assert by_condition["morning"]["rate"] == pytest.approx(0.0)


def test_tracking_coverage_flags_runs_at_the_tracker_floors(aggregator):
    rows = [
        _row(0, tracked_feature_count_min=4, tracked_confidence_min=0.1735, checks_passed=8),
        _row(1, tracked_feature_count_min=30, tracked_confidence_min=0.9),
    ]
    coverage = {row["run_directory"]: row for row in aggregator.aggregate(rows)["tracking_coverage"]}
    assert coverage["run_00"]["touched_feature_floor"] is True
    assert coverage["run_00"]["near_confidence_floor"] is True
    assert coverage["run_01"]["touched_feature_floor"] is False
    assert coverage["run_00"]["tracking_fraction"] == pytest.approx(0.39)


def test_every_committed_query_is_applied_and_hashed(aggregator):
    result = aggregator.aggregate([_row(0)])
    names = [entry["file"] for entry in result["queries"]]
    assert names == sorted(names)
    assert "02_success_rate.sql" in names
    for entry in result["queries"]:
        assert len(entry["sha256"]) == 64


def test_aggregating_nothing_is_refused_rather_than_reported_as_zero(aggregator):
    with pytest.raises(ValueError, match="refusing to report a rate over nothing"):
        aggregator.aggregate([])


def test_missing_capture_is_classified_from_the_job_log(tmp_path, aggregator):
    batch = tmp_path / "batch"
    (batch / "logs").mkdir(parents=True)
    plan = {
        "runs": [
            {
                "index": 0,
                "daylight": "source",
                "photometric_normalization": "raw",
                "target_tree_index": 0,
                "component_first_vertex": 530,
            },
            {
                "index": 1,
                "daylight": "source",
                "photometric_normalization": "raw",
                "target_tree_index": 0,
                "component_first_vertex": 590,
            },
        ]
    }
    (batch / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (batch / "logs/prune-vision-pilot-111_0.out").write_text(
        "RuntimeError: Orchard layout rejected: startup robot contact 226.126 N > 5 N", encoding="utf-8"
    )
    (batch / "logs/prune-vision-pilot-111_1.out").write_text("some unrelated output", encoding="utf-8")

    rows = aggregator.read_batch(batch, "source")
    assert [row["status"] for row in rows] == ["rejected_layout_startup_contact", "incomplete"]
    # Both remain planned attempts and reach the denominator.
    assert len(rows) == 2
