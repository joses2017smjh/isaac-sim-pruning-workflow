"""Protect the agreement-arm C0 gate: where the agreement arm may depart from a live strict recording, and the
check that its depth test equals the live one at every grey-check failure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def c0(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("check_agreement_c0", tools / "check_agreement_c0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(accept=True, strict=0.2, ncc=0.1, n_verified=169):
    return {
        "accept": accept,
        "ncc": ncc,
        "reasons": [] if accept else ["surface_moved_or_wrong_surface"],
        "failed_conditions": [] if accept else ["4_median_abs"],
        "not_evaluated_conditions": [],
        "conditions": {"4_median_abs": {"value": 4e-5, "limit": 0.003, "evaluated": True, "passed": accept}},
        "stats": {"n_verified": n_verified, "median_abs_m": 4e-5},
        "strict_confidence": strict,
        "agreement_confidence": 0.375,
        "arm": "strict",
    }


def _frame(index, event=None, state="tracking"):
    return {"index": index, "live_vision": {"measurement": {"state": state, "depth_appearance": event}}}


def test_the_agreement_arm_may_depart_only_where_an_accepted_event_ended_below_the_confidence_gate(c0):
    frames = [
        _frame(0),
        _frame(1, _event(strict=0.19)),
        _frame(2, _event(accept=False)),
        _frame(3, _event(strict=0.0005)),
    ]
    assert c0.expected_agreement_departures(frames, 3) == [3]
    assert c0.expected_agreement_departures(frames, 2) == []
    assert c0.expected_agreement_departures([_frame(0, _event(strict=None))], 0) == [0]  # no confidence computed


def _write_run(tmp_path, frames):
    run = tmp_path / "batch" / "run_00"
    run.mkdir(parents=True)
    (run / "frames.json").write_text(json.dumps({"fps": 10, "frames": frames}))
    return run


def _replay(rows, last, cut=()):
    return {
        "compared_through_frame": last,
        "frames_detail": rows,
        "comparison_through_recorded_stop": {"cut_mismatch_frames": list(cut), "measurement_mismatch_frames": []},
        "replay": {"stop_frame": None, "stop_reason": None},
    }


def test_live_agreement_check_passes_the_expected_departure_and_refuses_any_other(c0, tmp_path, monkeypatch):
    low = _event(strict=0.0005)
    frames = [_frame(0), _frame(1, _event(strict=0.19)), _frame(2, low, state="tracking_lost")]
    run = _write_run(tmp_path, frames)

    def rows(departures, event_at_2=low, state_at_2="tracking"):
        return [
            {
                "index": 0,
                "measurement_matches_recording": 0 not in departures,
                "state": "tracking",
                "depth_appearance": None,
            },
            {
                "index": 1,
                "measurement_matches_recording": 1 not in departures,
                "state": "tracking",
                "depth_appearance": _event(strict=0.19),
            },
            {
                "index": 2,
                "measurement_matches_recording": 2 not in departures,
                "state": state_at_2,
                "depth_appearance": event_at_2,
            },
        ]

    cases = {
        "ok": (rows({2}), []),
        "missing departure": (rows(set()), ["departure_frames"]),
        "extra departure": (rows({1, 2}), ["departure_frames"]),
        "did not continue": (rows({2}, state_at_2="tracking_lost"), ["agreement_did_not_continue"]),
        "different depth test": (rows({2}, event_at_2=_event(strict=0.0005, n_verified=150)), ["event_core"]),
        "event missing": (rows({2}, event_at_2=None), ["event_presence"]),
    }
    for name, (replayed, kinds) in cases.items():
        monkeypatch.setattr(c0.controller_replay, "replay_run", lambda path, **flags: _replay(replayed, 2))
        problems, detail = c0.check_live_agreement(run)
        assert [problem["kind"] for problem in problems] == kinds, name
        assert detail["expected_departure_frames"] == [2]


def test_the_arm_label_and_the_arm_dependent_confidences_are_not_part_of_the_depth_test(c0):
    strict = _event(strict=0.0005)
    agreement = {**strict, "arm": "agreement", "strict_confidence": 0.0005, "decision": "continues"}
    assert c0.event_core(strict) == c0.event_core(agreement)
    assert "arm" not in c0.EVENT_CORE and "decision" not in c0.EVENT_CORE
