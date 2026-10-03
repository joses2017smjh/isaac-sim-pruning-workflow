"""Protect the A1-A5 scoring of the agreement-arm closed-loop test: registered inputs, bounds and untested cases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APPEARANCE = "appearance_changed_or_occluded"
LOW = ["tracking_lost", "low_confidence"]
EVENING = [f"agree-eve-r{r}-20261003" for r in (1, 2, 3, 4)]
CTL_19444, CTL_12142 = "agree-ctl-19444-20261003", "agree-ctl-12142-20261003"


@pytest.fixture
def scorer(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_agreement_loop", tools / "score_agreement_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(accept=True, ended_by=None, failed=(), jaw_passed=None, strict=0.03, agreement=0.4, reason=None):
    jaw_passed = accept if jaw_passed is None else jaw_passed
    return {
        "accept": accept,
        "failed_conditions": list(failed),
        "not_evaluated_conditions": [],
        "strict_confidence": strict if accept else None,
        "agreement_confidence": agreement if accept else None,
        "decision": ("continues" if not ended_by else "stops_downstream") if accept else "loss_stands",
        "ended_by": ended_by,
        "result_reason": None if not ended_by else ended_by[1],
        "jaw_condition": {"passed": jaw_passed, "value": None if jaw_passed else "jaw_silhouette_touches_patch"},
        "at_frame": {"tracker_state": "tracking", "tracker_reason": reason, "cut_phase": "approach"},
    }


def _stop(frame, reason, state="tracking_lost"):
    return {"frame": frame, "stopped_reason": "vision_invalid", "tracker_state": state, "tracker_reason": reason}


def _closure(first=71, last=77):
    return [
        {"frame": i, "cut_phase": "closing", "closure_progress": 0.0, "patch_correlation": 0.9}
        for i in range(first, last + 1)
    ]


def _record(scorer, batch, outcome="pass", stop=None, events=None, scored=True, closure=None, failed_checks=None):
    registered = scorer.REGISTRY[batch]
    events = events or {}
    through = sorted(f for f in events if stop is None or f <= stop["frame"])
    return {
        "batch": batch,
        "kind": registered["kind"],
        "run_directory": registered["run_directory"],
        "target": registered["target"],
        "daylight": registered["daylight"],
        "outcome": outcome,
        "checks": "17/17" if outcome == "pass" else "9/17",
        "failed_checks": failed_checks if failed_checks is not None else ([] if outcome == "pass" else ["no_stops"]),
        "scored": scored,
        "unscored_reason": None if scored else "not graded (incomplete, checks 0/0)",
        "stop": stop,
        "grey_failures": through,
        "events": {f: events[f] for f in through},
        "post_stop_events": {f: events[f] for f in events if f not in through},
        "closure_frames": (_closure() if stop is None else []) if closure is None else closure,
    }


def _expected(scorer):
    """The registered expectation: evening 14944 accepts the frame-67 event, continues and passes; both controls
    stop on the depth test's rejection, 19444 on the jaw and 12142 at the wire."""
    records = [_record(scorer, batch, events={67: _event()}) for batch in EVENING]
    records.append(
        _record(
            scorer,
            CTL_19444,
            "stopped_vision_invalid",
            _stop(70, APPEARANCE),
            {70: _event(accept=False, failed=("4_median_abs", "J_jaw_silhouette"))},
        )
    )
    records.append(
        _record(
            scorer,
            CTL_12142,
            "stopped_vision_invalid",
            _stop(29, APPEARANCE),
            {29: _event(accept=False, failed=("3_near_fraction", "4_median_abs"))},
        )
    )
    return records


def _equal_comparisons(scorer, records):
    return {scorer.name(r): {"status": "equal", "problems": []} for r in records}


def _verdicts(scorer, records, comparisons=None):
    comparisons = comparisons if comparisons is not None else _equal_comparisons(scorer, records)
    return {p["id"]: p["verdict"] for p in scorer.score(records, comparisons)}


def _unscore(record, outcome="infrastructure"):
    """Unscore a run; outcome='pass' keeps a grader pass that must still count for nothing."""
    record.update(scored=False, unscored_reason="configuration refused", outcome=outcome)


def test_the_registered_expectation_supports_every_prediction(scorer):
    assert _verdicts(scorer, _expected(scorer)) == {f"A{i}": "supported" for i in range(1, 6)}


def test_a_prediction_with_no_tested_run_is_untested_not_partly_supported(scorer):
    records = _expected(scorer)
    for record in records:
        _unscore(record)
    comparisons = {scorer.name(r): {"status": "not_comparable", "why": "not graded"} for r in records}
    assert _verdicts(scorer, records, comparisons) == {f"A{i}": "untested" for i in range(1, 6)}


# ------------------------------------------------------------------------------------------------ A1
def test_a1_is_refuted_by_a_rejected_first_in_window_failure(scorer):
    records = _expected(scorer)
    records[1]["events"] = {60: _event(accept=False), 67: _event()}
    records[1]["grey_failures"] = [60, 67]
    a1 = scorer.score_a1(records)
    assert a1["verdict"] == "refuted"
    assert a1["clauses"][0]["observed"]["rejected_first_in_window"] == [scorer.name(records[1])]


def test_a1_is_refuted_by_any_accepted_event_ending_in_a_low_confidence_stop(scorer):
    records = _expected(scorer)  # ended_by names low_confidence
    records[0] = _record(
        scorer, EVENING[0], "stopped_vision_invalid", _stop(67, "low_confidence"), {67: _event(ended_by=LOW)}
    )
    a1 = scorer.score_a1(records)
    assert a1["verdict"] == "refuted"
    assert a1["clauses"][0]["observed"]["runs"][scorer.name(records[0])]["case"].endswith("low-confidence stop")
    records = _expected(scorer)  # only the frame's tracker reason says so
    records[0]["events"] = {67: _event(reason="low_confidence")}
    assert scorer.score_a1(records)["verdict"] == "refuted"
    records = _expected(scorer)  # an accepted closure event, outside the window, also refutes
    records[2]["events"] = {67: _event(), 74: _event(ended_by=LOW)}
    records[2]["grey_failures"] = [67, 74]
    a1 = scorer.score_a1(records)
    assert a1["verdict"] == "refuted"
    assert a1["clauses"][0]["observed"]["accepted_events_ending_low_confidence"] == {scorer.name(records[2]): [74]}


def test_a1_a_rejected_event_never_counts_as_a_low_confidence_stop(scorer):
    records = _expected(scorer)
    records[0]["events"] = {67: _event(), 80: _event(accept=False, reason="low_confidence")}
    records[0]["grey_failures"] = [67, 80]
    assert scorer.score_a1(records)["verdict"] == "supported"


def test_a1_untested_runs_and_the_untested_prediction(scorer):
    records = _expected(scorer)
    records[1]["events"], records[1]["grey_failures"] = {}, []
    a1 = scorer.score_a1(records)
    assert a1["verdict"] == "partly supported"
    assert scorer.name(records[1]) in a1["clauses"][0]["observed"]["untested"]
    records = _expected(scorer)
    records[1]["events"], records[1]["grey_failures"] = {80: _event(accept=False)}, [80]  # outside the window
    assert scorer.score_a1(records)["verdict"] == "partly supported"
    records = _expected(scorer)
    _unscore(records[3])
    assert scorer.score_a1(records)["verdict"] == "partly supported"
    records = _expected(scorer)  # no in-window failure anywhere, an accepted closure event continues: untested
    for record in records[:4]:
        record["events"], record["grey_failures"] = {74: _event()}, [74]
    assert scorer.score_a1(records)["verdict"] == "untested"
    records[0]["events"][74] = _event(ended_by=LOW)  # ... unless it ends in a low-confidence stop
    assert scorer.score_a1(records)["verdict"] == "refuted"


def test_a1_a_post_stop_failure_is_not_judged(scorer):
    records = _expected(scorer)
    records[0] = _record(
        scorer,
        EVENING[0],
        "stopped_hazard",
        _stop(50, None, state="tracking"),
        {65: _event(accept=False)},
    )
    a1 = scorer.score_a1(records)
    assert a1["verdict"] == "partly supported" and records[0]["post_stop_events"]


def test_a1_a_stop_at_an_accepted_event_on_another_gate_counts_against_a2_not_a1(scorer):
    records = _expected(scorer)
    gate = ["tracking_lost", "depth_measurement_rejected"]
    records[0] = _record(
        scorer,
        EVENING[0],
        "stopped_vision_invalid",
        _stop(67, "depth_measurement_rejected"),
        {67: _event(ended_by=gate)},
    )
    a1 = scorer.score_a1(records)
    observed = a1["clauses"][0]["observed"]
    assert a1["verdict"] == "supported"
    assert scorer.name(records[0]) in observed["stops_at_accepted_events_on_another_gate_counted_against_a2"]
    assert "another gate" in observed["runs"][scorer.name(records[0])]["case"]
    a2 = scorer.score_a2(records)
    assert a2["verdict"] == "supported"  # 3 of 4 pass and the stop is not on appearance
    assert a2["clauses"][0]["observed"]["passes"] == "3 of 4"


# ------------------------------------------------------------------------------------------------ A2
def _fail(scorer, records, index, stop, events=None):
    batch = records[index]["batch"]
    records[index] = _record(scorer, batch, "stopped_vision_invalid", stop, events if events is not None else {})


def test_a2_bounds(scorer):
    records = _expected(scorer)
    _fail(scorer, records, 0, _stop(67, APPEARANCE), {67: _event(accept=False)})
    assert scorer.score_a2(records)["verdict"] == "partly supported"  # 3 pass but one appearance stop
    _fail(scorer, records, 1, _stop(62, APPEARANCE), {62: _event(accept=False)})
    assert scorer.score_a2(records)["verdict"] == "refuted"  # two appearance stops in the window
    records = _expected(scorer)
    _fail(scorer, records, 0, _stop(67, "low_confidence"), {67: _event(ended_by=LOW)})
    assert scorer.score_a2(records)["verdict"] == "partly supported"  # low confidence at an accepted event
    _fail(scorer, records, 1, _stop(67, "low_confidence"), {67: _event(ended_by=LOW)})
    assert scorer.score_a2(records)["verdict"] == "refuted"
    records = _expected(scorer)
    _fail(scorer, records, 0, _stop(67, "low_confidence"))  # no depth event: not a stop on appearance
    _fail(scorer, records, 1, _stop(70, APPEARANCE), {70: _event(accept=False)})  # outside the window
    a2 = scorer.score_a2(records)
    assert a2["verdict"] == "partly supported" and a2["clauses"][0]["observed"]["appearance_stops_in_window"] == []
    records = _expected(scorer)
    for index in (0, 1, 2):
        _fail(scorer, records, index, _stop(150, None, state="tracking"))
    assert scorer.score_a2(records)["verdict"] == "refuted"  # fewer than 2 pass, every run scored


def test_a2_unscored_runs_are_not_failures(scorer):
    records = _expected(scorer)
    for index in (0, 1):
        _unscore(records[index])
    assert scorer.score_a2(records)["verdict"] == "partly supported"
    for index in (2, 3):
        _fail(scorer, records, index, _stop(150, None, state="tracking"))
    assert scorer.score_a2(records)["verdict"] == "partly supported"  # 0 + 2 unscored could still reach 2
    records = _expected(scorer)
    _unscore(records[0])
    for index in (1, 2, 3):
        _fail(scorer, records, index, _stop(150, None, state="tracking"))
    assert scorer.score_a2(records)["verdict"] == "refuted"  # 0 + 1 unscored cannot reach 2


def test_a2_reports_each_failed_run(scorer):
    records = _expected(scorer)
    records[2] = _record(
        scorer,
        EVENING[2],
        "stopped_vision_invalid",
        _stop(74, "low_confidence"),
        {67: _event(), 74: _event(ended_by=LOW), 90: _event(accept=False)},
        closure=_closure(71, 74),
        failed_checks=["no_recorded_stops", "returned_home_and_final_phase_complete"],
    )
    failed = scorer.score_a2(records)["clauses"][0]["observed"]["failed_runs"][scorer.name(records[2])]
    assert failed["stop_frame"] == 74
    assert failed["stop"]["tracker_state"] == "tracking_lost" and failed["stop"]["tracker_reason"] == "low_confidence"
    assert sorted(failed["depth_events"]) == [67, 74] and sorted(failed["post_stop_depth_events"]) == [90]
    assert failed["failed_checks"] == ["no_recorded_stops", "returned_home_and_final_phase_complete"]


@pytest.mark.parametrize("frame, inside", [(55, False), (56, True), (67, True), (68, False)])
def test_the_window_is_frames_56_to_67_inclusive(scorer, frame, inside):
    records = _expected(scorer)
    records[1]["events"], records[1]["grey_failures"] = {frame: _event(accept=False)}, [frame]
    assert scorer.score_a1(records)["verdict"] == ("refuted" if inside else "partly supported")
    if frame != 67:  # with an accepted event at 67, only an in-window rejection decides
        records = _expected(scorer)
        records[1]["events"] = {frame: _event(accept=False), 67: _event()}
        records[1]["grey_failures"] = sorted(records[1]["events"])
        assert scorer.score_a1(records)["verdict"] == ("refuted" if inside else "supported")
    records = _expected(scorer)
    for index in (0, 1):
        _fail(scorer, records, index, _stop(frame, APPEARANCE), {frame: _event(accept=False)})
    a2 = scorer.score_a2(records)
    assert a2["verdict"] == ("refuted" if inside else "partly supported")
    assert len(a2["clauses"][0]["observed"]["appearance_stops_in_window"]) == (2 if inside else 0)


def test_a2_an_unscored_pass_or_appearance_stop_counts_for_nothing(scorer):
    records = _expected(scorer)
    _unscore(records[0], outcome="pass")
    for index in (1, 2, 3):
        _fail(scorer, records, index, _stop(150, None, state="tracking"))
    a2 = scorer.score_a2(records)
    assert a2["verdict"] == "refuted" and a2["clauses"][0]["observed"]["passes"] == "0 of 4"  # 0 + 1 unscored < 2
    records = _expected(scorer)
    _unscore(records[3], outcome="pass")
    assert scorer.score_a2(records)["verdict"] == "partly supported"  # 3 scored passes and 1 unscored run
    records = _expected(scorer)
    for index in (0, 1):
        records[index] = _record(
            scorer,
            EVENING[index],
            "stopped_vision_invalid",
            _stop(67, APPEARANCE),
            {67: _event(accept=False)},
            scored=False,
        )
    a2 = scorer.score_a2(records)
    assert a2["verdict"] == "partly supported" and a2["clauses"][0]["observed"]["appearance_stops_in_window"] == []


def test_a2_two_in_window_stops_refute_only_on_appearance(scorer):
    records = _expected(scorer)
    gate = ["tracking_lost", "depth_measurement_rejected"]
    for index in (0, 1):
        _fail(scorer, records, index, _stop(67, "depth_measurement_rejected"), {67: _event(ended_by=gate)})
    a2 = scorer.score_a2(records)
    observed = a2["clauses"][0]["observed"]
    assert a2["verdict"] == "partly supported"
    assert len(observed["stops_in_window"]) == 2 and observed["appearance_stops_in_window"] == []
    for index in (0, 1):
        _fail(scorer, records, index, _stop(67, APPEARANCE), {67: _event(accept=False)})
    assert scorer.score_a2(records)["verdict"] == "refuted"


# ------------------------------------------------------------------------------------------------ A3
def test_a3_cases(scorer):
    records = _expected(scorer)
    records[0]["events"][74] = _event()
    records[0]["grey_failures"] = [67, 74]
    a3 = scorer.score_a3(records)
    assert a3["verdict"] == "supported"
    (failure,) = a3["clauses"][0]["observed"]["grey_failures_in_closure"]
    assert failure["frame"] == 74 and failure["J_clear"] and failure["agreement_confidence"] == 0.4
    records = _expected(scorer)  # rejected on J at a stop during closure: the expectation fails but cannot refute
    records[1] = _record(
        scorer,
        EVENING[1],
        "stopped_vision_invalid",
        _stop(74, APPEARANCE),
        {67: _event(), 74: _event(accept=False, failed=("J_jaw_silhouette",))},
        closure=_closure(71, 74),
    )
    a3 = scorer.score_a3(records)
    assert a3["verdict"] == "partly supported" and not a3["clauses"][0]["refutes"]
    assert a3["clauses"][0]["observed"]["not_accepted_with_j_clear"] == [f"{scorer.name(records[1])}@74"]
    records = _expected(scorer)  # accepted but J not passed (or missing) is not 'J clear'
    records[0]["events"][75] = {**_event(), "jaw_condition": None}
    records[0]["grey_failures"] = [67, 75]
    assert scorer.score_a3(records)["verdict"] == "partly supported"


def test_a3_without_closure_failures_holds_and_without_closure_is_untested(scorer):
    records = _expected(scorer)
    a3 = scorer.score_a3(records)
    observed = a3["clauses"][0]["observed"]["runs"][scorer.name(records[0])]
    assert a3["verdict"] == "supported" and observed["grey_failures_in_closure"].startswith("none")
    assert [item["patch_correlation"] for item in observed["closure_frames"]] == [0.9] * 7
    for index in range(4):
        _fail(scorer, records, index, _stop(67, "low_confidence"), {67: _event(ended_by=LOW)})
    assert scorer.score_a3(records)["verdict"] == "untested"  # no run reached closure
    records = _expected(scorer)
    _unscore(records[0])
    assert scorer.score_a3(records)["verdict"] == "partly supported"


def _fact(index, phase, event=None, state="tracking", reason=None):
    return {
        "index": index,
        "cut_phase": phase,
        "state": state,
        "reason": reason,
        "patch_correlation": 0.5,
        "closure_progress": 0.0,
        "event": event,
    }


def test_closure_frames_keep_a_stop_during_closure_and_the_completing_frame(scorer):
    phases = ["approach", "align", "closing", "closing", "closing", "stopped", "stopped"]
    facts = [_fact(70 + i, phase) for i, phase in enumerate(phases)]
    assert [c["frame"] for c in scorer.closure_frames(facts, 75)] == [72, 73, 74, 75]
    phases = ["align", "closing", "closing", "closing", "closing", "closing", "closing", "retreat", "retreat"]
    facts = [_fact(70 + i, phase) for i, phase in enumerate(phases)]
    closure = scorer.closure_frames(facts, None)
    assert [c["frame"] for c in closure] == list(range(71, 78))
    assert closure[-1]["cut_phase"] == "retreat" and closure[-1]["previous_cut_phase"] == "closing"


def test_a3_a_closure_event_rejected_with_j_passing_is_not_clear(scorer):
    records = _expected(scorer)
    records[0]["events"][74] = _event(accept=False, failed=("4_median_abs",), jaw_passed=True)
    records[0]["grey_failures"] = [67, 74]
    a3 = scorer.score_a3(records)
    observed = a3["clauses"][0]["observed"]
    assert a3["verdict"] == "partly supported" and not a3["clauses"][0]["refutes"]
    assert observed["not_accepted_with_j_clear"] == [f"{scorer.name(records[0])}@74"]
    assert observed["grey_failures_in_closure"][0]["J_clear"] is True


def test_a3_is_supported_only_when_all_four_runs_reached_closure(scorer):
    records = _expected(scorer)
    gate = ["tracking_lost", "depth_measurement_rejected"]
    _fail(scorer, records, 0, _stop(67, "depth_measurement_rejected"), {67: _event(ended_by=gate)})
    a3 = scorer.score_a3(records)
    assert a3["verdict"] == "partly supported" and a3["clauses"][0]["tested"] == 3
    assert a3["clauses"][0]["observed"]["runs_that_reached_closure"] == "3 of 4"


# ------------------------------------------------------------------------------------------------ A4, A5
def test_a4_refutations_and_mechanism_expectations(scorer):
    records = _expected(scorer)
    records[-1].update(outcome="pass", stop=None)
    assert scorer.score_a4(records)["verdict"] == "refuted"
    records = _expected(scorer)
    records[-2]["events"][20] = _event()
    records[-2]["grey_failures"] = [20, 70]
    assert scorer.score_a4(records)["verdict"] == "refuted"
    records = _expected(scorer)
    records[-1]["events"], records[-1]["grey_failures"] = {}, []
    assert scorer.score_a4(records)["verdict"] == "partly supported"
    records = _expected(scorer)
    records[-2]["events"][70] = _event(accept=False, failed=("4_median_abs",))  # 19444 stopped, not on the jaw
    a4 = scorer.score_a4(records)
    assert a4["verdict"] == "partly supported" and not any(c["refutes"] for c in a4["clauses"])
    records = _expected(scorer)
    records[-1]["events"][29] = _event(accept=False, failed=("4_median_abs",))  # 12142 stopped, not on the wire
    assert scorer.score_a4(records)["verdict"] == "partly supported"
    records = _expected(scorer)
    for index in (-1, -2):
        _unscore(records[index])
    assert scorer.score_a4(records)["verdict"] == "untested"


def test_a5_refutes_only_on_a_difference(scorer):
    records = _expected(scorer)
    comparisons = _equal_comparisons(scorer, records)
    comparisons[scorer.name(records[3])] = {"status": "mismatch", "problems": [{"kind": "event"}]}
    assert scorer.score_a5(records, comparisons)["verdict"] == "refuted"
    comparisons = _equal_comparisons(scorer, records)
    comparisons[scorer.name(records[3])] = {"status": "not_comparable", "why": "recording not complete"}
    a5 = scorer.score_a5(records, comparisons)
    assert a5["verdict"] == "partly supported"
    assert a5["clauses"][0]["observed"]["not_comparable"] == {scorer.name(records[3]): "recording not complete"}


def test_event_differences_compare_keys_both_ways(scorer):
    class Exact:
        @staticmethod
        def equal(a, b):
            return a == b

    offline = {"frame": 67, "accept": True, "agreement_confidence": 0.4, "recorded": {}}
    context = ("frame", "recorded")
    live = {"accept": True, "agreement_confidence": 0.4}
    assert scorer.event_differences(offline, live, Exact, context) == []
    assert scorer.event_differences(offline, {**live, "extra": 1}, Exact, context) == ["live only: extra"]
    assert scorer.event_differences(offline, {"accept": True}, Exact, context) == ["offline only: agreement_confidence"]
    assert scorer.event_differences(offline, {**live, "accept": False}, Exact, context) == ["accept"]
    assert scorer.event_differences(offline, None, Exact, ()) == ["no live event"]


def test_a4_an_unscored_pass_does_not_refute_and_one_control_with_failures_is_not_enough(scorer):
    records = _expected(scorer)
    _unscore(records[-1], outcome="pass")
    a4 = scorer.score_a4(records)
    assert a4["verdict"] == "partly supported" and a4["clauses"][0]["observed"]["passed"] == []
    records = _expected(scorer)
    records[-1]["events"], records[-1]["grey_failures"] = {}, []
    second = scorer.score_a4(records)["clauses"][1]
    assert not second["holds"] and not second["refutes"] and second["tested"] == 1
    assert second["observed"]["untested_no_grey_failure"] == [scorer.name(records[-1])]


LIVE_EVENT = {"accept": True, "arm": "agreement", "ncc": 0.1, "agreement_confidence": 0.4, "stats": {"n_verified": 9}}


def _offline_event(arm="agreement", **changes):
    event = {key: value for key, value in LIVE_EVENT.items() if key != "stats"}
    event.update(LIVE_EVENT["stats"], arm=arm, frame=1, recorded={}, replayed={}, recorded_cut_phase="approach")
    event.update(after_recorded_detach=False, **changes)
    return event


def _offline(scorer, tmp_path, monkeypatch, agreement_equal=True, strict_equal=True, agreement_event=None):
    """offline_comparison on a two-frame recording, with the replay tool's result faked (no images needed)."""
    import replay_depth_appearance

    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"stage": "complete", "initial_live_vision": {"measurement": {}}}))
    frames = [
        {"index": 0, "live_vision": {"measurement": {"state": "tracking"}}},
        {"index": 1, "live_vision": {"measurement": {"state": "tracking", "depth_appearance": LIVE_EVENT}}},
    ]
    (run / "frames.json").write_text(json.dumps({"frames": frames}))

    def divergence(equal, difference):
        return {
            "equals_recording": equal,
            "mismatch_frames": [] if equal else [1],
            "preview_mismatch_fields": [],
            "max_abs_patch_correlation_difference": difference,
        }

    result = {
        "divergence": {"agreement": divergence(agreement_equal, 1e-7), "strict": divergence(strict_equal, 0.5)},
        "events": {
            "agreement": [agreement_event or _offline_event()],
            "strict": [_offline_event(arm="strict", agreement_confidence=0.0)],
        },
        "replayed_through_frame": 1,
        "source_sha256": {"report.json": "r", "frames.json": "f"},
    }
    monkeypatch.setattr(replay_depth_appearance, "replay_run", lambda path: result)
    return scorer.offline_comparison({"path": str(run)})


def test_a5_compares_the_agreement_arm_only(scorer, tmp_path, monkeypatch):
    equal = _offline(scorer, tmp_path / "a", monkeypatch, strict_equal=False)  # the strict arm may diverge
    assert equal["status"] == "equal" and equal["max_agreement_correlation_difference"] == 1e-7
    differs = _offline(scorer, tmp_path / "b", monkeypatch, agreement_equal=False)
    assert differs["status"] == "mismatch"
    assert differs["problems"] == [{"kind": "agreement_differs_from_recording", "frames": [1], "preview_fields": []}]


def test_a5_an_event_value_difference_is_a_mismatch(scorer, tmp_path, monkeypatch):
    result = _offline(scorer, tmp_path, monkeypatch, agreement_event=_offline_event(agreement_confidence=0.3))
    assert result["status"] == "mismatch"
    assert result["problems"] == [{"kind": "event", "frame": "1", "fields": ["agreement_confidence"]}]


def test_a5_non_contiguous_frames_are_not_comparable(scorer, tmp_path):
    frame = {"visual_jaw_closure_progress": 0.0, "live_vision": {"measurement": {"state": "tracking"}, "cut": {}}}
    frames = [{**frame, "index": 0}, {**frame, "index": 2}]
    _captured_run(tmp_path, scorer, EVENING[0], "run_gap", frames=frames)
    record = scorer.run_record(tmp_path, EVENING[0], _row("run_gap"))
    assert record["scored"] and scorer.comparability(record) == "frame indexes are not contiguous"


# ------------------------------------------------------------------------------------------------ registry
def test_the_registered_run_directories_are_the_runners_run_labels(scorer):
    import run_vision_experiment

    assert len(scorer.REGISTRY) == 6 and scorer.EXPECTED_RUNS == 6
    for registered in scorer.REGISTRY.values():
        row = {
            "daylight": registered["daylight"],
            "target_tree_index": registered["target_tree_index"],
            "component_first_vertex": registered["target"],
            "strategy": {"name": registered["strategy"]},
        }
        assert run_vision_experiment.run_label(0, row) == registered["run_directory"]
    assert [b for b, r in scorer.REGISTRY.items() if r["kind"] == "evening"] == EVENING


def _grade_files(scorer, root, status="incomplete", checks_total=0, edit=None):
    for batch, registered in scorer.REGISTRY.items():
        row = {
            "run_directory": registered["run_directory"],
            "status": status,
            "outcome": status,
            "checks_passed": 0,
            "checks_total": checks_total,
            "failed_checks": "",
            "daylight": registered["daylight"],
            "target_tree_index": registered["target_tree_index"],
            "component_first_vertex": registered["target"],
            "strategy": registered["strategy"],
        }
        document = {
            "schema_version": 1,
            "protocol": scorer.PROTOCOL,
            "grader": scorer.GRADER,
            "batches": [{"condition": batch, "batch_dir": f"artifacts/vision_robustness/{batch}"}],
            "runs": [row],
        }
        if edit is not None:
            edit(batch, document)
        path = root / scorer.EVIDENCE_DIR / f"{batch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))


def test_load_keeps_ungraded_runs_and_scores_them_untested(scorer, tmp_path):
    _grade_files(scorer, tmp_path)
    records, files = scorer.load(tmp_path)
    assert len(records) == 6 and len(files) == 6
    assert not any(r["scored"] for r in records) and all("not graded" in r["unscored_reason"] for r in records)
    comparisons = {scorer.name(r): {"status": "not_comparable", "why": r["unscored_reason"]} for r in records}
    assert set(_verdicts(scorer, records, comparisons).values()) == {"untested"}


def _first_evening(edit):
    return lambda batch, doc: edit(batch, doc) if batch == EVENING[0] else None


@pytest.mark.parametrize(
    "edit",
    [
        lambda batch, doc: doc.update(protocol="docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md"),
        lambda batch, doc: doc.update(grader="other"),
        lambda batch, doc: doc.update(schema_version=2),
        lambda batch, doc: doc["batches"].append(dict(doc["batches"][0])),
        lambda batch, doc: doc["batches"][0].update(batch_dir="artifacts/vision_robustness/elsewhere/"),
        lambda batch, doc: doc["batches"][0].update(batch_dir=f"/some/other/checkout/{batch}"),
        _first_evening(lambda batch, doc: doc["batches"][0].update(batch_dir=f"artifacts/vision_robustness/{batch}x")),
        _first_evening(
            lambda batch, doc: doc["runs"][0].update(
                run_directory="run_00_evening_tree1_v14944_baseline_depth_appearance"
            )
        ),
        _first_evening(lambda batch, doc: doc["runs"][0].update(target_tree_index=0)),
        _first_evening(lambda batch, doc: doc["runs"][0].update(strategy="baseline_depth_appearance")),
        _first_evening(lambda batch, doc: doc["runs"][0].update(daylight="source")),
        _first_evening(lambda batch, doc: doc["runs"][0].update(component_first_vertex=15004)),
        _first_evening(lambda batch, doc: doc["runs"][0].pop("component_first_vertex")),
        _first_evening(lambda batch, doc: doc["runs"].append(dict(doc["runs"][0]))),
        _first_evening(lambda batch, doc: doc.update(runs=[])),
    ],
)
def test_load_refuses_a_grade_file_that_is_not_the_registered_batch(scorer, tmp_path, edit):
    _grade_files(scorer, tmp_path, edit=edit)
    with pytest.raises(scorer.InputError):
        scorer.load(tmp_path)


def test_load_refuses_a_missing_file_another_check_count_an_unregistered_file_and_malformed_json(scorer, tmp_path):
    with pytest.raises(scorer.InputError, match="missing"):
        scorer.load(tmp_path)
    _grade_files(scorer, tmp_path)
    (tmp_path / scorer.EVIDENCE_DIR / f"{CTL_12142}.json").unlink()
    with pytest.raises(scorer.InputError, match="missing"):
        scorer.load(tmp_path)
    _grade_files(scorer, tmp_path, status="graded", checks_total=16)
    with pytest.raises(scorer.InputError, match="16 checks"):
        scorer.load(tmp_path)
    for extra in ("agree-eve-r1-20261003-regrade.json", "agree-eve-r5-20261003.json", "notes.json"):
        _grade_files(scorer, tmp_path)
        (tmp_path / scorer.EVIDENCE_DIR / extra).write_text("{}")
        with pytest.raises(scorer.InputError, match="outside the registered"):
            scorer.load(tmp_path)
        (tmp_path / scorer.EVIDENCE_DIR / extra).unlink()
    (tmp_path / scorer.EVIDENCE_DIR / f"{EVENING[2]}.json").write_text("{not json")
    with pytest.raises(scorer.InputError, match="malformed"):
        scorer.load(tmp_path)


# ------------------------------------------------------------------------------------------------ captured runs
def _captured_run(
    root,
    scorer,
    batch,
    run_directory,
    arm="agreement",
    enabled=True,
    accept=True,
    stage="complete",
    result=True,
    jaw_self_mask=False,
    closure_hold=False,
    frames=None,
    constants=None,
):
    """A recorded run; result='runner error' writes the runner's result without a configuration check."""
    run = root / scorer.VISION_ROBUSTNESS / batch / run_directory
    (run / "frames").mkdir(parents=True)
    from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance

    report = {
        "stage": stage,
        "node": "cn-gpu5",
        "depth_appearance": {
            "enabled": enabled,
            "constants": {**registered_depth_appearance(arm), **(constants or {})},
        },
        "jaw_self_mask": {"enabled": jaw_self_mask},
        "closure_hold": {"enabled": closure_hold},
        "initial_live_vision": {"measurement": {"state": "tracking"}},
    }
    (run / "report.json").write_text(json.dumps(report))
    if frames is None:
        frames = [
            {
                "index": 0,
                "visual_jaw_closure_progress": 0.0,
                "live_vision": {
                    "measurement": {"state": "tracking", "depth_appearance": {"accept": accept, "ncc": 0.2}},
                    "cut": {"phase": "approach"},
                },
            }
        ]
    (run / "frames.json").write_text(json.dumps({"frames": frames}))
    if result == "runner error":
        (run / "experiment_result.json").write_text(json.dumps({"ok": False, "traceback": "ValueError"}))
    elif result is not None:
        (run / "experiment_result.json").write_text(json.dumps({"configuration_matches": result}))
    return run


def _row(run_directory):
    return {
        "run_directory": run_directory,
        "status": "graded",
        "checks_total": 17,
        "checks_passed": 9,
        "failed_checks": "no_recorded_stops,returned_home_and_final_phase_complete",
        "outcome": "stopped_vision_invalid",
        "daylight": "evening",
        "target_tree_index": 1,
        "component_first_vertex": 14944,
        "strategy": "baseline_depth_agreement",
    }


def test_only_a_complete_capture_of_the_registered_agreement_arm_is_scored(scorer, tmp_path):
    batch = EVENING[0]
    run = _captured_run(tmp_path, scorer, batch, "run_good")
    record = scorer.run_record(tmp_path, batch, _row("run_good"))
    assert record["scored"] and record["grey_failures"] == [0] and record["configuration"]["arm_agreement"]
    assert record["failed_checks"] == ["no_recorded_stops", "returned_home_and_final_phase_complete"]
    assert "preview files missing" in scorer.comparability(record)
    (run / "preview_wrist.png").write_bytes(b"")
    (run / "preview_depth.npy").write_bytes(b"")
    assert "frame files missing" in scorer.comparability(record)
    for directory, options, failed in (
        ("run_strict", {"arm": "strict"}, "constants_registered"),
        ("run_constant", {"constants": {"max_median_abs_m": 0.004}}, "constants_registered"),
        ("run_off", {"enabled": False}, "enabled"),
        ("run_mask", {"jaw_self_mask": True}, "jaw_arms_off"),
        ("run_hold", {"closure_hold": True}, "jaw_arms_off"),
        ("run_unmatched", {"result": False}, "configuration_matches"),
    ):
        _captured_run(tmp_path, scorer, batch, directory, **options)
        record = scorer.run_record(tmp_path, batch, _row(directory))
        assert record["captured"] and not record["scored"], directory
        assert record["unscored_reason"] == "configuration refused" and record["configuration"][failed] is not True
        assert scorer.comparability(record) == "configuration refused"
        if directory == "run_constant":
            assert record["configuration"]["arm_agreement"]  # only the constants differ
    killed = _captured_run(tmp_path, scorer, batch, "run_killed", stage="record", result=None)
    record = scorer.run_record(tmp_path, batch, _row(killed.name))
    assert not record["scored"] and "capture incomplete" in record["unscored_reason"]
    assert "capture incomplete" in scorer.comparability(record)
    crashed = _captured_run(tmp_path, scorer, batch, "run_crashed", stage="record")  # matching result, stage not done
    record = scorer.run_record(tmp_path, batch, _row(crashed.name))
    assert not record["scored"] and "capture incomplete" in record["unscored_reason"]
    errored = _captured_run(tmp_path, scorer, batch, "run_runner_error", result="runner error")
    record = scorer.run_record(tmp_path, batch, _row(errored.name))
    assert not record["scored"] and record["unscored_reason"] == scorer.RUNNER_ERROR
    assert (
        record["configuration"]["experiment_result_recorded"] and not record["configuration"]["configuration_checked"]
    )
    _captured_run(tmp_path, scorer, batch, "run_bad", accept=None)
    with pytest.raises(scorer.InputError, match="boolean"):
        scorer.run_record(tmp_path, batch, _row("run_bad"))
    record = scorer.run_record(tmp_path, batch, {**_row("run_missing"), "status": "incomplete", "checks_total": 0})
    assert not record["captured"] and "not graded" in record["unscored_reason"]


def test_the_registered_constants_must_name_the_agreement_arm(scorer, monkeypatch):
    from isaaclab_pruning.perception import depth_appearance

    assert scorer.registered_constants()["arm"] == "agreement"
    monkeypatch.setattr(depth_appearance, "registered_depth_appearance", lambda arm="strict": {"arm": "strict"})
    with pytest.raises(scorer.InputError, match="agreement"):
        scorer.registered_constants()


def test_run_record_judges_failures_through_the_stop_and_finds_closure(scorer, tmp_path):
    def frame(index, phase, event=None, state="tracking", reason=None):
        measurement = {"state": state, "reason": reason, "patch_correlation": 0.3 if event else 0.9}
        if event is not None:
            measurement["depth_appearance"] = event
        return {
            "index": index,
            "visual_jaw_closure_progress": 0.0,
            "live_vision": {"measurement": measurement, "cut": {"phase": phase, "stopped_reason": None}},
        }

    accepted = {"accept": True, "conditions": {"J_jaw_silhouette": {"passed": True}}, "agreement_confidence": 0.5}
    rejected = {"accept": False, "conditions": {"J_jaw_silhouette": {"passed": False}}, "failed_conditions": ["J"]}
    frames = [
        frame(0, "approach"),
        frame(1, "align", accepted),
        frame(2, "closing"),
        frame(3, "stopped", rejected, "tracking_lost", APPEARANCE),
        frame(4, "stopped", rejected, "tracking_lost", "explicit_initialization_required"),
    ]
    _captured_run(tmp_path, scorer, EVENING[0], "run_closure", frames=frames)
    record = scorer.run_record(tmp_path, EVENING[0], _row("run_closure"))
    assert record["stop"] == {
        "frame": 3,
        "stopped_reason": None,
        "tracker_state": "tracking_lost",
        "tracker_reason": APPEARANCE,
    }
    assert record["grey_failures"] == [1, 3] and record["post_stop_grey_failures"] == [4]
    assert [c["frame"] for c in record["closure_frames"]] == [2, 3]
    assert record["events"][3]["jaw_condition"] == {"passed": False}
    assert record["events"][1]["agreement_confidence"] == 0.5


# ------------------------------------------------------------------------------------------------ frozen code, output
def _plans(scorer, tmp_path, hashes, edit=None):
    for batch in scorer.REGISTRY:
        plan = tmp_path / "vision_robustness" / batch / "plan.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        frozen = dict(hashes)
        if edit is not None:
            edit(batch, frozen)
        plan.write_text(json.dumps({"source_sha256": frozen}))


def test_the_frozen_source_check_compares_every_batch_plan_and_records_the_scorer(scorer, tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "VISION_ROBUSTNESS", str(tmp_path / "vision_robustness"))
    paths = scorer.frozen_paths(ROOT)
    assert scorer.SELF not in paths
    assert set(scorer.FROZEN_EXTRA) <= set(paths)
    assert "tools/replay_depth_appearance.py" in paths
    assert "source/isaaclab_pruning/isaaclab_pruning/perception/depth_appearance.py" in paths
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    own = hashlib.sha256((ROOT / scorer.SELF).read_bytes()).hexdigest()
    with pytest.raises(scorer.InputError, match="unreadable"):
        scorer.check_frozen_sources(ROOT)
    _plans(scorer, tmp_path, hashes)  # the plans did not freeze the scorer
    with pytest.raises(scorer.InputError, match="does not contain tools/score_agreement_loop.py"):
        scorer.check_frozen_sources(ROOT)
    _plans(scorer, tmp_path, {**hashes, scorer.SELF: own})
    current, frozen_scorer = scorer.check_frozen_sources(ROOT)
    assert current == hashes and frozen_scorer["current_sha256"] == own
    assert all(not item["amended_since_freeze"] for item in frozen_scorer["batches"].values())
    assert sorted(frozen_scorer["batches"]) == sorted(scorer.REGISTRY)
    _plans(scorer, tmp_path, {**hashes, scorer.SELF: "1" * 64})  # an amendment after the freeze: reported only
    _, frozen_scorer = scorer.check_frozen_sources(ROOT)
    assert all(item["amended_since_freeze"] for item in frozen_scorer["batches"].values())
    assert frozen_scorer["batches"][CTL_12142]["frozen_sha256"] == "1" * 64

    def tamper(batch, frozen):
        if batch == CTL_19444:
            frozen["tools/replay_depth_appearance.py"] = "0" * 64

    _plans(scorer, tmp_path, {**hashes, scorer.SELF: own}, edit=tamper)
    with pytest.raises(scorer.InputError, match="replay_depth_appearance"):
        scorer.check_frozen_sources(ROOT)


def test_a_hashing_error_is_a_refusal_not_a_traceback(scorer, tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "frozen_paths", lambda root: ["tools/no_such_module.py"])
    with pytest.raises(scorer.InputError, match="cannot hash"):
        scorer.check_frozen_sources(tmp_path)


def _git(repo, *args):
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        check=True,
        capture_output=True,
    )


def test_the_scorer_must_be_tracked_and_equal_to_head(scorer, tmp_path):
    repo = tmp_path / "checkout"
    scorer_file = repo / scorer.SELF
    scorer_file.parent.mkdir(parents=True)
    with pytest.raises(scorer.InputError):  # not a repository at all
        scorer.verify_self(repo)
    _git(repo, "init", "-q")
    scorer_file.write_text("v1\n")
    (repo / "README").write_text("x\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "base")
    with pytest.raises(scorer.InputError, match="not tracked"):
        scorer.verify_self(repo)
    _git(repo, "add", scorer.SELF)  # staged but not committed: differs from HEAD
    with pytest.raises(scorer.InputError, match="differs from HEAD"):
        scorer.verify_self(repo)
    _git(repo, "commit", "-q", "-m", "scorer")
    assert scorer.verify_self(repo) == hashlib.sha256(b"v1\n").hexdigest()
    scorer_file.write_text("v2\n")
    with pytest.raises(scorer.InputError, match="differs from HEAD"):
        scorer.verify_self(repo)


def test_only_the_scorers_own_checkout_is_scored(scorer, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        scorer.main(["--root", str(tmp_path)])
    assert stopped.value.code == 2 and "own checkout" in capsys.readouterr().err


def test_an_existing_output_is_never_overwritten(scorer, tmp_path):
    output = tmp_path / "verdicts.json"
    output.write_text("kept")
    with pytest.raises(SystemExit):
        scorer.main(["--root", str(tmp_path), "--output", str(output)])
    assert output.read_text() == "kept"


def test_the_construction_notes_record_a_blind_scorer(scorer):
    text = " ".join(scorer.CONSTRUCTION_NOTES)
    for fact in (
        "committed before submission",
        "before any agreement-loop run was submitted",
        "no design-study or C0 evidence in docs/evidence",
        "amended after commits 6cd4e75 and 2cbdd93, following an independent blind review",
        "did not read those two commits' messages",
        "did not open the three evidence files",
        "it is written blind",
        "agreement_design_replay_2026-10-03.json",
        "agreement_design_closure_shadow_2026-10-03.json",
        "the C0 gate (A0",
        "No clause reads them",
        "exits 0 only when the capture completed, the unchanged grader passed all 17 checks",
        "Slurm state",
        "nothing seen afterwards, Slurm states included, can shape it",
        "depth-loop-*-20261001 recordings",
    ):
        assert fact in text
    interpretations = " ".join(scorer.INTERPRETATIONS)
    assert "registered_depth_appearance('agreement')" in interpretations
    assert "the previous frame's is" in interpretations
    assert "all 4 reached closure" in interpretations and "runner error: no configuration check" in interpretations
