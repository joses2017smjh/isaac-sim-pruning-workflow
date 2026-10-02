"""Protect the C1-C6 scoring of the depth-aware closed-loop test: registered inputs, bounds and untested cases."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scorer(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_depth_loop", tools / "score_depth_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(accept=True, confidence=0.2, failed=()):
    return {
        "accept": accept,
        "strict_confidence": confidence if accept else None,
        "failed_conditions": list(failed),
        "decision": "continues" if accept else "loss_stands",
    }


def _record(batch, target, outcome="pass", stop=None, events=None, scored=True):
    events = events or {}
    kind = (
        "control"
        if "-ctl-" in batch
        else "source"
        if "-src-" in batch
        else "evening"
        if "-eve-" in batch
        else "morning"
    )
    return {
        "batch": batch,
        "kind": kind,
        "run_directory": f"run_{target}",
        "target": target,
        "daylight": kind,
        "outcome": outcome,
        "checks": "17/17" if outcome == "pass" else "9/17",
        "scored": scored,
        "unscored_reason": None if scored else "not graded (incomplete, checks 0/0)",
        "stop": stop,
        "grey_failures": sorted(events),
        "events": events,
        "post_stop_events": {},
    }


def _stop(frame, reason):
    return {"frame": frame, "stopped_reason": "vision_invalid", "tracker_reason": reason}


APPEARANCE = "appearance_changed_or_occluded"


def _expected():
    """The registered expectation: 15004 continues and passes, evening 14944 stops at 67 on low confidence."""
    records = []
    for r in (1, 2):
        records.append(
            _record(
                f"depth-loop-eve-r{r}-20261001",
                14944,
                "stopped_vision_invalid",
                _stop(67, "low_confidence"),
                {67: _event(confidence=0.03)},
            )
        )
        records.append(_record(f"depth-loop-eve-r{r}-20261001", 15004, events={67: _event(confidence=0.25)}))
        records.append(_record(f"depth-loop-mor-r{r}-20261001", 15004, events={75: _event(confidence=0.2)}))
    records += [_record("depth-loop-src-20261001", 14944), _record("depth-loop-src-20261001", 15004)]
    records.append(
        _record(
            "depth-loop-ctl-19444-20261001",
            19444,
            "stopped_vision_invalid",
            _stop(68, APPEARANCE),
            {68: _event(accept=False, failed=("3_near_fraction", "J_jaw_silhouette"))},
        )
    )
    records.append(
        _record(
            "depth-loop-ctl-12142-20261001",
            12142,
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


def test_the_registered_expectation_supports_every_prediction(scorer):
    assert _verdicts(scorer, _expected()) == {f"C{i}": "supported" for i in range(1, 7)}


def test_a_prediction_with_no_tested_run_is_untested_not_partly_supported(scorer):
    unscored = [{**r, "scored": False, "unscored_reason": "not graded"} for r in _expected()]
    comparisons = {scorer.name(r): {"status": "not_comparable", "why": "not graded"} for r in unscored}
    assert _verdicts(scorer, unscored, comparisons) == {f"C{i}": "untested" for i in range(1, 7)}


def test_c1_is_refuted_by_a_rejected_shadow_event_and_untested_without_one(scorer):
    records = _expected()
    records[2]["events"] = {74: _event(accept=False), 75: _event()}
    records[2]["grey_failures"] = [74, 75]
    assert scorer.score_c1(records)["verdict"] == "refuted"
    records = _expected()
    records[1]["events"], records[1]["grey_failures"] = {}, []
    c1 = scorer.score_c1(records)
    assert (
        c1["verdict"] == "partly supported"
        and c1["clauses"][0]["observed"]["first_in_window"][scorer.name(records[1])]["stop"] is None
    )
    records = _expected()
    records[1]["events"], records[1]["grey_failures"] = {80: _event(accept=False)}, [80]  # outside the window
    assert scorer.score_c1(records)["verdict"] == "partly supported"


def test_c2_bounds_and_unscored_runs(scorer):
    records = _expected()
    records[1].update(outcome="stopped_vision_invalid", stop=_stop(67, "low_confidence"))
    assert scorer.score_c2(records)["verdict"] == "supported"  # 3 of 4 pass, the stop is not on appearance
    records[4].update(outcome="stopped_vision_invalid", stop=_stop(67, "low_confidence"))
    assert scorer.score_c2(records)["verdict"] == "refuted"  # two stops in the window
    records = _expected()
    records[2].update(outcome="stopped_vision_invalid", stop=_stop(75, APPEARANCE))
    assert scorer.score_c2(records)["verdict"] == "partly supported"  # 3 pass but one appearance stop
    records = _expected()
    for index in (1, 2, 4):
        records[index].update(outcome="stopped_vision_invalid", stop=_stop(150, "hazard_contact"))
    assert scorer.score_c2(records)["verdict"] == "refuted"  # fewer than 2 pass, every run scored
    records = _expected()
    for index in (1, 2, 4):
        records[index].update(scored=False, unscored_reason="not graded", outcome="infrastructure")
    assert scorer.score_c2(records)["verdict"] == "partly supported"  # unscored runs are not failures
    records[5].update(outcome="stopped_vision_invalid", stop=_stop(150, "hazard_contact"))
    assert scorer.score_c2(records)["verdict"] == "partly supported"  # 0 + 3 unscored could still reach 2


def test_c3_cases(scorer):
    records = _expected()
    records[0].update(outcome="pass", stop=None)
    c3 = scorer.score_c3(records)
    assert c3["verdict"] == "refuted" and "continued" in c3["clauses"][0]["observed"][scorer.name(records[0])]["case"]
    records = _expected()
    records[0]["events"] = {67: _event(accept=False)}
    records[0]["stop"] = _stop(67, APPEARANCE)
    assert scorer.score_c3(records)["verdict"] == "refuted"
    records = _expected()
    records[0]["events"], records[0]["grey_failures"] = {}, []
    records[0]["stop"] = _stop(40, "hazard_contact")
    assert scorer.score_c3(records)["verdict"] == "partly supported"
    records = _expected()
    records[0]["stop"] = _stop(67, "depth_measurement_rejected")
    c3 = scorer.score_c3(records)
    assert c3["verdict"] == "partly supported"
    assert (
        c3["clauses"][0]["observed"][scorer.name(records[0])]["case"]
        == "stopped at the event on depth_measurement_rejected"
    )


def test_c4_refutations_and_mechanism_expectations(scorer):
    records = _expected()
    records[-1].update(outcome="pass", stop=None)
    assert scorer.score_c4(records)["verdict"] == "refuted"
    records = _expected()
    records[-2]["events"][20] = _event()
    records[-2]["grey_failures"] = [20, 68]
    assert scorer.score_c4(records)["verdict"] == "refuted"
    records = _expected()
    records[-1]["events"], records[-1]["grey_failures"] = {}, []
    assert scorer.score_c4(records)["verdict"] == "partly supported"
    records = _expected()
    records[-2]["events"][68] = _event(accept=False, failed=("4_median_abs",))  # 19444 stopped, not on the jaw
    c4 = scorer.score_c4(records)
    assert c4["verdict"] == "partly supported" and not any(c["refutes"] for c in c4["clauses"])
    records = _expected()
    for index in (-1, -2):
        records[index].update(scored=False, unscored_reason="not graded", outcome="incomplete")
    assert scorer.score_c4(records)["verdict"] == "untested"  # controls that never ran keep no stop


def test_c5_unscored_is_untested_and_a_failed_check_refutes(scorer):
    records = _expected()
    records[6].update(outcome="stopped_vision_invalid")
    assert scorer.score_c5(records)["verdict"] == "refuted"
    records = _expected()
    records[6].update(scored=False, unscored_reason="not graded", outcome="infrastructure")
    assert scorer.score_c5(records)["verdict"] == "partly supported"


def test_c6_refutes_only_on_a_difference(scorer):
    records = _expected()
    comparisons = _equal_comparisons(scorer, records)
    comparisons[scorer.name(records[3])] = {"status": "mismatch", "problems": [{"kind": "event"}]}
    assert scorer.score_c6(records, comparisons)["verdict"] == "refuted"
    comparisons = _equal_comparisons(scorer, records)
    comparisons[scorer.name(records[3])] = {"status": "not_comparable", "why": "recording not complete"}
    assert scorer.score_c6(records, comparisons)["verdict"] == "partly supported"


def test_event_differences_compare_keys_both_ways(scorer):
    class Exact:
        @staticmethod
        def equal(a, b):
            return a == b

    offline = {"frame": 67, "accept": True, "ncc": 0.2, "recorded": {}}
    assert scorer.event_differences(offline, {"accept": True, "ncc": 0.2}, Exact, ("frame", "recorded")) == []
    assert scorer.event_differences(
        offline, {"accept": True, "ncc": 0.2, "extra": 1}, Exact, ("frame", "recorded")
    ) == ["live only: extra"]
    assert scorer.event_differences(offline, {"accept": True}, Exact, ("frame", "recorded")) == ["offline only: ncc"]
    assert scorer.event_differences(offline, None, Exact, ()) == ["no live event"]


def _grade_files(scorer, root, status="incomplete", checks_total=0, edit=None):
    for batch, (kind, strategy, targets) in scorer.REGISTRY.items():
        rows = [
            {
                "run_directory": f"run_{i:02d}_{scorer.DAYLIGHT[kind]}_v{t}",
                "status": status,
                "outcome": status,
                "checks_passed": 0,
                "checks_total": checks_total,
                "daylight": scorer.DAYLIGHT[kind],
                "component_first_vertex": t,
                "strategy": strategy,
            }
            for i, t in enumerate(targets)
        ]
        document = {
            "schema_version": 1,
            "protocol": scorer.PROTOCOL,
            "grader": scorer.GRADER,
            "batches": [{"condition": batch, "batch_dir": f"artifacts/vision_robustness/{batch}"}],
            "runs": rows,
        }
        if edit is not None:
            edit(batch, document)
        path = root / scorer.EVIDENCE_DIR / f"{batch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))


def test_load_keeps_ungraded_runs_and_scores_them_untested(scorer, tmp_path):
    _grade_files(scorer, tmp_path)
    records, files = scorer.load(tmp_path)
    assert len(records) == 10 and len(files) == 7
    assert not any(r["scored"] for r in records) and all("not graded" in r["unscored_reason"] for r in records)
    comparisons = {scorer.name(r): {"status": "not_comparable", "why": r["unscored_reason"]} for r in records}
    assert set(_verdicts(scorer, records, comparisons).values()) == {"untested"}


@pytest.mark.parametrize(
    "edit",
    [
        lambda batch, doc: doc.update(protocol="docs/EVAL_PROTOCOL_2026-09-23.md"),
        lambda batch, doc: doc.update(grader="other"),
        lambda batch, doc: doc["batches"].append(dict(doc["batches"][0])),
        lambda batch, doc: doc["batches"][0].update(batch_dir="artifacts/vision_robustness/elsewhere/"),
        lambda batch, doc: doc["runs"].reverse() if len(doc["runs"]) > 1 else None,
        lambda batch, doc: doc["runs"][0].update(strategy="baseline"),
    ],
)
def test_load_refuses_a_grade_file_that_is_not_the_registered_batch(scorer, tmp_path, edit):
    _grade_files(scorer, tmp_path, edit=edit)
    with pytest.raises(scorer.InputError):
        scorer.load(tmp_path)


def test_load_refuses_a_missing_file_and_a_grade_with_another_check_count(scorer, tmp_path):
    with pytest.raises(scorer.InputError, match="missing"):
        scorer.load(tmp_path)
    _grade_files(scorer, tmp_path, status="graded", checks_total=16)
    with pytest.raises(scorer.InputError, match="16 checks"):
        scorer.load(tmp_path)


def _captured_run(root, scorer, batch, run_directory, enabled=True, accept=True, stage="complete"):
    run = root / scorer.VISION_ROBUSTNESS / batch / run_directory
    (run / "frames").mkdir(parents=True)
    from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance

    report = {
        "stage": stage,
        "node": "cn-gpu5",
        "depth_appearance": {"enabled": enabled, "constants": registered_depth_appearance()},
        "jaw_self_mask": {"enabled": False},
        "closure_hold": {"enabled": False},
        "initial_live_vision": {"measurement": {"state": "tracking"}},
    }
    (run / "report.json").write_text(json.dumps(report))
    frame = {
        "index": 0,
        "live_vision": {
            "measurement": {"state": "tracking", "depth_appearance": {"accept": accept, "ncc": 0.2}},
            "cut": {"phase": "approach"},
        },
    }
    (run / "frames.json").write_text(json.dumps({"frames": [frame]}))
    return run


def test_a_capture_without_the_registered_arm_is_refused_and_accept_must_be_boolean(scorer, tmp_path):
    batch = "depth-loop-src-20261001"
    _grade_files(scorer, tmp_path, status="graded", checks_total=17)
    row = {"run_directory": "run_00_source_v14944", "status": "graded", "checks_total": 17, "checks_passed": 17}
    row.update(outcome="pass", daylight="source", component_first_vertex=14944)
    _captured_run(tmp_path, scorer, batch, "run_00_source_v14944", enabled=False)
    record = scorer.run_record(tmp_path, batch, row)
    assert record["captured"] and not record["scored"] and record["unscored_reason"] == "configuration refused"
    _captured_run(tmp_path, scorer, batch, "run_01_bad", accept=None)
    with pytest.raises(scorer.InputError, match="boolean"):
        scorer.run_record(tmp_path, batch, {**row, "run_directory": "run_01_bad"})
    run = _captured_run(tmp_path, scorer, batch, "run_02_good")
    record = scorer.run_record(tmp_path, batch, {**row, "run_directory": "run_02_good"})
    assert record["scored"] and record["grey_failures"] == [0]
    assert "preview files missing" in scorer.comparability(record)
    (run / "preview_wrist.png").write_bytes(b"")
    (run / "preview_depth.npy").write_bytes(b"")
    assert "frame files missing" in scorer.comparability(record)
    incomplete = _captured_run(tmp_path, scorer, batch, "run_03_killed", stage="record")
    record = scorer.run_record(tmp_path, batch, {**row, "run_directory": incomplete.name})
    assert "not complete" in scorer.comparability(record)
