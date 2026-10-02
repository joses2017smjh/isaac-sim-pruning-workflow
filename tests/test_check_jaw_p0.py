"""Protect the jaw-in-view P0 gate: it passes only on exact reproduction, the study's outputs and 530-only holds."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
RECORDED_530 = ROOT / "artifacts/vision_robustness/planned-pose-gpu-r2-20260929/run_00_source_tree0_v530_planned_pose"
NAME = "batch-a/run_00_fake"


@pytest.fixture
def checker(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("check_jaw_p0", tools / "check_jaw_p0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "EXPECTED_RUNS", 1)
    return module


def _measurement(i, state="tracking", reason=None, corr=0.9):
    return {
        "state": state,
        "reason": reason,
        "pixel_xy": [100.0 + i, 50.0],
        "patch_correlation": corr,
        "feature_count": 13,
        "target_position_world_m": [0.1, 0.2, 0.3],
    }


def _recording(tmp_path, phases, name=NAME):
    """A recorded run: one measurement per frame, cut phase per frame (a 'stopped' frame is the stop)."""
    run = tmp_path / name
    run.mkdir(parents=True)
    frames = [
        {
            "index": i,
            "live_vision": {
                "measurement": _measurement(i),
                "cut": {"phase": phase, "stopped_reason": "vision_invalid" if phase == "stopped" else None},
            },
        }
        for i, phase in enumerate(phases)
    ]
    (run / "frames.json").write_text(json.dumps({"fps": 10.0, "frames": frames}))
    report = {
        "vision_initialization": {"tracker": {"state": "initialized", "feature_count": 13}},
        "initial_live_vision": {"measurement": {"state": "tracking", "reason": None}, "cut": {"phase": "approach"}},
    }
    (run / "report.json").write_text(json.dumps(report))
    return run, frames


def _row(frame, **overrides):
    measurement = frame["live_vision"]["measurement"]
    cut = frame["live_vision"]["cut"]
    row = {
        "index": frame["index"],
        "time_s": frame["index"] / 10.0,
        **{k: measurement[k] for k in ("state", "reason", "pixel_xy", "patch_correlation", "feature_count")},
        "target_position_world_m": measurement["target_position_world_m"],
        "cut_phase": cut["phase"],
        "stopped_reason": cut["stopped_reason"],
        "detach_event": False,
        "unmasked_patch_elements": 169,
        "patch_correlation_kept": measurement["patch_correlation"],
        "masked_patch_elements_prev_cur": [0, 0],
        "mask_dropped_valid_features": 0,
        "unmasked_flow_would_pass": None,
        "flow_reason": None,
        "rendered_progress_matches_recording": True,
        "closure_hold": {"held": False, "held_target_age_s": None, "explanation": None, "dt_mm": 0.0, "dr_deg": 0.0},
        "vision_source": "live",
        "waived_checks": [],
        "held_target_age_s": None,
        "cut_without_hold": {"phase": cut["phase"], "stopped_reason": cut["stopped_reason"]},
        "measurement_matches_recording": True,
        "evaluable": True,
        "mouth_distance_m": 0.003,
    }
    row.update(overrides)
    return row


def _replay_run(run, rows, last, name=NAME, recorded_closure_start=None, **replay):
    return {
        "run": name,
        "path": str(run),
        "recorded": {"closure_start_frame": recorded_closure_start},
        "frames_detail": rows,
        "compared_through_frame": last,
        "comparison_through_recorded_stop": {"cut_mismatch_frames": []},
        "initialization": {"replayed_state": "initialized", "replayed_feature_count": 13},
        "preview": {"state": "tracking", "reason": None, "cut_phase": "approach", "matches_recording": True},
        "replay": {
            "stop_frame": None,
            "stop_reason": None,
            "detach_frame": None,
            "closure_start_frame": None,
            "held_frames": [],
            "cut_without_hold_stop_frame": None,
            "cut_without_hold_stop_reason": None,
            **replay,
        },
    }


def _document(*runs):
    return {"runs": list(runs), "summary": {"reproduce_recording_through_recorded_stop": len(runs)}}


def test_flags_off_passes_only_on_exact_reproduction_through_the_recorded_stop(checker, tmp_path):
    run, frames = _recording(tmp_path, ["approach", "approach", "stopped", "stopped"])
    rows = [_row(f) for f in frames]
    inventory = {NAME: run}
    assert checker.check_flags_off(_document(_replay_run(run, rows, 2)), inventory)["passed"]

    within = [_row(f) for f in frames]
    within[1]["patch_correlation"] += 5e-7
    assert checker.check_flags_off(_document(_replay_run(run, within, 2)), inventory)["passed"]

    beyond = [_row(f) for f in frames]
    beyond[1]["patch_correlation"] += 2e-6
    assert not checker.check_flags_off(_document(_replay_run(run, beyond, 2)), inventory)["passed"]

    pixel = [_row(f) for f in frames]
    pixel[2]["pixel_xy"] = [0.0, 0.0]
    assert not checker.check_flags_off(_document(_replay_run(run, pixel, 2)), inventory)["passed"]

    after_stop = [_row(f) for f in frames]
    after_stop[3]["pixel_xy"] = [0.0, 0.0]
    result = checker.check_flags_off(_document(_replay_run(run, after_stop, 2)), inventory)
    assert result["passed"]
    assert result["detail"]["totals"]["after_stop_frames_differing_informational"] == 1


def test_flags_off_fails_when_a_recorded_run_is_missing_or_errored(checker, tmp_path):
    run, frames = _recording(tmp_path, ["approach", "approach"])
    other, _ = _recording(tmp_path, ["approach"], name="batch-b/run_00_other")
    replayed = _document(_replay_run(run, [_row(f) for f in frames], 1))
    assert not checker.check_flags_off(replayed, {NAME: run, "batch-b/run_00_other": other})["passed"]
    errored = _document({"run": NAME, "path": str(run), "error": "ValueError()"})
    assert not checker.check_flags_off(errored, {NAME: run})["passed"]


def test_provenance_requires_the_registration_commits_sources_and_complete_replays(checker, monkeypatch):
    registered = {"tools/replay_jaw_self_mask.py": "a" * 64, "source/x.py": "b" * 64}
    monkeypatch.setattr(checker, "registered_sources", lambda paths: {p: registered.get(p) for p in paths})
    flags = {
        "off": {"jaw_self_mask": False, "closure_hold": False},
        "mask": {"jaw_self_mask": True, "closure_hold": False},
        "mask_hold": {"jaw_self_mask": True, "closure_hold": True},
    }

    def headers(**changes):
        base = {
            "code_revision": checker.REGISTRATION_COMMIT,
            "source_files_differ_from_revision": False,
            "source_sha256": dict(registered),
            "summary": {"runs": checker.EXPECTED_RUNS, "errors": []},
        }
        return {name: {**base, "flags": flags[name], **changes.get(name, {})} for name in flags}

    assert checker.check_provenance(headers())["passed"]
    # A replay written after the depth arm existed records it; off is the only accepted value.
    with_depth = {name: {"flags": {**flags[name], "depth_appearance": False}} for name in flags}
    assert checker.check_provenance(headers(**with_depth))["passed"]
    assert not checker.check_provenance(headers(off={"flags": {**flags["off"], "depth_appearance": True}}))["passed"]
    later = {name: {"code_revision": "0" * 40} for name in flags}
    assert not checker.check_provenance(headers(**later))["passed"]
    assert not checker.check_provenance(headers(off={"source_files_differ_from_revision": True}))["passed"]
    assert not checker.check_provenance(headers(mask_hold={"flags": flags["mask"]}))["passed"]
    assert not checker.check_provenance(headers(mask={"summary": {"runs": 3, "errors": []}}))["passed"]
    edited = {name: {"source_sha256": {**registered, "source/x.py": "c" * 64}} for name in flags}
    assert not checker.check_provenance(headers(**edited))["passed"]


def _study(tmp_path, divergences, last, key_rows=(), name=NAME):
    study = tmp_path / "study"
    batch, run = name.split("/")
    (study / "runs" / batch).mkdir(parents=True)
    (study / "key").mkdir(exist_ok=True)
    document = {
        "last_frame_replayed": last,
        "arms": {
            "M": {
                "min_unmasked": 140,
                "divergences": divergences,
                "cut": {"stop_frame": None, "stop_reason": None, "detach_frame": None},
            }
        },
        "init": {"arms": {"M": {"state": "initialized", "features": 13}}},
        "preview": {"arms": {"M": {"state": "tracking", "reason": None}}},
    }
    (study / "runs" / batch / f"{run}.json").write_text(json.dumps(document))
    runs = [{"summary": {"run": name}, "rows": list(key_rows)}] if key_rows else []
    (study / "key/key_runs.json").write_text(json.dumps({"runs": runs}))
    return study


def _key_row(row):
    return {
        "frame": row["index"],
        "M": {
            "state": row["state"],
            "reason": row["reason"],
            "pixel_xy": row["pixel_xy"],
            "jaw_mask_patch_unmasked": row["unmasked_patch_elements"],
            "feature_count": row["feature_count"],
            "jaw_mask_dropped_valid": row["mask_dropped_valid_features"],
            "jaw_mask_unmasked_flow_would_pass": row["unmasked_flow_would_pass"],
            "flow_reason": row["flow_reason"],
            "jaw_mask_patch_masked_prev": row["masked_patch_elements_prev_cur"][0],
            "jaw_mask_patch_masked_cur": row["masked_patch_elements_prev_cur"][1],
            "patch_correlation": row["patch_correlation"],
            "jaw_mask_patch_correlation_kept": row["patch_correlation_kept"],
            "target_position_world_m": row["target_position_world_m"],
        },
    }


def test_mask_check_requires_the_studys_divergent_frames_values_cut_and_key_rows(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "KEPT_19444_R3", (NAME, 1, 139))
    monkeypatch.setattr(checker, "KEY_RUNS", 1)
    monkeypatch.setattr(checker, "KEY_ROWS", 1)
    run, frames = _recording(tmp_path, ["approach", "approach", "approach"])
    rows = [_row(f) for f in frames]
    rows[1].update(state="jaw_mask_occluded", reason="jaw_mask_occluded", patch_correlation=None)
    rows[1].update(unmasked_patch_elements=139)
    divergence = checker._study_measurement_diff(rows[1], frames[1]["live_vision"]["measurement"])
    divergence.update(
        frame=1,
        mask={"jaw_mask_patch_unmasked": 139, "jaw_mask_dropped_valid": 0},
        cut_variant=["approach", None, False],
    )
    inventory = {NAME: run}
    study = _study(tmp_path, [divergence], 2, key_rows=[_key_row(rows[1])])
    assert checker.check_mask(_document(_replay_run(run, rows, 2)), study, inventory)["passed"]

    no_divergence = _study(tmp_path / "a", [], 2, key_rows=[_key_row(rows[1])])
    assert not checker.check_mask(_document(_replay_run(run, rows, 2)), no_divergence, inventory)["passed"]

    other_cut = dict(divergence, cut_variant=["stopped", "vision_invalid", False])
    stopped = _study(tmp_path / "b", [other_cut], 2, key_rows=[_key_row(rows[1])])
    assert not checker.check_mask(_document(_replay_run(run, rows, 2)), stopped, inventory)["passed"]

    no_key_rows = _study(tmp_path / "c", [divergence], 2)
    assert not checker.check_mask(_document(_replay_run(run, rows, 2)), no_key_rows, inventory)["passed"]

    wrong = [dict(r) for r in rows]
    wrong[1]["unmasked_patch_elements"] = 140
    assert not checker.check_mask(_document(_replay_run(run, wrong, 2)), study, inventory)["passed"]


def _held(row):
    hold = {"held": True, "held_target_age_s": 0.3, "explanation": "jaw_mask_occluded", "dt_mm": 0.0, "dr_deg": 0.0}
    return {
        **row,
        "closure_hold": hold,
        "vision_source": "closure_hold",
        "waived_checks": list(PROTOCOL_WAIVED),
        "held_target_age_s": 0.3,
    }


PROTOCOL_WAIVED = (
    "vision_invalid",
    "vision_stale_or_future",
    "vision_timestamp_regressed",
    "vision_frame_reused_during_closure",
)


def _instrumented(frames, **overrides):
    entry = {
        "sent_target_is_closure_start_target": True,
        "held_timestamp_is_closure_start_time": True,
        "closure_reference_is_closure_start": True,
        "sent_vision_source": "closure_hold",
    }
    return {
        NAME: {
            "closure_start_frame": 0,
            "held_frames": list(frames),
            "held": [{"frame": i, **entry, **overrides} for i in frames],
        }
    }


def _hold_case(checker, tmp_path, monkeypatch, held_frames=(2, 3, 4), stop_at_3=False):
    """A miniature 530 run: closure starts at 0, the recording stops at 3, the deadline is 4."""
    monkeypatch.setattr(checker, "RUNS_530", (NAME,))
    monkeypatch.setattr(checker, "CLOSURE_START_530", 0)
    monkeypatch.setattr(checker, "RECORDED_STOP_530", [3, "vision_invalid"])
    monkeypatch.setattr(checker, "HELD_THROUGH_STOP_530", [2, 3])
    monkeypatch.setattr(checker, "HELD_AFTER_STOP_530", [4])
    run, frames = _recording(tmp_path, ["closing", "closing", "closing", "stopped", "stopped"])
    base = [_row(f, cut_phase="closing", stopped_reason=None) for f in frames]
    base[4].update(cut_phase="retreat", detach_event=True, evaluable=False)
    if stop_at_3:
        base[3].update(cut_phase="stopped", stopped_reason="unstable_during_closure")
    rows = [_held(r) if r["index"] in held_frames else r for r in base]
    mask_only = _replay_run(run, base, 3, stop_frame=2, stop_reason="vision_invalid")
    mask_runs = checker.compact_mask_runs(_document(mask_only))
    replay = {
        "closure_start_frame": 0,
        "held_frames": list(held_frames),
        "cut_without_hold_stop_frame": 2,
        "cut_without_hold_stop_reason": "vision_invalid",
        "detach_frame": 4,
    }
    document = _document(_replay_run(run, rows, 3, recorded_closure_start=0, **replay))
    return run, document, mask_runs


def test_hold_check_passes_the_530_deadline_structure_and_rejects_each_departure(checker, tmp_path, monkeypatch):
    run, document, mask_runs = _hold_case(checker, tmp_path, monkeypatch)
    inventory = {NAME: run}
    assert checker.check_hold(document, mask_runs, _instrumented((2, 3, 4)), inventory)["passed"]

    moved = _instrumented((2, 3, 4), sent_target_is_closure_start_target=False)
    assert not checker.check_hold(document, mask_runs, moved, inventory)["passed"]

    unlabelled = json.loads(json.dumps(document))
    unlabelled["runs"][0]["frames_detail"][2]["waived_checks"] = PROTOCOL_WAIVED[:3]
    assert not checker.check_hold(unlabelled, mask_runs, _instrumented((2, 3, 4)), inventory)["passed"]


def test_hold_check_requires_the_deadline_hold_and_closing_before_it(checker, tmp_path, monkeypatch):
    run, document, mask_runs = _hold_case(checker, tmp_path / "a", monkeypatch, held_frames=(2, 3))
    result = checker.check_hold(document, mask_runs, _instrumented((2, 3)), {NAME: run})
    assert not result["passed"]
    assert "530_deadline_hold" in {p["kind"] for p in result["detail"]["problems"]}
    run, document, mask_runs = _hold_case(checker, tmp_path / "b", monkeypatch, stop_at_3=True)
    result = checker.check_hold(document, mask_runs, _instrumented((2, 3, 4)), {NAME: run})
    assert "530_not_closing_before_deadline" in {p["kind"] for p in result["detail"]["problems"]}


def test_hold_check_rejects_a_hold_outside_530_and_an_incomplete_replay(checker, tmp_path, monkeypatch):
    run, document, mask_runs = _hold_case(checker, tmp_path, monkeypatch)
    inventory = {NAME: run}
    monkeypatch.setattr(checker, "RUNS_530", ("batch-z/run_00_elsewhere",))
    kinds = {p["kind"] for p in checker.check_hold(document, mask_runs, {}, inventory)["detail"]["problems"]}
    assert kinds >= {"hold_outside_530", "530_runs_missing"}
    monkeypatch.setattr(checker, "RUNS_530", (NAME,))
    other, _ = _recording(tmp_path, ["approach"], name="batch-b/run_00_other")
    wider = {NAME: run, "batch-b/run_00_other": other}
    result = checker.check_hold(document, mask_runs, _instrumented((2, 3, 4)), wider)
    assert not result["passed"]
    assert "run_set" in {p["kind"] for p in result["detail"]["problems"]}


def test_the_protocols_waived_checks_are_hard_coded(checker):
    assert list(PROTOCOL_WAIVED) == checker.PROTOCOL_WAIVED_CHECKS


@pytest.mark.skipif(not (RECORDED_530 / "frames.json").is_file(), reason="recorded planned-pose run not present")
def test_held_frames_of_a_recorded_530_run_carry_the_closure_start_target_and_time(checker):
    result = checker.held_targets({RECORDED_530.parent.name + "/" + RECORDED_530.name: RECORDED_530})
    [run] = result.values()
    assert run["held_frames"] == [76, 77, 78, 79, 80]
    assert [h["frame"] for h in run["held"]] == run["held_frames"]
    assert all(
        h["sent_target_is_closure_start_target"]
        and h["held_timestamp_is_closure_start_time"]
        and h["closure_reference_is_closure_start"]
        for h in run["held"]
    )
    assert checker.replay.VisionPruningDemo.__name__ == "VisionPruningDemo"


def test_outside_530_unchanged_measurements_must_keep_the_recorded_cut_decisions(checker, tmp_path, monkeypatch):
    run, document, mask_runs = _hold_case(checker, tmp_path, monkeypatch)
    monkeypatch.setattr(checker, "RUNS_530", ("batch-z/run_00_elsewhere",))
    no_hold = json.loads(json.dumps(document))
    for row in no_hold["runs"][0]["frames_detail"]:
        row.update(closure_hold={"held": False}, vision_source="live", waived_checks=[], held_target_age_s=None)
    no_hold["runs"][0]["replay"]["held_frames"] = []
    no_hold["runs"][0]["comparison_through_recorded_stop"]["cut_mismatch_frames"] = [3]
    kinds = {p["kind"] for p in checker.check_hold(no_hold, mask_runs, {}, {NAME: run})["detail"]["problems"]}
    assert "recorded_measurements_but_other_cut_decisions" in kinds


def test_the_named_morning_15004_control_must_stop_at_75_without_a_hold(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "RUNS_530", ())
    name = "tree1-listed-morning-20260926/run_02_morning_tree1_v15004_baseline"
    monkeypatch.setattr(checker, "MORNING_15004", name)
    monkeypatch.setattr(checker, "MORNING_15004_STOP", [1, "vision_invalid"])
    run, frames = _recording(tmp_path, ["approach", "stopped"], name=name)
    rows = [_row(f) for f in frames]
    shadow = {"cut_without_hold_stop_frame": 1, "cut_without_hold_stop_reason": "vision_invalid"}
    replayed = _replay_run(run, rows, 1, name=name, stop_frame=1, stop_reason="vision_invalid", **shadow)
    mask_runs = checker.compact_mask_runs(_document(replayed))
    result = checker.check_hold(_document(replayed), mask_runs, {}, {name: run})
    assert result["detail"]["problems"] == [], result["detail"]["problems"]
    assert result["passed"] and result["detail"]["morning_15004"]["mask_hold_stop"] == [1, "vision_invalid"]
    moved = _replay_run(run, rows, 1, name=name, stop_frame=None, stop_reason=None, **shadow)
    result = checker.check_hold(_document(moved), mask_runs, {}, {name: run})
    assert "morning_15004_stop" in {p["kind"] for p in result["detail"]["problems"]}
    missing = checker.check_hold(_document(), {}, {}, {name: run})
    assert "morning_15004_missing" in {p["kind"] for p in missing["detail"]["problems"]}


def test_a_recorded_preview_pixel_or_correlation_is_compared_with_the_report(checker, tmp_path):
    run, frames = _recording(tmp_path, ["approach", "approach"])
    report = json.loads((run / "report.json").read_text())
    report["initial_live_vision"]["measurement"].update(pixel_xy=[10.0, 20.0], patch_correlation=0.9)
    (run / "report.json").write_text(json.dumps(report))
    replayed = _replay_run(run, [_row(f) for f in frames], 1)
    replayed["preview"].update(pixel_xy=[10.0, 20.0], patch_correlation=0.9 + 5e-7)
    result = checker.check_flags_off(_document(replayed), {NAME: run})
    assert result["passed"] and result["detail"]["totals"]["preview_pixel_and_correlation_compared"] == 1
    replayed["preview"]["pixel_xy"] = [10.0, 21.0]
    assert not checker.check_flags_off(_document(replayed), {NAME: run})["passed"]
