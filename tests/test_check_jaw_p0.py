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
        "evaluable": True,
        "mouth_distance_m": 0.003,
    }
    row.update(overrides)
    return row


def _replay_run(run, rows, last, name=NAME, **replay):
    return {
        "run": name,
        "path": str(run),
        "frames_detail": rows,
        "compared_through_frame": last,
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


def test_provenance_requires_one_revision_unchanged_sources_and_this_checkouts_sources(checker):
    current = {name: checker.sha256(ROOT / name) for name in checker.replay.SOURCE_FILES}
    flags = {
        "off": {"jaw_self_mask": False, "closure_hold": False},
        "mask": {"jaw_self_mask": True, "closure_hold": False},
        "mask_hold": {"jaw_self_mask": True, "closure_hold": True},
    }

    def headers(**changes):
        base = {"code_revision": "abc", "source_files_differ_from_revision": False, "source_sha256": current}
        return {name: {**base, "flags": flags[name], **changes.get(name, {})} for name in flags}

    assert checker.check_provenance(headers())["passed"]
    assert not checker.check_provenance(headers(mask={"code_revision": "def"}))["passed"]
    assert not checker.check_provenance(headers(off={"source_files_differ_from_revision": True}))["passed"]
    assert not checker.check_provenance(headers(mask_hold={"flags": flags["mask"]}))["passed"]
    other = {**current, next(iter(current)): "0" * 64}
    assert not checker.check_provenance(headers(off={"source_sha256": other}))["passed"]


def _study(tmp_path, divergences, last, name=NAME):
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
    (study / "key/key_runs.json").write_text(json.dumps({"runs": []}))
    return study


def test_mask_check_requires_the_studys_divergent_frames_and_values(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "KEPT_19444_R3", (NAME, 1, 139))
    run, frames = _recording(tmp_path, ["approach", "approach", "approach"])
    rows = [_row(f) for f in frames]
    rows[1].update(state="jaw_mask_occluded", reason="jaw_mask_occluded", patch_correlation=None)
    rows[1].update(unmasked_patch_elements=139)
    divergence = checker._study_measurement_diff(rows[1], frames[1]["live_vision"]["measurement"])
    divergence.update(frame=1, mask={"jaw_mask_patch_unmasked": 139, "jaw_mask_dropped_valid": 0})
    study = _study(tmp_path, [divergence], 2)
    assert checker.check_mask(_document(_replay_run(run, rows, 2)), study)["passed"]

    other_tmp = tmp_path / "other"
    other_tmp.mkdir()
    missing = _study(other_tmp, [], 2)
    assert not checker.check_mask(_document(_replay_run(run, rows, 2)), missing)["passed"]

    wrong = [dict(r) for r in rows]
    wrong[1]["unmasked_patch_elements"] = 140
    assert not checker.check_mask(_document(_replay_run(run, wrong, 2)), study)["passed"]


def _held(row, waived, age=0.3):
    hold = {"held": True, "held_target_age_s": age, "explanation": "jaw_mask_occluded", "dt_mm": 0.0, "dr_deg": 0.0}
    return {
        **row,
        "closure_hold": hold,
        "vision_source": "closure_hold",
        "waived_checks": waived,
        "held_target_age_s": age,
    }


def _hold_setup(checker, tmp_path, monkeypatch, held_frames=(1,)):
    monkeypatch.setattr(checker, "RUNS_530", (NAME,))
    monkeypatch.setattr(checker, "HELD_THROUGH_STOP_530", list(held_frames))
    run, frames = _recording(tmp_path, ["closing", "closing", "closing"])
    base = [_row(f) for f in frames]
    waived = list(checker.CLOSURE_HOLD_WAIVED_CHECKS)
    rows = [_held(r, waived) if r["index"] in held_frames else r for r in base]
    held = {
        NAME: {
            "closure_start_frame": 0,
            "held_frames": list(held_frames),
            "held": [
                {
                    "frame": i,
                    "sent_target_is_reference": True,
                    "reference_is_closure_start_frame": True,
                    "held_timestamp_is_reference": True,
                    "sent_vision_source": "closure_hold",
                }
                for i in held_frames
            ],
        }
    }
    mask_runs = checker.compact_mask_runs(_document(_replay_run(run, base, 2)))
    return run, rows, held, mask_runs


def test_hold_check_passes_a_labelled_530_hold_and_rejects_bad_labels_and_targets(checker, tmp_path, monkeypatch):
    run, rows, held, mask_runs = _hold_setup(checker, tmp_path, monkeypatch)
    document = _document(_replay_run(run, rows, 2, closure_start_frame=0, held_frames=[1]))
    assert checker.check_hold(document, mask_runs, held)["passed"]

    unlabelled = [dict(r) for r in rows]
    unlabelled[1]["waived_checks"] = []
    document = _document(_replay_run(run, unlabelled, 2, closure_start_frame=0, held_frames=[1]))
    assert not checker.check_hold(document, mask_runs, held)["passed"]

    moved = {NAME: {**held[NAME], "held": [{**held[NAME]["held"][0], "sent_target_is_reference": False}]}}
    document = _document(_replay_run(run, rows, 2, closure_start_frame=0, held_frames=[1]))
    assert not checker.check_hold(document, mask_runs, moved)["passed"]


def test_hold_check_rejects_a_hold_outside_the_530_runs(checker, tmp_path, monkeypatch):
    run, rows, held, mask_runs = _hold_setup(checker, tmp_path, monkeypatch)
    monkeypatch.setattr(checker, "RUNS_530", ("batch-z/run_00_elsewhere",))
    document = _document(_replay_run(run, rows, 2, closure_start_frame=0, held_frames=[1]))
    result = checker.check_hold(document, mask_runs, held)
    assert not result["passed"]
    assert {p["kind"] for p in result["detail"]["problems"]} >= {"hold_outside_530", "530_runs_missing"}


@pytest.mark.skipif(not (RECORDED_530 / "frames.json").is_file(), reason="recorded planned-pose run not present")
def test_held_frames_of_a_recorded_530_run_carry_the_closure_reference(checker):
    result = checker.held_targets({RECORDED_530.parent.name + "/" + RECORDED_530.name: RECORDED_530})
    [run] = result.values()
    assert run["held_frames"] == [76, 77, 78, 79, 80]
    assert [h["frame"] for h in run["held"]] == run["held_frames"]
    assert all(
        h["sent_target_is_reference"] and h["reference_is_closure_start_frame"] and h["held_timestamp_is_reference"]
        for h in run["held"]
    )
    assert checker.replay.VisionPruningDemo.__name__ == "VisionPruningDemo"
