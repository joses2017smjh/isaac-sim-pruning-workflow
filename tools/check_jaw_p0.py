#!/usr/bin/env python3
"""Check the jaw-in-view P0 gate (protocol of October 1) on three replays of every recorded run.

The replays are ``tools/replay_jaw_self_mask.py --all`` with both flags off, with the jaw self-mask, and with
the mask and the closure hold, all from one revision whose replayed sources were unchanged. Every recorded
value is read here from the run's own ``frames.json`` and ``report.json``, never from the replay's copies.

1. Flags off. Every recorded run with frames is replayed, and through its recorded stop the replay equals the
   recording: state, reason and pixel exactly, correlation within 1e-6, cut phase, stop reason and detach
   event exactly, plus the seed initialization and the preview.
2. Mask on. The offline study's stored outputs (arm M) are reproduced. The frames that differ from the
   recording are the study's frames, with the study's values on them, and the key runs' full rows match,
   including 19444 r3 frame 67 at 139 kept patch elements.
3. Mask and hold.
   - The hold applies only in the three 530 runs: at 76-79 through the recorded stop, and after it only on
     frames that are not evaluable.
   - Held frames carry the closure reference's target, its timestamp and their labels.
   - The measurements equal the mask-only replay, and so do the cut decisions outside 530. The shadow cutter
     equals the mask-only replay's cutter.
   - The target a held frame sent is read by re-running the three 530 replays with a recording subclass of
     the controller, which passes every value through unchanged.

Nothing here grades a live run; only tools/validate_vision_sequence.py does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/isaaclab_pruning"))
import replay_jaw_self_mask as replay  # noqa: E402

from isaaclab_pruning.task.simulated_cut import CLOSURE_HOLD_WAIVED_CHECKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md"
VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
#: The counterfactual batches recorded after the study; P0 covers the earlier runs only.
EXCLUDED_BATCH_PREFIXES = ("jaw-shadow-",)
EXPECTED_RUNS = 129
TOLERANCE = 1e-6
NEW_STOPS = {"closure_hold_outside_closing", "closure_hold_target_mismatch", "invalid_vision_source"}
RUNS_530 = tuple(f"planned-pose-gpu-r{r}-20260929/run_00_source_tree0_v530_planned_pose" for r in (1, 2, 3))
HELD_THROUGH_STOP_530 = [76, 77, 78, 79]
KEPT_19444_R3 = ("planned-pose-gpu-r3-20260929/run_01_source_tree1_v19444_planned_pose", 67, 139)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_digest(paths, base):
    """One digest over many files: sha256 of the sorted 'relative-path sha256' lines."""
    lines = sorted(f"{Path(p).relative_to(base)} {sha256(p)}" for p in paths)
    return {"files": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def recorded_runs(base=VISION_ROBUSTNESS):
    """Every run directory with at least one recorded wrist frame, except the excluded batches."""
    runs = {}
    for path in sorted(Path(base).glob("*/run_*")):
        if path.parent.name.startswith(EXCLUDED_BATCH_PREFIXES):
            continue
        if (path / "frames").is_dir() and any((path / "frames").glob("wrist_*.png")):
            runs[f"{path.parent.name}/{path.name}"] = path
    return runs


def recording(path):
    """The run's recorded frames and report (read per use: 129 parsed recordings would not fit in memory)."""
    frames = json.loads((Path(path) / "frames.json").read_text())["frames"]
    report = json.loads((Path(path) / "report.json").read_text())
    return frames, report


def recorded_stop(frames):
    return next((f["index"] for f in frames if f["live_vision"]["cut"]["phase"] == "stopped"), None)


def correlation_close(a, b, tolerance=TOLERANCE):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


def check(identifier, statement, passed, detail):
    return {"id": identifier, "statement": statement, "passed": bool(passed), "detail": detail}


def header(document):
    """A replay document's provenance fields, kept after the document itself is released."""
    keys = ("flags", "code_revision", "source_files_differ_from_revision", "source_sha256", "summary")
    return {key: document[key] for key in keys}


def check_provenance(documents):
    """All three replays from one revision, sources unchanged, and the same sources as this checkout."""
    flags = {name: doc["flags"] for name, doc in documents.items()}
    expected = {
        "off": {"jaw_self_mask": False, "closure_hold": False},
        "mask": {"jaw_self_mask": True, "closure_hold": False},
        "mask_hold": {"jaw_self_mask": True, "closure_hold": True},
    }
    revisions = {doc["code_revision"] for doc in documents.values()}
    hashes = [doc["source_sha256"] for doc in documents.values()]
    current = {name: sha256(ROOT / name) for name in replay.SOURCE_FILES}
    passed = (
        flags == expected
        and len(revisions) == 1
        and None not in revisions
        and all(doc["source_files_differ_from_revision"] is False for doc in documents.values())
        and all(h == hashes[0] for h in hashes)
        and hashes[0] == current
    )
    return check(
        "provenance",
        "The three replays have the expected flags, one revision, unchanged sources, and this checkout's sources.",
        passed,
        {
            "flags": flags,
            "code_revision": sorted(revisions, key=str),
            "source_files_differ_from_revision": {
                n: d["source_files_differ_from_revision"] for n, d in documents.items()
            },
            "source_sha256": hashes[0],
            "sources_equal_this_checkout": hashes[0] == current,
        },
    )


def check_flags_off(document, inventory):
    """P0 part 1: the flags-off replay equals every recording through its recorded stop."""
    names = [r["run"] for r in document["runs"]]
    mismatches = []
    if sorted(names) != sorted(inventory) or len(set(names)) != len(names):
        mismatches.append(
            {
                "kind": "run_set",
                "missing": sorted(set(inventory) - set(names)),
                "extra": sorted(set(names) - set(inventory)),
            }
        )
    frames_compared = correlation_pairs = after_frames = after_mismatch = 0
    max_dcorr = 0.0
    per_run = []
    for run in document["runs"]:
        if "error" in run:
            mismatches.append({"run": run["run"], "kind": "replay_error", "detail": run["error"]})
            continue
        frames, report = recording(run["path"])
        rows = run["frames_detail"]
        if [row["index"] for row in rows] != list(range(len(frames))):
            mismatches.append({"run": run["run"], "kind": "row_indexes", "detail": [len(rows), len(frames)]})
            continue
        stop = recorded_stop(frames)
        last = stop if stop is not None else len(frames) - 1
        run_max = 0.0
        for i, (frame, row) in enumerate(zip(frames, rows)):
            measurement = frame["live_vision"].get("measurement") or {}
            cut = frame["live_vision"].get("cut") or {}
            problems = [
                [k, row[k], measurement.get(k)] for k in ("state", "reason", "pixel_xy") if row[k] != measurement.get(k)
            ]
            a, b = row["patch_correlation"], measurement.get("patch_correlation")
            if not correlation_close(a, b):
                problems.append(["patch_correlation", a, b])
            elif a is not None and i <= last:
                correlation_pairs += 1
                run_max = max(run_max, abs(a - b))
            for key, recorded in (("cut_phase", cut.get("phase")), ("stopped_reason", cut.get("stopped_reason"))):
                if row[key] != recorded:
                    problems.append([key, row[key], recorded])
            if bool(row["detach_event"]) != bool(cut.get("detach_event")):
                problems.append(["detach_event", row["detach_event"], cut.get("detach_event")])
            if i <= last and problems:
                mismatches.append({"run": run["run"], "kind": "frame", "frame": i, "problems": problems})
            elif i > last:
                after_frames += 1
                after_mismatch += bool(problems)
        recorded_init = (report.get("vision_initialization") or {}).get("tracker") or {}
        if (run["initialization"]["replayed_state"], run["initialization"]["replayed_feature_count"]) != (
            recorded_init.get("state"),
            recorded_init.get("feature_count", 0),
        ):
            mismatches.append({"run": run["run"], "kind": "initialization"})
        preview = report.get("initial_live_vision") or {}
        if not (
            run["preview"]["state"] == (preview.get("measurement") or {}).get("state")
            and run["preview"]["reason"] == (preview.get("measurement") or {}).get("reason")
            and run["preview"]["cut_phase"] == (preview.get("cut") or {}).get("phase")
            and run["preview"]["matches_recording"]
        ):
            mismatches.append({"run": run["run"], "kind": "preview"})
        if last != run["compared_through_frame"]:
            mismatches.append(
                {"run": run["run"], "kind": "compared_through", "detail": [last, run["compared_through_frame"]]}
            )
        frames_compared += last + 1
        max_dcorr = max(max_dcorr, run_max)
        per_run.append([run["run"], len(frames), stop, last + 1, run_max])
    totals = {
        "runs_in_replay": len(names),
        "recorded_runs_with_frames": len(inventory),
        "frames_compared_through_stop": frames_compared,
        "correlation_pairs": correlation_pairs,
        "max_abs_correlation_difference": max_dcorr,
        "frames_after_stop_informational": after_frames,
        "after_stop_frames_differing_informational": after_mismatch,
        "tool_reproduces_true": document["summary"]["reproduce_recording_through_recorded_stop"],
    }
    passed = not mismatches and len(names) == len(inventory) == EXPECTED_RUNS
    return check(
        "flags_off_exact",
        "With both flags off the replay reproduces all recorded runs through each recorded stop.",
        passed,
        {"totals": totals, "mismatches": mismatches[:100], "mismatch_count": len(mismatches), "per_run": per_run},
    )


def _study_measurement_diff(replayed, recorded):
    """The study's divergence criterion (state, reason, pixel, correlation, target and feature count)."""
    same = (
        replayed.get("state") == recorded.get("state")
        and replayed.get("reason") == recorded.get("reason")
        and replayed.get("pixel_xy") == recorded.get("pixel_xy")
        and correlation_close(replayed.get("patch_correlation"), recorded.get("patch_correlation"))
    )
    tm, tr = replayed.get("target_position_world_m"), recorded.get("target_position_world_m")
    target_mm = None if tm is None or tr is None else float(np.linalg.norm(np.asarray(tm) - np.asarray(tr)) * 1e3)
    same_target = (tm is None) == (tr is None) and (target_mm is None or target_mm <= 1e-6)
    if same and same_target and replayed.get("feature_count") == recorded.get("feature_count"):
        return None
    pm, pr = replayed.get("pixel_xy"), recorded.get("pixel_xy")
    changed_class = replayed.get("state") != recorded.get("state") or replayed.get("reason") != recorded.get("reason")
    return {
        "kind": "state" if changed_class else "numeric",
        "recorded": [recorded.get("state"), recorded.get("reason")],
        "variant": [replayed.get("state"), replayed.get("reason")],
        "pixel_diff_px": None if pm is None or pr is None else float(np.linalg.norm(np.asarray(pm) - np.asarray(pr))),
        "corr": [recorded.get("patch_correlation"), replayed.get("patch_correlation")],
        "target_diff_mm": target_mm,
        "features": [recorded.get("feature_count"), replayed.get("feature_count")],
    }


def _divergent_frame_problems(mine, study, row):
    problems = []
    for key in ("kind", "recorded", "variant", "pixel_diff_px", "features"):
        if mine[key] != study[key]:
            problems.append([key, mine[key], study[key]])
    if not correlation_close(mine["corr"][0], study["corr"][0], 0.0):
        problems.append(["recorded_corr", mine["corr"][0], study["corr"][0]])
    if not correlation_close(mine["corr"][1], study["corr"][1]):
        problems.append(["variant_corr", mine["corr"][1], study["corr"][1]])
    if not correlation_close(mine["target_diff_mm"], study["target_diff_mm"], 1e-9):
        problems.append(["target_diff_mm", mine["target_diff_mm"], study["target_diff_mm"]])
    mask = study.get("mask") or {}
    for field, value in (
        ("jaw_mask_patch_unmasked", row["unmasked_patch_elements"]),
        ("jaw_mask_dropped_valid", row["mask_dropped_valid_features"]),
        ("jaw_mask_unmasked_flow_would_pass", row["unmasked_flow_would_pass"]),
        ("flow_reason", row["flow_reason"]),
    ):
        if mask.get(field) != value and not (field not in mask and value is None):
            problems.append([field, value, mask.get(field)])
    return problems


def _key_row_problems(row, key):
    masked = row["masked_patch_elements_prev_cur"] or [None, None]
    problems = [
        [field, value, key.get(field)]
        for field, value in (
            ("state", row["state"]),
            ("reason", row["reason"]),
            ("pixel_xy", row["pixel_xy"]),
            ("jaw_mask_patch_unmasked", row["unmasked_patch_elements"]),
            ("feature_count", row["feature_count"]),
            ("jaw_mask_dropped_valid", row["mask_dropped_valid_features"]),
            ("jaw_mask_unmasked_flow_would_pass", row["unmasked_flow_would_pass"]),
            ("flow_reason", row["flow_reason"]),
            ("jaw_mask_patch_masked_prev", masked[0]),
            ("jaw_mask_patch_masked_cur", masked[1]),
        )
        if value != key.get(field)
    ]
    for field, value in (
        ("patch_correlation", row["patch_correlation"]),
        ("jaw_mask_patch_correlation_kept", row["patch_correlation_kept"]),
    ):
        if not correlation_close(value, key.get(field)):
            problems.append([field, value, key.get(field)])
    tm, tk = row["target_position_world_m"], key.get("target_position_world_m")
    if (tm is None) != (tk is None) or (tm is not None and float(np.linalg.norm(np.subtract(tm, tk))) > 1e-9):
        problems.append(["target_position_world_m", tm, tk])
    return problems


def check_mask(document, study_dir):
    """P0 part 2: the mask-only replay reproduces the study's stored per-frame outputs (arm M)."""
    key_rows = {
        run["summary"]["run"]: {row["frame"]: row["M"] for row in run["rows"]}
        for run in json.loads((study_dir / "key/key_runs.json").read_text())["runs"]
    }
    mismatches = []
    totals = dict.fromkeys(
        ("runs", "frames_compared", "study_divergent_frames", "replay_divergent_frames", "key_rows_compared"), 0
    )
    totals["frames_rendered_with_other_progress_through_stop"] = 0
    totals["unchanged_frames_with_masked_patch_elements_informational"] = 0
    cut_differs_from_study = []
    kept_19444 = None
    for run in document["runs"]:
        name = run["run"]
        if "error" in run:
            mismatches.append({"run": name, "kind": "replay_error", "detail": run["error"]})
            continue
        batch, run_name = name.split("/")
        study = json.loads((study_dir / "runs" / batch / f"{run_name}.json").read_text())
        if "error" in study:
            mismatches.append({"run": name, "kind": "study_error", "detail": study["error"]})
            continue
        arm = study["arms"]["M"]
        frames, _ = recording(run["path"])
        rows = run["frames_detail"]
        last = study["last_frame_replayed"]
        if last != run["compared_through_frame"]:
            mismatches.append(
                {"run": name, "kind": "last_frame", "study": last, "replay": run["compared_through_frame"]}
            )
        if arm["min_unmasked"] != 140:
            mismatches.append({"run": name, "kind": "study_min_unmasked", "value": arm["min_unmasked"]})
        study_divergent = {d["frame"]: d for d in arm["divergences"]}
        if any(f > last for f in study_divergent):
            mismatches.append({"run": name, "kind": "study_divergence_after_last"})
        for i in range(last + 1):
            row = rows[i]
            totals["frames_rendered_with_other_progress_through_stop"] += not row["rendered_progress_matches_recording"]
            mine = _study_measurement_diff(row, frames[i]["live_vision"].get("measurement") or {})
            theirs = study_divergent.get(i)
            totals["replay_divergent_frames"] += mine is not None
            if (mine is None) != (theirs is None):
                mismatches.append({"run": name, "kind": "divergent_set", "frame": i, "replay": mine, "study": theirs})
            elif mine is None and row["unmasked_patch_elements"] is not None and row["unmasked_patch_elements"] < 169:
                totals["unchanged_frames_with_masked_patch_elements_informational"] += 1
            elif mine is not None:
                problems = _divergent_frame_problems(mine, theirs, row)
                if problems:
                    mismatches.append({"run": name, "kind": "divergent_frame", "frame": i, "problems": problems})
        for i, key in sorted(key_rows.get(name, {}).items()):
            if i > last:
                continue
            totals["key_rows_compared"] += 1
            problems = _key_row_problems(rows[i], key)
            if problems:
                mismatches.append({"run": name, "kind": "key_row", "frame": i, "problems": problems})
        init, preview = study["init"]["arms"]["M"], study["preview"]["arms"]["M"]
        mine_init = [run["initialization"]["replayed_state"], run["initialization"]["replayed_feature_count"]]
        same_rejection = init["state"] == "not_initialized_seed_rejected" and mine_init[0] == "initialization_rejected"
        if mine_init != [init["state"], init["features"]] and not same_rejection:
            mismatches.append({"run": name, "kind": "initialization", "replay": mine_init, "study": init})
        if [run["preview"]["state"], run["preview"]["reason"]] != [preview["state"], preview["reason"]]:
            mismatches.append({"run": name, "kind": "preview", "study": preview})
        through = {
            k: (run["replay"][k] if (run["replay"][k] or 0) <= last else None) for k in ("stop_frame", "detach_frame")
        }
        mine_cut = [through["stop_frame"], run["replay"]["stop_reason"] if through["stop_frame"] is not None else None]
        mine_cut.append(through["detach_frame"])
        if mine_cut != [arm["cut"]["stop_frame"], arm["cut"]["stop_reason"], arm["cut"]["detach_frame"]]:
            cut_differs_from_study.append({"run": name, "replay": mine_cut, "study": arm["cut"]})
        if name == KEPT_19444_R3[0]:
            row = rows[KEPT_19444_R3[1]]
            kept_19444 = {
                "frame": KEPT_19444_R3[1],
                "kept_elements": row["unmasked_patch_elements"],
                "state": row["state"],
            }
        totals["runs"] += 1
        totals["frames_compared"] += last + 1
        totals["study_divergent_frames"] += len(study_divergent)
    kept_ok = kept_19444 is not None and kept_19444["kept_elements"] == KEPT_19444_R3[2]
    if not kept_ok:
        mismatches.append({"run": KEPT_19444_R3[0], "kind": "kept_elements_frame_67", "detail": kept_19444})
    return check(
        "mask_reproduces_study",
        "With the mask on the replay reproduces the study's per-frame outputs, including 19444 r3 frame 67 at 139.",
        not mismatches and totals["runs"] == EXPECTED_RUNS,
        {
            "totals": totals,
            "19444_r3_frame_67": kept_19444,
            "cut_decision_differs_from_study_informational": cut_differs_from_study,
            "mismatches": mismatches[:100],
            "mismatch_count": len(mismatches),
        },
    )


def held_targets(paths):
    """Re-run the 530 replays with a recording controller; per held frame, what the cutter was sent."""
    log = []

    class RecordingDemo(replay.VisionPruningDemo):
        def _closure_hold(self, observation, tool_pose_wxyz, time_s):
            sent, record = super()._closure_hold(observation, tool_pose_wxyz, time_s)
            reference = self.cutter.closure_reference
            if record["held"]:
                log.append(
                    {
                        "time_s": time_s,
                        "sent_target": [float(v) for v in sent.target_position_w],
                        "sent_vision_source": sent.vision_source,
                        "sent_held_target_timestamp_s": sent.held_target_timestamp_s,
                        "reference_target": [float(v) for v in reference.target_position_w],
                        "reference_time_s": reference.time_s,
                        "reference_vision_timestamp_s": reference.vision_timestamp_s,
                    }
                )
            return sent, record

    original, replay.VisionPruningDemo = replay.VisionPruningDemo, RecordingDemo
    try:
        results = {}
        for name, path in paths.items():
            log.clear()
            run = replay.replay_run(path, jaw_self_mask=True, closure_hold=True)
            rows = run["frames_detail"]
            start = run["replay"]["closure_start_frame"]
            held = []
            for row in rows:
                if not (row["closure_hold"] or {}).get("held"):
                    continue
                events = [e for e in log if e["time_s"] == row["time_s"]]
                if len(events) != 1:
                    held.append({"frame": row["index"], "recorded_hold_events": len(events)})
                    continue
                event = events[0]
                held.append(
                    {
                        "frame": row["index"],
                        "sent_target_is_reference": event["sent_target"] == event["reference_target"],
                        "reference_is_closure_start_frame": start is not None
                        and event["reference_time_s"] == rows[start]["time_s"]
                        and event["reference_target"] == [float(v) for v in rows[start]["target_position_world_m"]],
                        "held_timestamp_is_reference": event["sent_held_target_timestamp_s"]
                        == event["reference_vision_timestamp_s"],
                        "sent_vision_source": event["sent_vision_source"],
                    }
                )
            results[name] = {"closure_start_frame": start, "held_frames": run["replay"]["held_frames"], "held": held}
        return results
    finally:
        replay.VisionPruningDemo = original


MEASUREMENT_FIELDS = ("state", "reason", "pixel_xy", "patch_correlation", "unmasked_patch_elements")


def cut_decision(row):
    return row["cut_phase"], row["stopped_reason"], bool(row["detach_event"])


def compact_mask_runs(document):
    """Per run of the mask-only replay, what the hold check compares: its cut stop and per-frame outputs."""
    compact = {}
    for run in document["runs"]:
        if "error" in run:
            compact[run["run"]] = {"error": run["error"]}
            continue
        compact[run["run"]] = {
            "stop": [run["replay"]["stop_frame"], run["replay"]["stop_reason"]],
            "measurements": [tuple(json.dumps(row[k]) for k in MEASUREMENT_FIELDS) for row in run["frames_detail"]],
            "decisions": [cut_decision(row) for row in run["frames_detail"]],
        }
    return compact


def check_hold(mask_hold, mask_runs, held):
    """P0 part 3: where the closure hold applies, and what a held frame carries."""
    problems = []
    holds = {}
    per_530 = {}
    for run in mask_hold["runs"]:
        name = run["run"]
        if "error" in run:
            problems.append({"run": name, "kind": "replay_error", "detail": run["error"]})
            continue
        rows = run["frames_detail"]
        frames, _ = recording(run["path"])
        stop = recorded_stop(frames)
        held_frames = [row["index"] for row in rows if (row["closure_hold"] or {}).get("held")]
        if held_frames != run["replay"]["held_frames"]:
            problems.append(
                {"run": name, "kind": "held_list", "rows": held_frames, "replay": run["replay"]["held_frames"]}
            )
        if held_frames:
            holds[name] = held_frames
        for row in rows:
            is_held = bool((row["closure_hold"] or {}).get("held"))
            label = (row["vision_source"], row["waived_checks"])
            want = ("closure_hold", list(CLOSURE_HOLD_WAIVED_CHECKS)) if is_held else ("live", [])
            if label != want:
                problems.append({"run": name, "kind": "labels", "frame": row["index"], "got": label})
            age = row["held_target_age_s"]
            if is_held and (age is None or not correlation_close(age, row["closure_hold"]["held_target_age_s"], 1e-12)):
                problems.append({"run": name, "kind": "held_age", "frame": row["index"]})
            if not is_held and age is not None:
                problems.append({"run": name, "kind": "held_age_on_live_frame", "frame": row["index"]})
            if row["stopped_reason"] in NEW_STOPS:
                problems.append({"run": name, "kind": "new_stop", "frame": row["index"], "got": row["stopped_reason"]})
            if row["cut_without_hold"] is None:
                problems.append({"run": name, "kind": "cut_without_hold_missing", "frame": row["index"]})
        other = mask_runs.get(name)
        if other is None or "error" in other:
            problems.append({"run": name, "kind": "mask_run_missing"})
            continue
        shadow = [run["replay"]["cut_without_hold_stop_frame"], run["replay"]["cut_without_hold_stop_reason"]]
        if shadow != other["stop"]:
            problems.append({"run": name, "kind": "shadow_differs_from_mask_run", "shadow": shadow})
        last = run["compared_through_frame"]
        measured = [
            i
            for i in range(last + 1)
            if tuple(json.dumps(rows[i][k]) for k in MEASUREMENT_FIELDS) != other["measurements"][i]
        ]
        if measured:
            problems.append({"run": name, "kind": "measurement_differs_from_mask_run", "frames": measured[:20]})
        decided = [i for i in range(last + 1) if cut_decision(rows[i]) != other["decisions"][i]]
        if name not in RUNS_530:
            if held_frames:
                problems.append({"run": name, "kind": "hold_outside_530", "frames": held_frames})
            if decided:
                problems.append({"run": name, "kind": "cut_decision_differs_outside_530", "frames": decided[:20]})
            continue
        through = [i for i in held_frames if stop is None or i <= stop]
        after = [i for i in held_frames if stop is not None and i > stop]
        instrumented = held.get(name) or {}
        per_530[name] = {
            "recorded_stop": [stop, None if stop is None else frames[stop]["live_vision"]["cut"]["stopped_reason"]],
            "closure_start_frame": run["replay"]["closure_start_frame"],
            "held_through_recorded_stop": through,
            "held_after_recorded_stop": after,
            "evaluable_held_frames": [i for i in held_frames if rows[i]["evaluable"]],
            "cut_without_hold_stop": shadow,
            "cut_decision_differs_from_mask_run_through_stop": decided,
            "held_ages_s": [rows[i]["held_target_age_s"] for i in held_frames],
            "explanations": {i: rows[i]["closure_hold"]["explanation"] for i in held_frames},
            "mouth_distance_mm_held": [
                None if rows[i]["mouth_distance_m"] is None else 1e3 * rows[i]["mouth_distance_m"] for i in held_frames
            ],
            "max_tool_motion_held": [
                max((rows[i]["closure_hold"]["dt_mm"] for i in held_frames), default=None),
                max((rows[i]["closure_hold"]["dr_deg"] for i in held_frames), default=None),
            ],
            "held_target_instrumented": instrumented.get("held"),
        }
        if through != HELD_THROUGH_STOP_530:
            problems.append({"run": name, "kind": "530_held_through_stop", "got": through})
        if any(rows[i]["evaluable"] for i in after):
            problems.append({"run": name, "kind": "530_evaluable_hold_after_stop", "frames": after})
        targets = instrumented.get("held") or []
        if [h["frame"] for h in targets] != held_frames or instrumented.get("held_frames") != held_frames:
            problems.append(
                {"run": name, "kind": "530_instrumented_held_frames", "got": instrumented.get("held_frames")}
            )
        elif not all(
            h.get("sent_target_is_reference")
            and h.get("reference_is_closure_start_frame")
            and h.get("held_timestamp_is_reference")
            and h.get("sent_vision_source") == "closure_hold"
            for h in targets
        ):
            problems.append({"run": name, "kind": "530_held_target"})
    missing = [name for name in RUNS_530 if name not in per_530]
    if missing:
        problems.append({"kind": "530_runs_missing", "runs": missing})
    return check(
        "hold_only_530",
        "The hold applies only in the three 530 runs, at 76-79 through the recorded stop and after it only on "
        "frames that are not evaluable; held frames carry the closure reference's target and their labels.",
        not problems and sorted(holds) == sorted(RUNS_530),
        {"runs_with_holds": holds, "per_530": per_530, "problems": problems[:100], "problem_count": len(problems)},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off", type=Path, required=True, help="Replay with both flags off")
    parser.add_argument("--mask", type=Path, required=True, help="Replay with --jaw-self-mask")
    parser.add_argument("--mask-hold", type=Path, required=True, help="Replay with --jaw-self-mask --closure-hold")
    parser.add_argument("--study-dir", type=Path, required=True, help="The offline study's output directory")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    import cv2

    cv2.setNumThreads(1)
    paths = {"off": args.off, "mask": args.mask, "mask_hold": args.mask_hold}
    inventory = recorded_runs()
    study_files = sorted((args.study_dir / "runs").rglob("*.json")) + [args.study_dir / "key/key_runs.json"]
    # One replay document in memory at a time: the job running this may have only a few GB.
    held = held_targets({name: inventory[name] for name in RUNS_530 if name in inventory})
    headers = {}
    document = json.loads(args.off.read_text())
    headers["off"] = header(document)
    flags_off = check_flags_off(document, inventory)
    document = json.loads(args.mask.read_text())
    headers["mask"] = header(document)
    mask = check_mask(document, args.study_dir)
    mask_runs = compact_mask_runs(document)
    document = json.loads(args.mask_hold.read_text())
    headers["mask_hold"] = header(document)
    hold = check_hold(document, mask_runs, held)
    del document, mask_runs
    checks = [check_provenance(headers), flags_off, mask, hold]
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    recorded_files = [p / name for p in inventory.values() for name in ("frames.json", "report.json")]
    document = {
        "schema_version": 1,
        "scope": (
            "P0 pre-submission gate of the jaw-in-view experiment: offline CPU replays of the 129 earlier recorded "
            "runs through the repository controller with both flags off, with the jaw self-mask, and with the mask "
            "and the closure hold, against the recordings and the offline study. Recorded images only; no frame "
            "here is a live outcome or a grade."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "replay_code_revision": headers["off"]["code_revision"],
        "inputs_sha256": {
            **{f"replay_{name}": {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
            "study_outputs": {"dir": str(args.study_dir), **manifest_digest(study_files, args.study_dir)},
            "recorded_frames_and_reports": {
                "dir": str(VISION_ROBUSTNESS),
                **manifest_digest(recorded_files, VISION_ROBUSTNESS),
            },
        },
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }
    serialized = json.dumps(replay._json_safe(document), indent=1, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({c["id"]: c["passed"] for c in checks} | {"passed": document["passed"]}, indent=1))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
