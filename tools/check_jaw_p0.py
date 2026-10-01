#!/usr/bin/env python3
"""Check the jaw-in-view P0 gate (protocol of October 1) on three replays of every recorded run.

The replays are ``tools/replay_jaw_self_mask.py --all`` with both flags off, with the jaw self-mask, and with
the mask and the closure hold, all from the registration commit with its replayed sources unchanged. Each must
cover exactly the 129 recorded runs found here. Every recorded value is read here from the run's own
``frames.json`` and ``report.json`` (the files of this checkout's inventory, which must be the files the replay
read), never from the replay's copies.

1. Flags off. Through each run's recorded stop the replay equals the recording: state, reason and pixel
   exactly, correlation within 1e-6, cut phase, stop reason and detach event exactly, plus the seed
   initialization and the preview state. Frames after a recorded stop show a stopped robot; their differences
   are counted, not judged. For the runs whose seed the runner's visibility check rejected, the replay takes
   that rejection from the report, so their initialization agrees by construction.
2. Mask on. The offline study's stored outputs (arm M) are reproduced: the frames that differ from the
   recording are the study's frames, with the study's values and cut decision on them; each run's stop and
   detach equal the study's; and all 149 key rows of the 6 key runs match, including 19444 r3 frame 67 at
   139 kept patch elements.
3. Mask and hold.
   - The hold applies only in the three 530 runs. In each, closure starts at 74, the recording stops at 79
     (gate lost during closure), and the hold applies at 76-79 with the main cutter still closing at 79, and
     at the deadline frame 80, which is after the recorded stop and so not evaluable.
   - Every held frame is labelled ``closure_hold`` and lists exactly the protocol's four waived checks; every
     other frame is ``live`` and waives nothing; no new stop reason occurs.
   - The measurements equal the mask-only replay, and so do the cut decisions outside 530. The shadow
     cutter's stop equals the mask-only replay's stop.
   - The target and timestamp each held frame sent are read by re-running the three 530 replays with a
     recording subclass of the controller, which passes every value through unchanged, and are compared with
     the closure-start frame's own measurement and time.

Nothing here grades a live run; only tools/validate_vision_sequence.py does.
"""

from __future__ import annotations

import argparse
import gc
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
CHECKER = "tools/check_jaw_p0.py"
VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
#: The commit that registered the protocol; P0's replays are regenerated from it.
REGISTRATION_COMMIT = "a6e8266ed64b0589d98241954b036a3a529389ec"
#: Batches recorded after the study (the counterfactual and the live experiment); P0 covers the earlier runs.
EXCLUDED_BATCH_PREFIXES = ("jaw-shadow-", "jaw-hold-")
EXPECTED_RUNS = 129
TOLERANCE = 1e-6
#: The four checks the protocol lets a held frame waive.
PROTOCOL_WAIVED_CHECKS = [
    "vision_invalid",
    "vision_stale_or_future",
    "vision_timestamp_regressed",
    "vision_frame_reused_during_closure",
]
NEW_STOPS = {"closure_hold_outside_closing", "closure_hold_target_mismatch", "invalid_vision_source"}
RUNS_530 = tuple(f"planned-pose-gpu-r{r}-20260929/run_00_source_tree0_v530_planned_pose" for r in (1, 2, 3))
CLOSURE_START_530 = 74
RECORDED_STOP_530 = [79, "gate_lost_during_closure"]
HELD_THROUGH_STOP_530 = [76, 77, 78, 79]
HELD_AFTER_STOP_530 = [80]
KEPT_19444_R3 = ("planned-pose-gpu-r3-20260929/run_01_source_tree1_v19444_planned_pose", 67, 139)
KEY_RUNS, KEY_ROWS = 6, 149
MEASUREMENT_FIELDS = ("state", "reason", "pixel_xy", "patch_correlation", "unmasked_patch_elements")


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


def run_set_problems(document, inventory):
    """The replay must cover exactly the inventory's runs, each once."""
    names = [run["run"] for run in document["runs"]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if sorted(set(names)) == sorted(inventory) and not duplicates and len(names) == EXPECTED_RUNS:
        return []
    return [
        {
            "kind": "run_set",
            "runs": len(names),
            "missing": sorted(set(inventory) - set(names)),
            "extra": sorted(set(names) - set(inventory)),
            "duplicates": duplicates,
        }
    ]


def recorded_path(run, inventory, problems):
    """The inventory's directory for a replayed run; a different directory in the replay is a mismatch."""
    path = inventory.get(run["run"])
    if path is None:
        problems.append({"run": run["run"], "kind": "not_in_inventory"})
        return None
    if Path(run["path"]).resolve() != Path(path).resolve():
        problems.append({"run": run["run"], "kind": "recording_path", "replay": run["path"], "inventory": str(path)})
    return path


def index_problem(run, frames):
    if [row["index"] for row in run["frames_detail"]] != list(range(len(frames))):
        return {"run": run["run"], "kind": "row_indexes", "rows": len(run["frames_detail"]), "frames": len(frames)}
    return None


def check_provenance(headers):
    """All three replays from the registration commit, sources unchanged and equal to this checkout's."""
    flags = {name: doc["flags"] for name, doc in headers.items()}
    expected = {
        "off": {"jaw_self_mask": False, "closure_hold": False},
        "mask": {"jaw_self_mask": True, "closure_hold": False},
        "mask_hold": {"jaw_self_mask": True, "closure_hold": True},
    }
    revisions = {name: doc["code_revision"] for name, doc in headers.items()}
    hashes = [doc["source_sha256"] for doc in headers.values()]
    current = {name: sha256(ROOT / name) for name in replay.SOURCE_FILES}
    summaries = {
        name: {"runs": doc["summary"]["runs"], "errors": doc["summary"]["errors"]} for name, doc in headers.items()
    }
    passed = (
        flags == expected
        and set(revisions.values()) == {REGISTRATION_COMMIT}
        and all(doc["source_files_differ_from_revision"] is False for doc in headers.values())
        and all(h == hashes[0] for h in hashes)
        and hashes[0] == current
        and all(s["runs"] == EXPECTED_RUNS and not s["errors"] for s in summaries.values())
    )
    return check(
        "provenance",
        "The three replays have the expected flags, come from the registration commit with unchanged sources equal "
        "to this checkout's, and each replayed 129 runs without an error.",
        passed,
        {
            "flags": flags,
            "registration_commit": REGISTRATION_COMMIT,
            "code_revision": revisions,
            "source_files_differ_from_revision": {
                n: d["source_files_differ_from_revision"] for n, d in headers.items()
            },
            "source_sha256": hashes[0],
            "sources_equal_this_checkout": hashes[0] == current,
            "summaries": summaries,
        },
    )


def check_flags_off(document, inventory):
    """P0 part 1: the flags-off replay equals every recording through its recorded stop."""
    mismatches = run_set_problems(document, inventory)
    frames_compared = correlation_pairs = after_frames = after_mismatch = 0
    max_dcorr = 0.0
    per_run = []
    for run in document["runs"]:
        if "error" in run:
            mismatches.append({"run": run["run"], "kind": "replay_error", "detail": run["error"]})
            continue
        path = recorded_path(run, inventory, mismatches)
        if path is None:
            continue
        frames, report = recording(path)
        rows = run["frames_detail"]
        problem = index_problem(run, frames)
        if problem:
            mismatches.append(problem)
            continue
        stop = recorded_stop(frames)
        last = stop if stop is not None else len(frames) - 1
        run_max = 0.0
        for i, (frame, row) in enumerate(zip(frames, rows, strict=True)):
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
        "runs_in_replay": len(document["runs"]),
        "recorded_runs_with_frames": len(inventory),
        "frames_compared_through_stop": frames_compared,
        "correlation_pairs": correlation_pairs,
        "max_abs_correlation_difference": max_dcorr,
        "frames_after_stop_informational": after_frames,
        "after_stop_frames_differing_informational": after_mismatch,
        "tool_reproduces_true": document["summary"]["reproduce_recording_through_recorded_stop"],
    }
    return check(
        "flags_off_exact",
        "With both flags off the replay reproduces all 129 recorded runs through each recorded stop.",
        not mismatches and len(per_run) == EXPECTED_RUNS,
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


def cut_decision(row):
    return [row["cut_phase"], row["stopped_reason"], bool(row["detach_event"])]


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
    if "cut_variant" in study and cut_decision(row) != [*study["cut_variant"][:2], bool(study["cut_variant"][2])]:
        problems.append(["cut_variant", cut_decision(row), study["cut_variant"]])
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


def check_mask(document, study_dir, inventory):
    """P0 part 2: the mask-only replay reproduces the study's stored per-frame outputs (arm M)."""
    key_rows = {
        run["summary"]["run"]: {row["frame"]: row["M"] for row in run["rows"]}
        for run in json.loads((study_dir / "key/key_runs.json").read_text())["runs"]
    }
    mismatches = run_set_problems(document, inventory)
    totals = dict.fromkeys(
        ("runs", "frames_compared", "study_divergent_frames", "replay_divergent_frames", "key_rows_compared"), 0
    )
    totals["frames_rendered_with_other_progress_through_stop"] = 0
    totals["unchanged_frames_with_masked_patch_elements_informational"] = 0
    key_runs_seen = set()
    kept_19444 = None
    for run in document["runs"]:
        name = run["run"]
        if "error" in run:
            mismatches.append({"run": name, "kind": "replay_error", "detail": run["error"]})
            continue
        batch, run_name = name.split("/")
        study_path = study_dir / "runs" / batch / f"{run_name}.json"
        if not study_path.is_file():
            mismatches.append({"run": name, "kind": "study_file_missing"})
            continue
        study = json.loads(study_path.read_text())
        if "error" in study:
            mismatches.append({"run": name, "kind": "study_error", "detail": study["error"]})
            continue
        path = recorded_path(run, inventory, mismatches)
        if path is None:
            continue
        frames, _ = recording(path)
        problem = index_problem(run, frames)
        if problem:
            mismatches.append(problem)
            continue
        arm = study["arms"]["M"]
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
        for i in range(min(last + 1, len(rows))):
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
        if name in key_rows:
            key_runs_seen.add(name)
            for i, key in sorted(key_rows[name].items()):
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
            k: (run["replay"][k] if run["replay"][k] is not None and run["replay"][k] <= last else None)
            for k in ("stop_frame", "detach_frame")
        }
        mine_cut = [through["stop_frame"], run["replay"]["stop_reason"] if through["stop_frame"] is not None else None]
        mine_cut.append(through["detach_frame"])
        if mine_cut != [arm["cut"]["stop_frame"], arm["cut"]["stop_reason"], arm["cut"]["detach_frame"]]:
            mismatches.append(
                {"run": name, "kind": "cut_decision_differs_from_study", "replay": mine_cut, "study": arm["cut"]}
            )
        if name == KEPT_19444_R3[0] and len(rows) > KEPT_19444_R3[1]:
            row = rows[KEPT_19444_R3[1]]
            kept_19444 = {
                "frame": KEPT_19444_R3[1],
                "kept_elements": row["unmasked_patch_elements"],
                "state": row["state"],
            }
        totals["runs"] += 1
        totals["frames_compared"] += last + 1
        totals["study_divergent_frames"] += len(study_divergent)
    if kept_19444 is None or kept_19444["kept_elements"] != KEPT_19444_R3[2]:
        mismatches.append({"run": KEPT_19444_R3[0], "kind": "kept_elements_frame_67", "detail": kept_19444})
    if len(key_rows) != KEY_RUNS or key_runs_seen != set(key_rows) or totals["key_rows_compared"] != KEY_ROWS:
        mismatches.append(
            {
                "kind": "key_rows",
                "key_runs": sorted(key_rows),
                "key_runs_replayed": sorted(key_runs_seen),
                "rows_compared": totals["key_rows_compared"],
                "expected_rows": KEY_ROWS,
            }
        )
    return check(
        "mask_reproduces_study",
        "With the mask on the replay reproduces the study's per-frame outputs and cut decisions on all 129 runs, "
        "and all 149 key rows, including 19444 r3 frame 67 at 139 kept elements.",
        not mismatches and totals["runs"] == EXPECTED_RUNS,
        {
            "totals": totals,
            "19444_r3_frame_67": kept_19444,
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
            start_target = None if start is None else [float(v) for v in rows[start]["target_position_world_m"]]
            start_time = None if start is None else rows[start]["time_s"]
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
                        # Independent of the cutter: the closure-start frame's own measurement and time.
                        "sent_target_is_closure_start_target": event["sent_target"] == start_target,
                        "held_timestamp_is_closure_start_time": event["sent_held_target_timestamp_s"] == start_time,
                        # Consistency with the cutter's closure reference.
                        "closure_reference_is_closure_start": event["reference_target"] == start_target
                        and event["reference_time_s"] == start_time,
                        "sent_vision_source": event["sent_vision_source"],
                    }
                )
            results[name] = {"closure_start_frame": start, "held_frames": run["replay"]["held_frames"], "held": held}
        return results
    finally:
        replay.VisionPruningDemo = original


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


def _check_530(name, run, rows, frames, held_frames, instrumented, problems):
    """The 530 deadline structure and the held target, appended to ``problems``; returns the per-run record."""
    stop = recorded_stop(frames)
    recorded = [stop, None if stop is None else frames[stop]["live_vision"]["cut"]["stopped_reason"]]
    starts = [run["recorded"]["closure_start_frame"], run["replay"]["closure_start_frame"]]
    through = [i for i in held_frames if stop is None or i <= stop]
    after = [i for i in held_frames if stop is not None and i > stop]
    deadline = HELD_AFTER_STOP_530[0]
    if starts != [CLOSURE_START_530, CLOSURE_START_530] or recorded != RECORDED_STOP_530:
        problems.append({"run": name, "kind": "530_closure_start_or_recorded_stop", "starts": starts, "stop": recorded})
    if through != HELD_THROUGH_STOP_530:
        problems.append({"run": name, "kind": "530_held_through_stop", "got": through})
    if len(rows) <= deadline or cut_decision(rows[deadline - 1]) != ["closing", None, False]:
        problems.append({"run": name, "kind": "530_not_closing_before_deadline"})
    if after != HELD_AFTER_STOP_530 or len(rows) <= deadline or rows[deadline]["evaluable"]:
        problems.append({"run": name, "kind": "530_deadline_hold", "after": after})
    targets = instrumented.get("held") or []
    if [h["frame"] for h in targets] != held_frames or instrumented.get("held_frames") != held_frames:
        problems.append({"run": name, "kind": "530_instrumented_held_frames", "got": instrumented.get("held_frames")})
    elif not all(
        h.get("sent_target_is_closure_start_target")
        and h.get("held_timestamp_is_closure_start_time")
        and h.get("closure_reference_is_closure_start")
        and h.get("sent_vision_source") == "closure_hold"
        for h in targets
    ):
        problems.append({"run": name, "kind": "530_held_target"})
    return {
        "recorded_stop": recorded,
        "closure_start_frame": starts,
        "held_through_recorded_stop": through,
        "held_after_recorded_stop": after,
        "evaluable_held_frames": [i for i in held_frames if rows[i]["evaluable"]],
        "cut_without_hold_stop": [
            run["replay"]["cut_without_hold_stop_frame"],
            run["replay"]["cut_without_hold_stop_reason"],
        ],
        "replay_detach_frame": run["replay"]["detach_frame"],
        "held_ages_s": [rows[i]["held_target_age_s"] for i in held_frames],
        "explanations": {str(i): rows[i]["closure_hold"]["explanation"] for i in held_frames},
        "mouth_distance_mm_held": [
            None if rows[i]["mouth_distance_m"] is None else 1e3 * rows[i]["mouth_distance_m"] for i in held_frames
        ],
        "max_tool_motion_held": [
            max((rows[i]["closure_hold"]["dt_mm"] for i in held_frames), default=None),
            max((rows[i]["closure_hold"]["dr_deg"] for i in held_frames), default=None),
        ],
        "held_target_instrumented": targets,
    }


def check_hold(mask_hold, mask_runs, held, inventory):
    """P0 part 3: where the closure hold applies, and what a held frame carries."""
    problems = run_set_problems(mask_hold, inventory)
    if sorted(mask_runs) != sorted(inventory):
        problems.append({"kind": "mask_run_set", "runs": len(mask_runs)})
    if list(CLOSURE_HOLD_WAIVED_CHECKS) != PROTOCOL_WAIVED_CHECKS:
        problems.append({"kind": "waived_checks_constant", "code": list(CLOSURE_HOLD_WAIVED_CHECKS)})
    holds = {}
    per_530 = {}
    for run in mask_hold["runs"]:
        name = run["run"]
        if "error" in run:
            problems.append({"run": name, "kind": "replay_error", "detail": run["error"]})
            continue
        path = recorded_path(run, inventory, problems)
        if path is None:
            continue
        frames, _ = recording(path)
        problem = index_problem(run, frames)
        if problem:
            problems.append(problem)
            continue
        rows = run["frames_detail"]
        held_frames = [row["index"] for row in rows if (row["closure_hold"] or {}).get("held")]
        if held_frames != run["replay"]["held_frames"]:
            problems.append(
                {"run": name, "kind": "held_list", "rows": held_frames, "replay": run["replay"]["held_frames"]}
            )
        if held_frames:
            holds[name] = held_frames
        for row in rows:
            is_held = bool((row["closure_hold"] or {}).get("held"))
            label = [row["vision_source"], row["waived_checks"]]
            if label != (["closure_hold", PROTOCOL_WAIVED_CHECKS] if is_held else ["live", []]):
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
        if other is None or "error" in other or len(other["measurements"]) != len(rows):
            problems.append({"run": name, "kind": "mask_run_missing_or_short"})
            continue
        shadow = [run["replay"]["cut_without_hold_stop_frame"], run["replay"]["cut_without_hold_stop_reason"]]
        if shadow != other["stop"]:
            problems.append({"run": name, "kind": "shadow_stop_differs_from_mask_run", "shadow": shadow})
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
        per_530[name] = _check_530(name, run, rows, frames, held_frames, held.get(name) or {}, problems)
        per_530[name]["cut_decision_differs_from_mask_run_through_stop"] = decided
    missing = [name for name in RUNS_530 if name not in per_530]
    if missing:
        problems.append({"kind": "530_runs_missing", "runs": missing})
    return check(
        "hold_only_530",
        "The hold applies only in the three 530 runs, at 76-79 through the recorded stop (still closing at 79) and "
        "at the deadline frame 80 after it (not evaluable); held frames carry the closure-start target and time "
        "and the protocol's labels.",
        not problems and sorted(holds) == sorted(RUNS_530),
        {"runs_with_holds": holds, "per_530": per_530, "problems": problems[:100], "problem_count": len(problems)},
    )


def _load(path):
    gc.collect()
    return json.loads(Path(path).read_text())


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
    held = held_targets({name: inventory[name] for name in RUNS_530 if name in inventory})
    # One replay document in memory at a time: the job running this may have only a few GB.
    headers = {}
    document = _load(args.off)
    headers["off"] = header(document)
    flags_off = check_flags_off(document, inventory)
    document = None
    document = _load(args.mask)
    headers["mask"] = header(document)
    mask = check_mask(document, args.study_dir, inventory)
    mask_runs = compact_mask_runs(document)
    document = None
    document = _load(args.mask_hold)
    headers["mask_hold"] = header(document)
    hold = check_hold(document, mask_runs, held, inventory)
    document = mask_runs = None
    checks = [check_provenance(headers), flags_off, mask, hold]
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    own = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", CHECKER, *replay.SOURCE_FILES], text=True
    ).strip()
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
        "checker_or_replayed_sources_differ_from_revision": bool(own),
        "checker_sha256": sha256(ROOT / CHECKER),
        "replay_code_revision": headers["off"]["code_revision"],
        "inputs_sha256": {
            **{f"replay_{name}": {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
            "study_outputs": {"dir": str(args.study_dir), **manifest_digest(study_files, args.study_dir)},
            "recorded_frames_and_reports": {
                "dir": str(VISION_ROBUSTNESS.resolve()),
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
