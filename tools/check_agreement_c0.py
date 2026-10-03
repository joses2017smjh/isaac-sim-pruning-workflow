#!/usr/bin/env python3
"""Check the agreement-arm closed-loop C0 gate (protocol of October 3) before any GPU submission.

The controller gained one option, ``depth_appearance_arm``, that selects the depth-aware check's registered
``agreement`` arm. This gate shows that the change selects that arm and changes nothing else. The published
held-out and regression replays (``docs/evidence/depth_*_replay_2026-10-01.json``, code ``ff363c9``) recorded
both arms on 141 recordings; they are the reference.

A. The offline tool still reproduces the published replays run for run (``tools/replay_depth_appearance.py``).
B. The controller with every arm off reproduces all 141 recordings through each recorded stop.
C. The controller with the strict arm measures exactly what the published strict arm measured, and changes a
   cut decision only in runs where that arm diverged (the depth closed-loop C0 criterion, re-run).
D. The controller with the agreement arm measures exactly what the published agreement arm measured: the same
   divergent frames, with the same state, reason, pixel and correlation, and the same depth event at every
   grey-check failure; it changes a cut decision only in runs where that arm diverged.
E. The controller with the strict arm reproduces the 10 live depth closed-loop recordings through each stop.
F. On those 10 recordings the agreement arm departs from the recording exactly where the recorded live event
   was accepted and the strict confidence fell below the gate, and continues there; at every grey-check
   failure its depth test (acceptance, conditions, reasons and statistics) equals the recorded live one.
G. The controller with the jaw self-mask and the closure hold reproduces the 6 live jaw-in-view recordings
   through each stop, holds included.

"Equal" uses the depth closed-loop C0 rule: exact for every string, boolean, integer and list shape, and
within 1e-6 for floats (OpenCV and OpenBLAS differ in the last bits between CPU models). Any mismatch blocks
submission. Offline replay of recorded images only; nothing here grades a live run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/isaaclab_pruning"))
import check_depth_loop_c0 as depth_c0  # noqa: E402 - the committed depth closed-loop gate's comparison rules
import replay_depth_appearance as depth_tool  # noqa: E402
import replay_jaw_self_mask as controller_replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_AGREEMENT_ARM_CLOSED_LOOP_2026-10-03.md"
PUBLISHED = depth_c0.PUBLISHED
EXPECTED_RUNS = depth_c0.EXPECTED_RUNS
VISION_ROBUSTNESS = depth_tool.VISION_ROBUSTNESS
#: The live depth closed-loop batches (strict arm) and the live jaw-in-view batches (mask and hold).
DEPTH_LOOP_BATCHES = (
    "depth-loop-eve-r1-20261001",
    "depth-loop-mor-r1-20261001",
    "depth-loop-eve-r2-20261001",
    "depth-loop-mor-r2-20261001",
    "depth-loop-src-20261001",
    "depth-loop-ctl-19444-20261001",
    "depth-loop-ctl-12142-20261001",
)
JAW_HOLD_BATCHES = ("jaw-hold-a-20261001", "jaw-hold-b-20261001", "jaw-hold-c-20261001")
EXPECTED_LIVE_RUNS = {"depth_loop": 10, "jaw_hold": 6}
#: The depth test's own output in an event: equal across arms for the same frames (the arm changes only what
#: happens after acceptance).
EVENT_CORE = ("accept", "ncc", "reasons", "failed_conditions", "not_evaluated_conditions", "conditions", "stats")
MIN_CONFIDENCE = 0.15
COMPARE = depth_c0.COMPARE


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_runs(batches):
    """The recorded run directories of the named batches, in batch and run order."""
    return [run for batch in batches for run in sorted((VISION_ROBUSTNESS / batch).glob("run_*")) if run.is_dir()]


def check_controller_arm(run, path, arm):
    """C or D: the controller with the depth-aware check's ``arm`` on, against the published arm of that name."""
    result = controller_replay.replay_run(path, depth_appearance=True, depth_appearance_arm=arm)
    problems = []
    last = run["replayed_through_frame"]
    if result["compared_through_frame"] != last:
        problems.append({"kind": "compared_through", "controller": result["compared_through_frame"], "tool": last})
    rows = result["frames_detail"][: last + 1]
    published = run["divergence"][arm]
    controller_frames = [row["index"] for row in rows if not row["measurement_matches_recording"]]
    if controller_frames != published["mismatch_frames"]:
        problems.append(
            {"kind": "divergent_frames", "controller": controller_frames, "tool": published["mismatch_frames"]}
        )
    keys = ("state", "reason", "pixel_xy", "patch_correlation")
    for divergence in published["divergences"]:
        index = divergence["frame"]
        if index >= len(rows):
            continue
        mine = depth_c0.normalized({key: rows[index][key] for key in keys})
        theirs = {key: divergence["replayed"].get(key) for key in keys}
        if not COMPARE.equal(mine, theirs):
            problems.append({"kind": "divergent_measurement", "frame": index, "controller": mine, "tool": theirs})
    published_events = {str(event["frame"]): event for event in run["events"][arm]}
    live_events = {"preview": depth_c0.flat_event(result["preview"].get("depth_appearance"))}
    live_events.update({str(row["index"]): depth_c0.flat_event(row["depth_appearance"]) for row in rows})
    live_events = {frame: event for frame, event in live_events.items() if event is not None}
    if sorted(live_events) != sorted(published_events):
        problems.append({"kind": "event_frames", "controller": sorted(live_events), "tool": sorted(published_events)})
    for frame, event in published_events.items():
        differing = depth_c0.event_problems(event, live_events.get(frame))
        if differing:
            problems.append({"kind": "event", "frame": frame, "fields": differing})
    cut_frames = result["comparison_through_recorded_stop"]["cut_mismatch_frames"]
    if cut_frames and not published["mismatch_frames"]:
        problems.append({"kind": "cut_decision_without_divergence", "frames": cut_frames})
    return problems, {
        "measurement_divergence_frames": controller_frames,
        "cut_decision_divergence_frames": cut_frames,
        "replay_stop": [result["replay"]["stop_frame"], result["replay"]["stop_reason"]],
        "events": {
            frame: {key: event.get(key) for key in ("accept", "decision", "strict_confidence", "agreement_confidence")}
            for frame, event in live_events.items()
        },
    }


def check_live_reproduced(path, **flags):
    """E or G: the controller with ``flags`` reproduces a live recording made with them, through its stop."""
    result = controller_replay.replay_run(path, **flags)
    comparison = result["comparison_through_recorded_stop"]
    return {
        "reproduces": result["reproduces_recording_through_recorded_stop"],
        "tracker_config_matches": result["tracker_config_matches_recording"],
        "measurement_mismatch_frames": comparison["measurement_mismatch_frames"],
        "cut_mismatch_frames": comparison["cut_mismatch_frames"],
        "recorded_stop": [result["recorded"]["stop_frame"], result["recorded"]["stop_reason"]],
        "held_frames": result["replay"].get("held_frames"),
    }


def event_core(event):
    return {key: event.get(key) for key in EVENT_CORE} if event else None


def expected_agreement_departures(frames, last):
    """Frames where a live strict run's accepted event ended in a low-confidence stop: there the agreement arm
    keeps tracking (its confidence replaces the NCC by the agreement fraction, at least 0.5 by condition 4)."""
    departures = []
    for frame in frames[: last + 1]:
        event = (frame["live_vision"].get("measurement") or {}).get("depth_appearance")
        if event and event.get("accept") is True and (event.get("strict_confidence") or 0.0) < MIN_CONFIDENCE:
            departures.append(frame["index"])
    return departures


def check_live_agreement(path):
    """F: the agreement arm on a live strict depth closed-loop recording."""
    frames = json.loads((path / "frames.json").read_text())["frames"]
    result = controller_replay.replay_run(path, depth_appearance=True, depth_appearance_arm="agreement")
    last = result["compared_through_frame"]
    rows = result["frames_detail"][: last + 1]
    problems = []
    expected = expected_agreement_departures(frames, last)
    departed = [row["index"] for row in rows if not row["measurement_matches_recording"]]
    if departed != expected:
        problems.append({"kind": "departure_frames", "controller": departed, "expected": expected})
    for index in expected:
        if index < len(rows) and rows[index]["state"] != "tracking":
            problems.append({"kind": "agreement_did_not_continue", "frame": index, "state": rows[index]["state"]})
    cut_frames = result["comparison_through_recorded_stop"]["cut_mismatch_frames"]
    if cut_frames and not expected:
        problems.append({"kind": "cut_decision_without_departure", "frames": cut_frames})
    for row in rows:
        recorded = (frames[row["index"]]["live_vision"].get("measurement") or {}).get("depth_appearance")
        replayed = row.get("depth_appearance")
        if (recorded is None) != (replayed is None):
            problems.append({"kind": "event_presence", "frame": row["index"]})
        elif recorded is not None:
            mine = depth_c0.normalized(event_core(replayed))
            theirs = depth_c0.normalized(event_core(recorded))
            if not COMPARE.equal(mine, theirs):
                differing = sorted(key for key in EVENT_CORE if not COMPARE.equal(mine.get(key), theirs.get(key)))
                problems.append({"kind": "event_core", "frame": row["index"], "fields": differing})
    return problems, {
        "departure_frames": departed,
        "expected_departure_frames": expected,
        "cut_decision_divergence_frames": cut_frames,
        "compared_through_frame": last,
        "replay_stop": [result["replay"]["stop_frame"], result["replay"]["stop_reason"]],
    }


def main(argv=None):  # noqa: C901 - one linear pass over the gate's checks
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    import cv2

    cv2.setNumThreads(1)
    runs = []
    for selection, relative in PUBLISHED.items():
        document = json.loads((ROOT / relative).read_text())
        if document["selection"] != selection or len(document["runs"]) != EXPECTED_RUNS[selection]:
            parser.error(f"{relative}: unexpected selection or run count")
        runs += [(selection, run) for run in document["runs"]]
    depth_loop, jaw_hold = live_runs(DEPTH_LOOP_BATCHES), live_runs(JAW_HOLD_BATCHES)
    if (len(depth_loop), len(jaw_hold)) != (EXPECTED_LIVE_RUNS["depth_loop"], EXPECTED_LIVE_RUNS["jaw_hold"]):
        parser.error(
            f"expected 10 depth closed-loop and 6 jaw-in-view recordings, found {len(depth_loop)}/{len(jaw_hold)}"
        )

    problems = {key: [] for key in "ABCDEFG"}
    detail = {key: {} for key in "ABCDEFG"}
    for number, (selection, run) in enumerate(runs, 1):
        name = run["run"]
        path = VISION_ROBUSTNESS / name
        differing = depth_c0.check_tool(run, path)
        if differing:
            problems["A"].append({"run": name, "fields": differing})
        off = depth_c0.check_controller_off(path)
        detail["B"][name] = off
        if not (off["reproduces"] and off["tracker_config_matches"]):
            problems["B"].append({"run": name, **off})
        for key, arm in (("C", "strict"), ("D", "agreement")):
            found, info = check_controller_arm(run, path, arm)
            detail[key][name] = {"selection": selection, **info}
            problems[key] += [{"run": name, **item} for item in found]
        print(
            f"[{number}/{len(runs)}] {name}: A={differing or 'ok'} "
            f"B={'ok' if off['reproduces'] else 'MISMATCH'} "
            f"C_div={detail['C'][name]['measurement_divergence_frames']} "
            f"D_div={detail['D'][name]['measurement_divergence_frames']}",
            flush=True,
        )
    for path in depth_loop:
        name = f"{path.parent.name}/{path.name}"
        strict = check_live_reproduced(path, depth_appearance=True)
        detail["E"][name] = strict
        if not (strict["reproduces"] and strict["tracker_config_matches"]):
            problems["E"].append({"run": name, **strict})
        found, info = check_live_agreement(path)
        detail["F"][name] = info
        problems["F"] += [{"run": name, **item} for item in found]
        print(f"{name}: E={'ok' if strict['reproduces'] else 'MISMATCH'} F={info['departure_frames']}", flush=True)
    for path in jaw_hold:
        name = f"{path.parent.name}/{path.name}"
        jaw = check_live_reproduced(path, jaw_self_mask=True, closure_hold=True)
        detail["G"][name] = jaw
        if not (jaw["reproduces"] and jaw["tracker_config_matches"]):
            problems["G"].append({"run": name, **jaw})
        print(f"{name}: G={'ok' if jaw['reproduces'] else 'MISMATCH'} held={jaw['held_frames']}", flush=True)

    statements = {
        "A": "The offline tool reproduces the published held-out and regression replays run for run.",
        "B": "The controller with every arm off reproduces all 141 recordings through each recorded stop.",
        "C": "The controller with the strict arm measures exactly what the published strict arm measured, and "
        "changes a cut decision only in runs where that arm diverged.",
        "D": "The controller with the agreement arm measures exactly what the published agreement arm measured, "
        "and changes a cut decision only in runs where that arm diverged.",
        "E": "The controller with the strict arm reproduces the 10 live depth closed-loop recordings through each "
        "recorded stop.",
        "F": "On the 10 live depth closed-loop recordings the agreement arm departs from the recording exactly where "
        "an accepted event ended in a low-confidence stop, continues there, and its depth test equals the live one "
        "at every grey-check failure.",
        "G": "The controller with the jaw self-mask and the closure hold reproduces the 6 live jaw-in-view "
        "recordings through each recorded stop.",
    }
    checks = [
        {
            "id": key,
            "statement": statements[key],
            "passed": not problems[key],
            "problems": problems[key][:100],
            "detail": detail[key],
        }
        for key in "ABCDEFG"
    ]
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    inputs = {relative: sha256(ROOT / relative) for relative in PUBLISHED.values()}
    for path in depth_loop + jaw_hold:
        for name in ("frames.json", "report.json"):
            inputs[f"{path.parent.name}/{path.name}/{name}"] = sha256(path / name)
    document = {
        "schema_version": 1,
        "scope": (
            "C0 pre-submission gate of the agreement-arm closed-loop test: offline CPU replays of the 12 held-out "
            "and 129 earlier recordings, the 10 live depth closed-loop recordings and the 6 live jaw-in-view "
            "recordings through the repository's offline tool and controller, against the published replays and "
            "the recordings themselves. Recorded images only; nothing here is a live outcome or a grade."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": bool(dirty),
        "inputs_sha256": inputs,
        "float_tolerance": depth_c0.FLOAT_TOLERANCE,
        "max_float_difference_accepted": COMPARE.max_float_difference,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    serialized = json.dumps(depth_c0.normalized(document), indent=1, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({check["id"]: check["passed"] for check in checks} | {"passed": document["passed"]}, indent=1))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
