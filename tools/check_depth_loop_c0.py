#!/usr/bin/env python3
"""Check the depth-aware appearance closed-loop C0 gate (protocol of October 1) before any GPU submission.

The published held-out replay (``docs/evidence/depth_*_replay_2026-10-01.json``, code ``ff363c9``) is the
reference. On its 141 recordings (12 held-out, 129 earlier):

A. The moved implementation reproduces it. ``tools/replay_depth_appearance.py`` now imports the tracker from
   ``perception/depth_appearance.py``; its per-run results must equal the published ones (every field except
   the run's wall-clock time and directory).
B. The controller with every arm off still reproduces every recording through its recorded stop
   (``tools/replay_jaw_self_mask.py``).
C. The controller with ``depth_appearance=True`` measures what the offline strict arm measured. Through each
   recorded stop, its measurement differs from the recording on exactly the frames where the published strict
   arm did, with the same state, reason, pixel and correlation, and its depth event at every grey-check failure
   equals the published strict event field for field.
D. Where the strict arm never diverged, the controller with the arm on makes the recorded cut decisions through
   the recorded stop; the runs where it does diverge are listed.

"Equal" is exact for every string, boolean, integer and list shape, and within ``FLOAT_TOLERANCE`` for
floats: OpenCV's and OpenBLAS's kernels differ by CPU, so a replay on another node differs in the last bits of
the NCC (float32) and of the depth geometry (as the P0 criterion already allowed 1e-6 on the correlation). A
float near a threshold that flips a decision changes a discrete field and is caught. The largest float
difference is recorded. Any mismatch blocks submission. Offline replay of recorded images only; nothing here
grades a live run.
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
import replay_depth_appearance as depth_tool  # noqa: E402
import replay_jaw_self_mask as controller_replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md"
PUBLISHED = {
    "heldout": "docs/evidence/depth_heldout_replay_2026-10-01.json",
    "regression": "docs/evidence/depth_regression_replay_2026-10-01.json",
}
EXPECTED_RUNS = {"heldout": 12, "regression": 129}
#: Per-run fields that legitimately differ between two replays of the same code and inputs.
VOLATILE = ("replay_seconds", "path")
#: Floats agree within this absolute tolerance plus a relative 1e-9 (metres, pixels, correlations, fractions).
FLOAT_TOLERANCE = 1e-6
#: Fields of a published event record that are context, not the depth test's output.
EVENT_CONTEXT = ("frame", "after_recorded_detach", "recorded", "replayed", "recorded_cut_phase")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    """JSON-normalized exactly as the depth tool writes its output."""
    return json.loads(json.dumps(depth_tool._json_safe(value), allow_nan=False))


class Comparison:
    """Recursive equality with the float tolerance; remembers the largest float difference it accepted."""

    def __init__(self):
        self.max_float_difference = 0.0

    def equal(self, a, b):
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return a is b
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            difference = abs(float(a) - float(b))
            if difference > FLOAT_TOLERANCE + 1e-9 * abs(float(b)):
                return False
            self.max_float_difference = max(self.max_float_difference, difference)
            return True
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(self.equal(a[key], b[key]) for key in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(self.equal(x, y) for x, y in zip(a, b, strict=True))
        return type(a) is type(b) and a == b


COMPARE = Comparison()


def flat_event(event):
    """A live depth event in the published record's shape: the event's fields with its statistics merged in."""
    if event is None:
        return None
    flat = {key: value for key, value in event.items() if key != "stats"}
    flat.update(event.get("stats") or {})
    return normalized(flat)


def event_problems(published, live):
    """Fields where a live event differs from the published one (the published record's own fields only)."""
    if live is None:
        return ["no live event"]
    return [
        key
        for key, value in published.items()
        if key not in EVENT_CONTEXT and (key not in live or not COMPARE.equal(live[key], value))
    ]


def check_tool(run, path):
    """A: the moved implementation reproduces one published run."""
    mine = normalized(depth_tool.replay_run(path))
    theirs = {key: value for key, value in run.items() if key not in VOLATILE}
    mine = {key: value for key, value in mine.items() if key not in VOLATILE}
    keys = set(mine) | set(theirs)
    differing = sorted(k for k in keys if k not in mine or k not in theirs or not COMPARE.equal(mine[k], theirs[k]))
    return differing


def check_controller_off(path):
    """B: the controller with every arm off reproduces the recording through its recorded stop."""
    result = controller_replay.replay_run(path)
    return {
        "reproduces": result["reproduces_recording_through_recorded_stop"],
        "tracker_config_matches": result["tracker_config_matches_recording"],
        "measurement_mismatch_frames": result["comparison_through_recorded_stop"]["measurement_mismatch_frames"],
        "cut_mismatch_frames": result["comparison_through_recorded_stop"]["cut_mismatch_frames"],
    }


def check_controller_depth(run, path):
    """C and D: the controller with depth_appearance on, against the published strict arm."""
    result = controller_replay.replay_run(path, depth_appearance=True)
    problems = []
    last = run["replayed_through_frame"]
    if result["compared_through_frame"] != last:
        problems.append({"kind": "compared_through", "controller": result["compared_through_frame"], "tool": last})
    rows = result["frames_detail"][: last + 1]
    strict = run["divergence"]["strict"]
    controller_frames = [row["index"] for row in rows if not row["measurement_matches_recording"]]
    if controller_frames != strict["mismatch_frames"]:
        problems.append(
            {"kind": "divergent_frames", "controller": controller_frames, "tool": strict["mismatch_frames"]}
        )
    for divergence in strict["divergences"]:
        index = divergence["frame"]
        if index >= len(rows):
            continue
        row, replayed = rows[index], divergence["replayed"]
        mine = normalized({k: row[k] for k in ("state", "reason", "pixel_xy", "patch_correlation")})
        theirs = {k: replayed.get(k) for k in ("state", "reason", "pixel_xy", "patch_correlation")}
        if not COMPARE.equal(mine, theirs):
            problems.append({"kind": "divergent_measurement", "frame": index, "controller": mine, "tool": theirs})
    published_events = {str(event["frame"]): event for event in run["events"]["strict"]}
    live_events = {"preview": flat_event(result["preview"].get("depth_appearance"))}
    live_events.update({str(row["index"]): flat_event(row["depth_appearance"]) for row in rows})
    live_events = {frame: event for frame, event in live_events.items() if event is not None}
    if sorted(live_events) != sorted(published_events):
        problems.append({"kind": "event_frames", "controller": sorted(live_events), "tool": sorted(published_events)})
    for frame, event in published_events.items():
        differing = event_problems(event, live_events.get(frame))
        if differing:
            problems.append({"kind": "event", "frame": frame, "fields": differing})
    cut_frames = result["comparison_through_recorded_stop"]["cut_mismatch_frames"]
    if cut_frames and not strict["mismatch_frames"]:
        problems.append({"kind": "cut_decision_without_divergence", "frames": cut_frames})
    return problems, {
        "measurement_divergence_frames": controller_frames,
        "cut_decision_divergence_frames": cut_frames,
        "recorded_stop": [result["recorded"]["stop_frame"], result["recorded"]["stop_reason"]],
        "replay_stop": [result["replay"]["stop_frame"], result["replay"]["stop_reason"]],
        "replay_detach_frame": result["replay"]["detach_frame"],
        "events": {
            frame: {k: event.get(k) for k in ("accept", "decision", "strict_confidence")}
            for frame, event in live_events.items()
        },
    }


def check(identifier, statement, problems, detail):
    return {
        "id": identifier,
        "statement": statement,
        "passed": not problems,
        "problems": problems[:100],
        "detail": detail,
    }


def main(argv=None):
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
    tool_problems, off_problems, depth_problems, depth_detail, off_detail = [], [], [], {}, {}
    for number, (selection, run) in enumerate(runs, 1):
        name = run["run"]
        path = depth_tool.VISION_ROBUSTNESS / name
        differing = check_tool(run, path)
        if differing:
            tool_problems.append({"run": name, "fields": differing})
        off = check_controller_off(path)
        off_detail[name] = off
        if not (off["reproduces"] and off["tracker_config_matches"]):
            off_problems.append({"run": name, **off})
        problems, detail = check_controller_depth(run, path)
        depth_detail[name] = {"selection": selection, **detail}
        depth_problems += [{"run": name, **problem} for problem in problems]
        print(
            f"[{number}/{len(runs)}] {name}: tool={differing or 'ok'} "
            f"off={'ok' if off['reproduces'] else 'MISMATCH'} depth_problems={len(problems)} "
            f"div={detail['measurement_divergence_frames']} cut_div={detail['cut_decision_divergence_frames']}",
            flush=True,
        )
    checks = [
        check(
            "A_tool_reproduces_published",
            "The moved implementation reproduces the published held-out and regression replays run for run.",
            tool_problems,
            {"runs": len(runs)},
        ),
        check(
            "B_controller_off_exact",
            "The controller with every arm off reproduces all 141 recordings through each recorded stop.",
            off_problems,
            {"runs": len(off_detail)},
        ),
        check(
            "C_D_controller_depth_equals_strict_arm",
            "The controller with depth_appearance on measures exactly what the published strict arm measured, "
            "and changes a cut decision only in runs where that arm diverged.",
            depth_problems,
            {
                "runs": depth_detail,
                "runs_with_cut_decision_changes": sorted(
                    name for name, detail in depth_detail.items() if detail["cut_decision_divergence_frames"]
                ),
            },
        ),
    ]
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    document = {
        "schema_version": 1,
        "scope": (
            "C0 pre-submission gate of the depth-aware appearance closed-loop test: offline CPU replays of the 12 "
            "held-out and 129 earlier recordings through the moved D_strict + J implementation and through the "
            "repository controller, against the published held-out replay. Recorded images only."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": bool(dirty),
        "inputs_sha256": {relative: sha256(ROOT / relative) for relative in PUBLISHED.values()},
        "float_tolerance": FLOAT_TOLERANCE,
        "max_float_difference_accepted": COMPARE.max_float_difference,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }
    serialized = json.dumps(normalized(document), indent=1, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({c["id"]: c["passed"] for c in checks} | {"passed": document["passed"]}, indent=1))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
