#!/usr/bin/env python3
"""Score the depth-aware appearance closed-loop test (protocol of October 1) against C1-C6.

Outcomes come from the per-batch evidence that ``aggregate_eval.py`` wrote (the grader's classes, never
re-graded here); failed and incomplete runs stay in every table. Everything a clause names comes from each run's
own records:
- the depth event the live tracker attached to every grey-check failure
  (``live_vision.measurement.depth_appearance``);
- the recorded stop, and the tracker's reason at it.

C6 replays each run's own frames offline through the strict arm of ``tools/replay_depth_appearance.py``. Each run
is labelled with its GPU model.

Where the protocol leaves a detail open it is fixed in ``INTERPRETATIONS`` and recorded in the output. Results of
this arm change the 0.35 appearance gate's rule and are never pooled with unchanged-gate results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402
from score_planned_approach import gpu_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md"
EVIDENCE_DIR = "docs/evidence/depth_loop_2026-10-01"
LABEL = (
    "depth-aware appearance check D_strict + J; changes the 0.35 appearance gate's rule; simulator depth (RTX "
    "optical-Z ground truth, noise-free, perfectly registered); not a sensor"
)
#: Registered shadow windows (inclusive frame indexes) of the low-sun scenes.
SHADOW_WINDOW = {"evening": (56, 67), "morning": (72, 75)}
MIN_CONFIDENCE = 0.15
APPEARANCE_LOSS = "appearance_changed_or_occluded"
LOW_CONFIDENCE = "low_confidence"
CONTROL_PREFIX = "depth-loop-ctl-"
SOURCE_BATCH_PREFIX = "depth-loop-src-"
EXPECTED_RUNS = 10
INTERPRETATIONS = [
    "a grey-check failure is a frame whose live measurement carries a depth_appearance event (the tracker attaches "
    "one exactly when the grey NCC fails 0.35); the preview update is not a frame and is reported separately",
    "a run's stop is its first frame whose recorded cut phase is 'stopped'; 'in the shadow window' means that "
    "frame index lies in the registered inclusive window",
    "C1 is supported only when all 6 low-sun runs have a grey failure in their window and every first one is "
    "accepted; a run without one is untested and makes C1 partly supported",
    "C2 counts a pass only from the grader's class 'pass' (all 17 checks); 'stops in the shadow window' counts a "
    "stop for any reason, 'stops on appearance' only a stop whose tracker reason is appearance_changed_or_occluded",
    "C3 judges the first accepted grey failure in each evening 14944 run's window; a run with no grey failure "
    "there is untested",
    "C4's second clause is untested for a control with no grey failure; any accepted grey failure in a control "
    "refutes it",
    "C5 counts a failed check as any grader class other than 'pass'",
    "C6 compares the live run with the offline strict arm on its own frames: the strict arm must equal the "
    "recording (state, reason and pixel exactly, correlation within 1e-6) through the run's stop, and every live "
    "depth event must equal the offline event field for field (floats within 1e-6, as the C0 gate). The offline "
    "base tracker diverges by construction wherever the live arm continued, so it is not judged",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kind(batch):
    """The scene a batch registered: evening, morning, source or control."""
    if batch.startswith(CONTROL_PREFIX):
        return "control"
    if batch.startswith(SOURCE_BATCH_PREFIX):
        return "source"
    return "evening" if "-eve-" in batch else "morning"


def frame_facts(frames):
    """Per frame: index, cut phase, stop reason, the tracker's state and reason, and the depth event if any."""
    facts = []
    for frame in frames:
        live = frame.get("live_vision") or {}
        measurement = live.get("measurement") or {}
        cut = live.get("cut") or {}
        event = measurement.get("depth_appearance")
        facts.append(
            {
                "index": frame["index"],
                "cut_phase": cut.get("phase"),
                "stopped_reason": cut.get("stopped_reason"),
                "state": measurement.get("state"),
                "reason": measurement.get("reason"),
                "event": None
                if event is None
                else {
                    key: event.get(key)
                    for key in (
                        "accept",
                        "failed_conditions",
                        "not_evaluated_conditions",
                        "reasons",
                        "strict_confidence",
                        "decision",
                        "ended_by",
                        "ncc",
                    )
                },
            }
        )
    return facts


def run_record(root, batch_dir, row):
    """The fields C1-C6 need for one run, from its own records."""
    run = root / batch_dir / row["run_directory"]
    report = json.loads((run / "report.json").read_text())
    frames = json.loads((run / "frames.json").read_text())["frames"]
    facts = frame_facts(frames)
    stop = next((f for f in facts if f["cut_phase"] == "stopped"), None)
    batch = batch_dir.split("/")[-1]
    preview_event = ((report.get("initial_live_vision") or {}).get("measurement") or {}).get("depth_appearance")
    return {
        "batch": batch,
        "kind": kind(batch),
        "run_directory": row["run_directory"],
        "target": int(row["component_first_vertex"]),
        "daylight": row["daylight"],
        "outcome": row["outcome"],
        "checks": f"{row['checks_passed']}/{row['checks_total']}",
        "node": report.get("node"),
        "gpu_model": gpu_model(report.get("node")),
        "arm_readback": (report.get("depth_appearance") or {}).get("enabled"),
        "stop": None
        if stop is None
        else {"frame": stop["index"], "stopped_reason": stop["stopped_reason"], "tracker_reason": stop["reason"]},
        "grey_failures": [f["index"] for f in facts if f["event"] is not None],
        "events": {f["index"]: f["event"] for f in facts if f["event"] is not None},
        "facts": facts,
        "preview_event": None if preview_event is None else {"accept": preview_event.get("accept")},
        "path": str(run),
    }


def load(root):
    records = []
    for path in sorted((root / EVIDENCE_DIR).glob("depth-loop-*.json")):
        document = json.loads(path.read_text())
        batch_dir = document["batches"][0]["batch_dir"]
        records += [run_record(root, batch_dir, row) for row in document["runs"]]
    return records


def first_in_window(record):
    """The first grey failure inside the run's registered shadow window, as (frame, event), or None."""
    first, last = SHADOW_WINDOW[record["kind"]]
    frame = next((index for index in record["grey_failures"] if first <= index <= last), None)
    return None if frame is None else (frame, record["events"][frame])


def stops_in_window(record):
    first, last = SHADOW_WINDOW[record["kind"]]
    return record["stop"] is not None and first <= record["stop"]["frame"] <= last


def name(record):
    return f"{record['batch']}/{record['run_directory']}"


def score_c1(records):
    low_sun = [r for r in records if r["kind"] in SHADOW_WINDOW]
    firsts = {name(r): first_in_window(r) for r in low_sun}
    observed = {key: None if value is None else {"frame": value[0], **value[1]} for key, value in firsts.items()}
    rejected = [key for key, value in firsts.items() if value is not None and not value[1]["accept"]]
    untested = [key for key, value in firsts.items() if value is None]
    return prediction(
        "C1",
        "In each of the 6 low-sun runs the depth test accepts the first grey-check failure inside the shadow window.",
        [
            clause(
                "the first in-window grey failure is accepted in all 6 low-sun runs",
                {"first_in_window": observed, "rejected": rejected, "untested_no_grey_failure": untested},
                len(low_sun) == 6 and not rejected and not untested,
                refutes=bool(rejected),
            )
        ],
    )


def score_c2(records):
    runs = [r for r in records if r["kind"] in SHADOW_WINDOW and r["target"] == 15004]
    passes = sum(r["outcome"] == "pass" for r in runs)
    in_window = [name(r) for r in runs if stops_in_window(r)]
    on_appearance = [name(r) for r in runs if stops_in_window(r) and r["stop"]["tracker_reason"] == APPEARANCE_LOSS]
    return prediction(
        "C2",
        "At least 3 of the 4 15004 runs pass all 17 checks, and none stops on appearance in its shadow window.",
        [
            clause(
                "15004: passes, and stops in the shadow window",
                {
                    "runs": {
                        name(r): {"outcome": r["outcome"], "checks": r["checks"], "stop": r["stop"]} for r in runs
                    },
                    "passes": f"{passes} of {len(runs)}",
                    "stops_in_window": in_window,
                    "appearance_stops_in_window": on_appearance,
                },
                len(runs) == 4 and passes >= 3 and not on_appearance,
                refutes=len(in_window) >= 2 or passes < 2,
            )
        ],
    )


def score_c3(records):
    runs = [r for r in records if r["kind"] == "evening" and r["target"] == 14944]
    observed, holds, refutes = {}, len(runs) == 2, False
    for record in runs:
        accepted = next(
            (
                (frame, record["events"][frame])
                for frame in record["grey_failures"]
                if SHADOW_WINDOW["evening"][0] <= frame <= SHADOW_WINDOW["evening"][1]
                and record["events"][frame]["accept"]
            ),
            None,
        )
        first = first_in_window(record)
        stop = record["stop"]
        entry = {"first_in_window": None if first is None else first[0], "stop": stop}
        if accepted is None:
            if first is not None and stop is not None and stop["frame"] == first[0]:
                refutes = True  # it stopped there with the depth test rejecting
                entry["verdict"] = "stopped at the event with the depth test rejecting"
            else:
                entry["verdict"] = "untested: no accepted grey failure in the window"
            holds = False
        else:
            frame, event = accepted
            entry.update({"accepted_frame": frame, "strict_confidence": event["strict_confidence"]})
            stopped_there = stop is not None and stop["frame"] == frame and stop["tracker_reason"] == LOW_CONFIDENCE
            low = event["strict_confidence"] is not None and event["strict_confidence"] < MIN_CONFIDENCE
            entry["verdict"] = "low-confidence stop at the event" if stopped_there and low else "other"
            holds = holds and stopped_there and low
            if stop is None or stop["frame"] > frame:
                refutes = True  # it continued past the event
        observed[name(record)] = entry
    return prediction(
        "C3",
        "Both evening 14944 runs stop at the accepted shadow event on low_confidence (strict confidence below 0.15).",
        [clause("evening 14944: low-confidence stop at the accepted event", observed, holds, refutes=refutes)],
    )


def score_c4(records):
    controls = [r for r in records if r["kind"] == "control"]
    passed = [name(r) for r in controls if r["outcome"] == "pass"]
    accepted = {name(r): [f for f in r["grey_failures"] if r["events"][f]["accept"]] for r in controls}
    accepted = {key: value for key, value in accepted.items() if value}
    untested = [name(r) for r in controls if not r["grey_failures"]]
    table = {name(r): {"outcome": r["outcome"], "stop": r["stop"], "grey_failures": r["events"]} for r in controls}
    return prediction(
        "C4",
        "Neither control passes, and the depth test rejects every grey-check failure in both.",
        [
            clause(
                "neither control passes",
                {"runs": table, "passed": passed},
                len(controls) == 2 and not passed,
                refutes=bool(passed),
            ),
            clause(
                "the depth test rejects every grey failure in both controls",
                {"accepted": accepted, "untested_no_grey_failure": untested},
                len(controls) == 2 and not accepted and not untested,
                refutes=bool(accepted),
            ),
        ],
    )


def score_c5(records):
    runs = [r for r in records if r["kind"] == "source"]
    failed = [name(r) for r in runs if r["outcome"] != "pass"]
    return prediction(
        "C5",
        "Source-light 14944 and 15004 both pass 17/17.",
        [
            clause(
                "both source-light runs pass all 17 checks",
                {
                    "runs": {name(r): {"outcome": r["outcome"], "checks": r["checks"]} for r in runs},
                    "grey_failures_reported": {name(r): r["grey_failures"] for r in runs if r["grey_failures"]},
                },
                len(runs) == 2 and not failed,
                refutes=bool(failed),
            )
        ],
    )


def offline_comparison(record):
    """C6 for one run: the offline strict arm on the run's own frames against the live record."""
    import check_depth_loop_c0 as c0
    import replay_depth_appearance as depth_tool

    result = c0.normalized(depth_tool.replay_run(Path(record["path"])))
    strict = result["divergence"]["strict"]
    problems = []
    if not strict["equals_recording"]:
        problems.append(
            {
                "kind": "strict_differs_from_recording",
                "frames": strict["mismatch_frames"],
                "preview_fields": strict["preview_mismatch_fields"],
            }
        )
    frames = json.loads((Path(record["path"]) / "frames.json").read_text())["frames"]
    live = {
        str(f["index"]): c0.flat_event(((f.get("live_vision") or {}).get("measurement") or {}).get("depth_appearance"))
        for f in frames[: result["replayed_through_frame"] + 1]
    }
    live = {frame: event for frame, event in live.items() if event is not None}
    offline = {str(event["frame"]): event for event in result["events"]["strict"] if event["frame"] != "preview"}
    if sorted(live) != sorted(offline):
        problems.append({"kind": "event_frames", "live": sorted(live), "offline": sorted(offline)})
    for frame, event in offline.items():
        differing = c0.event_problems(event, live.get(frame))
        if differing:
            problems.append({"kind": "event", "frame": frame, "fields": differing})
    return problems


def score_c6(records, comparisons):
    mismatched = {key: value for key, value in comparisons.items() if value}
    return prediction(
        "C6",
        "Every live run equals the offline strict arm replayed on its own frames, event for event.",
        [
            clause(
                "live = offline in every run",
                {"runs_compared": len(comparisons), "mismatches": mismatched},
                len(comparisons) == len(records) == EXPECTED_RUNS and not mismatched,
                refutes=bool(mismatched),
            )
        ],
    )


def score(records, comparisons):
    return [
        score_c1(records),
        score_c2(records),
        score_c3(records),
        score_c4(records),
        score_c5(records),
        score_c6(records, comparisons),
    ]


def run_summary(record):
    return {key: value for key, value in record.items() if key not in ("facts", "path")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    import cv2

    cv2.setNumThreads(1)
    records = load(args.root)
    comparisons = {}
    for record in records:
        try:
            comparisons[name(record)] = offline_comparison(record)
        except Exception as error:  # noqa: BLE001 - a run that cannot be replayed is a mismatch, never dropped
            comparisons[name(record)] = [{"kind": "replay_error", "error": repr(error)}]
    head = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain", "--untracked-files=no"])
    )
    evidence = sorted((args.root / EVIDENCE_DIR).glob("depth-loop-*.json"))
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the depth-aware appearance check's closed-loop test (C1-C6): 10 simulator runs with "
            "D_strict + J in the live controller, the jaw surrogate casting its shadow. Simulator renders and depth; "
            "two targets and two low suns; results never pooled with unchanged-gate runs."
        ),
        "label": LABEL,
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "inputs_sha256": {str(p.relative_to(args.root)): sha256(p) for p in evidence},
        "interpretations": INTERPRETATIONS,
        "runs": [run_summary(record) for record in records],
        "predictions": score(records, comparisons),
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
